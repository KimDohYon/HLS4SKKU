// LLaMA2 Vitis HLS C Simulation Testbench (HEAP-safe)
#include "forward.h"
#include "config.h"
#include <iostream>
#include <cstring>
#include <string>
#include <fstream>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>

struct TokenIndex {
    char *str;
    int id;
};

struct Tokenizer {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
};

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, std::string tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;

    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    FILE *file = fopen(tokenizer_path.c_str(), "rb");
    if (!file) { std::cerr << "Failed to open tokenizer.\n"; exit(1); }
    fread(&t->max_token_length, sizeof(int), 1, file);
    int len;
    for (int i = 0; i < vocab_size; i++) {
        fread(t->vocab_scores + i, sizeof(float), 1, file);
        fread(&len, sizeof(int), 1, file);
        t->vocab[i] = (char *)malloc(len + 1);
        fread(t->vocab[i], len, 1, file);
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

char *decode(Tokenizer *t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size) return (char *)"[INVALID]";
    return t->vocab[token];
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = {.str = str};
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { std::cerr << "NULL text\n"; exit(1); }
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char *str_buffer = (char *)malloc((t->max_token_length * 2 + 3));
    size_t str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup((char *)" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) tokens[(*n_tokens)++] = id;
        else {
            for (int i = 0; i < str_len; i++)
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
        }
        str_len = 0;
    }

    while (true) {
        float best_score = -1e10;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

inline void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

// === Forward declaration for build_transformer ===
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void build_transformer(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *t, std::string path);

// === Main function ===
int main() {
    auto *transformer = new Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>();
    Tokenizer tokenizer;

    build_tokenizer(&tokenizer, "tokenizer.bin", vocab_size);
    build_transformer(transformer, "model.bin");

    const char *prompt = "My name is";
    int *prompt_tokens = (int *)malloc(seq_len * sizeof(int));
    int num_prompt_tokens = 0;
    encode(&tokenizer, (char *)prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    float *key_cache = (float *)calloc(n_layers * seq_len * ((dim * n_kv_heads) / n_heads), sizeof(float));
    float *value_cache = (float *)calloc(n_layers * seq_len * ((dim * n_kv_heads) / n_heads), sizeof(float));
    float *logits = (float *)calloc(vocab_size, sizeof(float));

    int token = prompt_tokens[0];
    int pos = 0;
    int next = 0;
    int steps = 10;

    std::cout << "Input: " << prompt << std::endl;
    std::cout << "Output: " << std::endl;

    for (int i = 0; i < steps; i++) {
        forward(transformer, token, pos, key_cache, value_cache, logits);
        softmax(logits, vocab_size);
        
        for (int j = 0; j < 10; j++) {
            printf("logits[%d] = %f\n", j, logits[j]);
        }

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            float max_val = logits[0];
            int max_idx = 0;
            for (int j = 1; j < vocab_size; j++) {
                if (logits[j] > max_val) {
                    max_val = logits[j];
                    max_idx = j;
                }
            }
            printf("Predicted token index: %d\n", max_idx); // 🔍 추가

            next = max_idx;
        }

        char *piece = decode(&tokenizer, token, next);
        std::cout << piece << std::endl;  // 이 줄이 없거나 주석처리된 듯 보임

        if (next == 2) break;
        std::cout << piece << std::endl;
        token = next;
        pos++;
    }

    std::cout << std::endl;

    delete transformer;
    free(prompt_tokens);
    free(key_cache);
    free(value_cache);
    free(logits);

    return 0;
}

// === Simplified float32 weight loader ===
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void build_transformer(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *t, std::string path) {
    FILE *file = fopen(path.c_str(), "rb");
    if (!file) { std::cerr << "Failed to open weight file: " << path << std::endl; exit(1); }

    uint32_t magic;
    fread(&magic, sizeof(uint32_t), 1, file);
    if (magic != 0x616b3432) { std::cerr << "Invalid magic number" << std::endl; exit(1); }

    int version;
    fread(&version, sizeof(int), 1, file);
    if (version != 1) { std::cerr << "Unsupported version: " << version << std::endl; exit(1); }

    fread(&t->config, sizeof(Config), 1, file);
    fseek(file, 256, SEEK_SET); // skip padded header

    fread(t->weights.token_embedding_table, sizeof(float), vocab_size * dim, file);
    fread(t->weights.rms_att_weight, sizeof(float), n_layers * dim, file);
    fread(t->weights.rms_ffn_weight, sizeof(float), n_layers * dim, file);
    fread(t->weights.rms_final_weight, sizeof(float), dim, file);

    for (int i = 0; i < n_layers; i++) {
        fread(t->weights.wq[i].weight, sizeof(float), dim * dim, file);
        fread(t->weights.wk[i].weight, sizeof(float), dim * dim, file);
        fread(t->weights.wv[i].weight, sizeof(float), dim * dim, file);
        fread(t->weights.wo[i].weight, sizeof(float), dim * dim, file);
        fread(t->weights.w1[i].weight, sizeof(float), dim * hidden_dim, file);
        fread(t->weights.w3[i].weight, sizeof(float), dim * hidden_dim, file);
        fread(t->weights.w2[i].weight, sizeof(float), hidden_dim * dim, file);
    }

    fread(t->weights.wcls[0].weight, sizeof(float), dim * vocab_size, file);

    // shared_classifier가 true일 경우 token_embedding_table 복사
    bool shared_classifier = true; // export.py version 1에서는 항상 공유됨
    if (shared_classifier) {
        std::memcpy(t->weights.wcls[0].weight, t->weights.token_embedding_table, sizeof(float) * dim * vocab_size);
    }
    printf("wcls[0]: %f, token_embedding[0]: %f\n", t->weights.wcls[0].weight[0], t->weights.token_embedding_table[0]);

    fclose(file);
}
