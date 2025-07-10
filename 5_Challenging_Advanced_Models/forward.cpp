#include "typedefs.h"
#include "config.h"
#include <math.h>
#include <cstring>
#include "llm_module.h"

extern "C" void forward(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>* transformer,
                        int token, int pos,
                        float key_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)],
                        float value_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)],
                        float* out) {
#pragma HLS INTERFACE m_axi port = transformer offset = slave bundle = gmem0 depth=1
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem1 depth=32000

  auto w = &transformer->weights;
  static float x[dim];
  static float xb[dim];
  static float xb2[dim];
  static float hb[hidden_dim];
  static float hb2[hidden_dim];
  static float q[dim];
  static float k[(dim * n_kv_heads) / n_heads];
  static float v[(dim * n_kv_heads) / n_heads];
  static float att[n_heads * seq_len];

  constexpr int kv_dim = (dim * n_kv_heads) / n_heads;
  constexpr int kv_mul = n_heads / n_kv_heads;
  constexpr int head_size = dim / n_heads;

  std::memcpy(x, w->token_embedding_table + token * dim, sizeof(float) * dim);
  printf("max(x) = %f\n", *std::max_element(x, x + dim));

  for (int l = 0; l < n_layers; l++) {
    rmsnorm<dim>(xb, x, w->rms_att_weight + l * dim);
    matmul<dim, dim>(q, xb, w->wq[l].weight);
    matmul<dim, kv_dim>(k, xb, w->wk[l].weight);
    matmul<dim, kv_dim>(v, xb, w->wv[l].weight);

    for (int i = 0; i < kv_dim; i += 2) {
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      float v0 = q[i], v1 = q[i + 1];
      q[i] = v0 * fcr - v1 * fci;
      q[i + 1] = v0 * fci + v1 * fcr;

      v0 = k[i], v1 = k[i + 1];
      k[i] = v0 * fcr - v1 * fci;
      k[i + 1] = v0 * fci + v1 * fcr;
    }

    int loff = l * seq_len * kv_dim;
    float* key_cache_row = key_cache + loff + pos * kv_dim;
    float* value_cache_row = value_cache + loff + pos * kv_dim;
    std::memcpy(key_cache_row, k, kv_dim * sizeof(float));
    std::memcpy(value_cache_row, v, kv_dim * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
      int q_offset = h * head_size;
      int att_offset = h * seq_len;

      for (int t = 0; t <= pos; t++) {
        int k_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
          score += q[q_offset + i] * key_cache[k_offset + i];
        }
        score /= sqrtf(head_size);
        att[att_offset + t] = score;
      }

      softmax<seq_len>(att + att_offset, pos + 1);

      int xb_offset = h * head_size;
      std::memset(xb + xb_offset, 0, sizeof(float) * head_size);
      for (int t = 0; t <= pos; t++) {
        int v_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
        float a = att[att_offset + t];
        for (int i = 0; i < head_size; i++) {
          xb[xb_offset + i] += a * value_cache[v_offset + i];
        }
      }
    }

    matmul<dim, dim>(xb2, xb, w->wo[l].weight);
    for (int i = 0; i < dim; i++) {
      x[i] += xb2[i];
    }

    rmsnorm<dim>(xb, x, w->rms_ffn_weight + l * dim);
    matmul<dim, hidden_dim>(hb, xb, w->w1[l].weight);
    matmul<dim, hidden_dim>(hb2, xb, w->w3[l].weight);
    for (int i = 0; i < hidden_dim; i++) {
      float val = hb[i];
      val *= (1.0f / (1.0f + expf(-val)));
      hb[i] = val * hb2[i];
    }
    matmul<hidden_dim, dim>(xb, hb, w->w2[l].weight);
    for (int i = 0; i < dim; i++) {
      x[i] += xb[i];
    }
  }

  rmsnorm<dim>(xb, x, w->rms_final_weight);
  matmul<dim, vocab_size>(out, xb, w->wcls[0].weight);  // 여기서 xb 사용
}
