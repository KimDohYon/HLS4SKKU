#include <stdint.h>
#include <stdio.h>

#ifndef TYPEDEFS
#define TYPEDEFS

struct Config {
  int dim;
  int hidden_dim;
  int n_layers;
  int n_heads;
  int n_kv_heads;
  int vocab_size;
  int seq_len;
  int GS; // retained for compatibility, unused in float32
};

template <int IN, int OUT>
struct Linear {
  float weight[IN * OUT];
};

template <int SIZE>
struct FloatTensor {
  float weight[SIZE];
};

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
struct TransformerWeights {
  float token_embedding_table[vocab_size * dim];

  float rms_att_weight[n_layers * dim];
  float rms_ffn_weight[n_layers * dim];
  float rms_final_weight[dim];

  Linear<dim, dim> wq[n_layers];
  Linear<dim, dim> wk[n_layers];
  Linear<dim, dim> wv[n_layers];
  Linear<dim, dim> wo[n_layers];

  Linear<dim, hidden_dim> w1[n_layers];
  Linear<dim, hidden_dim> w3[n_layers];
  Linear<hidden_dim, dim> w2[n_layers];

  FloatTensor<dim * vocab_size> wcls[1];
};

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
struct Transformer {
  Config config;
  TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> weights;
};

#endif