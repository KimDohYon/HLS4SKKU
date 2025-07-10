#include "typedefs.h"
#include "config.h"
#include <math.h>
#include <cstring>

extern "C" void forward(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>* transformer,
                          int token, int pos,
                          float key_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)],
                          float value_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)],
                          float* out);
