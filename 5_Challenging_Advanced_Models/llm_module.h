#include <cmath>
#include <cstring>
#include <cstdint>
#include "config.h"
#include <algorithm>  // for std::max_element

// Root Mean Square Norm
template <int S>
void rmsnorm(float o[S], float x[S], float weight[S]) {
  float ss = 0.0f;
  for (int j = 0; j < S; j++) {
    ss += x[j] * x[j];
  }
  ss /= S;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  printf("rmsnorm scale = %f\n", ss);


  for (int j = 0; j < S; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}

// Softmax activation
template <int MAXSIZE>
void softmax(float *x, int size) {
  float buffer[MAXSIZE];
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }

  for (int i = 0; i < size; i++) {
    buffer[i] = expf(x[i] - max_val);
  }

  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    sum += buffer[i];
  }

  float inv_sum = 1.0f / sum;
  for (int i = 0; i < size; i++) {
    x[i] = buffer[i] * inv_sum;
  }
}

// Standard matmul for float32 weights
template <int IN, int OUT>
void matmul(float out[OUT], const float in[IN], const float weight[IN * OUT]) {
  for (int j = 0; j < OUT; j++) {
    float sum = 0.0f;
    for (int i = 0; i < IN; i++) {
      sum += in[i] * weight[j * IN + i];
    }
    out[j] = sum;
  }
}