#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include <cmath>
#include "model.h"

//
// Batch Normalization Layer
//
// This function applies batch normalization to the input tensor.
// It uses pre-computed mean and variance values for normalization.
// Inputs:
//   - input: input tensor with shape [CH][H][W]
//   - gamma: scale parameters for each channel [CH]
//   - beta: shift parameters for each channel [CH]
//   - mean: pre-computed mean values for each channel [CH]
//   - var: pre-computed variance values for each channel [CH]
//   - eps: small constant to prevent division by zero (default: 1e-5)
// Output:
//   - output: normalized and scaled tensor with shape [CH][H][W]
//
template<int CH, int H, int W>
void batchnorm_layer(
    data_t input[CH][H][W],
    data_t gamma[CH],
    data_t beta[CH],
    data_t mean[CH],
    data_t var[CH],
    data_t output[CH][H][W],
    float eps = 1e-5
) {
    // Precompute scale and bias terms for each channel
    data_t scale[CH];
    data_t bias[CH];

    for (int c = 0; c < CH; ++c) {
        data_t inv_std = 1.0f / std::sqrt(var[c] + data_t(eps));
        scale[c] = gamma[c] * inv_std;
        bias[c]  = beta[c] - scale[c] * mean[c];
    }

    // Apply normalization using only multiplication and addition
    for (int c = 0; c < CH; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                output[c][h][w] = scale[c] * input[c][h][w] + bias[c];
            }
        }
    }
}

#endif // BATCHNORM_LAYER_H
