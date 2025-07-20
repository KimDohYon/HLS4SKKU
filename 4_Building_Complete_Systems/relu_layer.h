#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include <cstddef>
#include <ap_fixed.h>
#include <hls_math.h>
#include "model.h"


//
// ReLU Activation Layer
//
// Applies the Rectified Linear Unit activation function:
//   output = max(0, input)
//
// Template Parameters:
//   - CH: number of channels
//   - H:  height of feature map
//   - W:  width of feature map
//
// Input:
//   - input:  input tensor of shape [CH][H][W]
//
// Output:
//   - output: output tensor of shape [CH][H][W], where each value is max(0, input)
//
template<int CH, int H, int W>
void relu_layer(
    data_t input[CH][H][W],
    data_t output[CH][H][W]
) {
    for (int c = 0; c < CH; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[c][h][w] = (input[c][h][w] > data_t(0)) ? input[c][h][w] : data_t(0);
            }
        }
    }
}

#endif // RELU_LAYER_H
