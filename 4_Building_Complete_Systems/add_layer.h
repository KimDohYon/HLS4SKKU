#ifndef ADD_LAYER_H
#define ADD_LAYER_H

#include "model.h"

//
// Element-wise addition layer
//
// This function performs element-wise addition of two input tensors.
// Inputs:
//   - input1: first input tensor with shape [CH][H][W]
//   - input2: second input tensor with shape [CH][H][W]
// Output:
//   - output: result tensor with shape [CH][H][W], where each element is the sum of the corresponding elements in input1 and input2
//
template<int CH, int H, int W>
void add_layer(
    data_t input1[CH][H][W],
    data_t input2[CH][H][W],
    data_t output[CH][H][W]
) {
    for (int c = 0; c < CH; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[c][h][w] = input1[c][h][w] + input2[c][h][w];
            }
        }
    }
}

#endif // ADD_LAYER_H
