#ifndef GEMM_LAYER_H
#define GEMM_LAYER_H

#include "model.h"


//
// General Matrix Multiplication (GEMM) for 2D inputs
//
// Performs matrix multiplication with bias addition:
//   output[m][n] = sum_k(input[m][k] * weight[n][k]) + bias[n]
//
// Template Parameters:
//   - M: number of input rows
//   - N: number of output features (output columns)
//   - K: number of input features (shared dimension)
//
// Inputs:
//   - input:  input matrix of shape [M][K]
//   - weight: weight matrix of shape [N][K]
//   - bias:   bias vector of shape [N]
//
// Output:
//   - output: result matrix of shape [M][N]
//
template<int M, int N, int K>
void gemm_layer(
    data_t input[M][K],
    data_t weight[N][K],
    data_t bias[N],
    data_t output[M][N]
) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            data_t sum = bias[n];
            for (int k = 0; k < K; k++) {
                sum += input[m][k] * weight[n][k];
            }
            output[m][n] = sum;
        }
    }
}

//
// General Matrix Multiplication (GEMM) for 1D input
//
// Performs matrix-vector multiplication with bias addition:
//   output[n] = sum_k(input[k] * weight[n][k]) + bias[n]
//
// Template Parameters:
//   - N: number of output features
//   - K: number of input features
//
// Inputs:
//   - input:  input vector of shape [K]
//   - weight: weight matrix of shape [N][K]
//   - bias:   bias vector of shape [N]
//
// Output:
//   - output: result vector of shape [N]
//
template<int N, int K>
void gemm_layer_1d(
    data_t input[K],
    data_t weight[N][K],
    data_t bias[N],
    data_t output[N]
) {
    for (int n = 0; n < N; n++) {
        data_t sum = bias[n];
        for (int k = 0; k < K; k++) {
            sum += input[k] * weight[n][k];
        }
        output[n] = sum;
    }
}

#endif // GEMM_LAYER_H
