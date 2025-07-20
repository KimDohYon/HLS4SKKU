// external_weights.h

#ifndef EXTERNAL_WEIGHTS_H
#define EXTERNAL_WEIGHTS_H

#include "model.h" // for `data_t` type

//===== Stem 0 =====
extern data_t onnx_Conv_307[12][2][3][3];
extern data_t onnx_Conv_308[12];

//===== Cell 0 Preprocess =====
extern data_t onnx_Conv_310[8][12][1][1];
extern data_t onnx_Conv_311[8];
extern data_t onnx_Conv_313[8][12][1][1];
extern data_t onnx_Conv_314[8];

// ===== Preprocess =====
extern data_t cells_1_preprocess0_bn_bias[16];
extern data_t cells_1_preprocess0_bn_running_mean[16];
extern data_t cells_1_preprocess0_bn_running_var[16];
extern data_t cells_1_preprocess0_bn_weight[16];
extern data_t cells_1_preprocess0_conv1_weight[8][12][1][1];
extern data_t cells_1_preprocess0_conv2_weight[8][12][1][1];
extern data_t onnx_Conv_343[16][32][1][1];
extern data_t onnx_Conv_344[16];

// ===== Classifier =====
extern data_t classifier_bias[13];
extern data_t classifier_weight[13][64];

#endif // EXTERNAL_WEIGHTS_H