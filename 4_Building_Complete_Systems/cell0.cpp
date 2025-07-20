#include "model.h"
#include "cell0_weights.h"

#include "relu_layer.h"
#include "add_layer.h"
#include "concat_layer.h"
#include "maxpool_layer.h"
#include "conv_layer.h"
#include "batchnorm_layer.h"
#include <iostream>

extern "C" void cell0(
    data_t cells_0_preprocess0_op_op_1_Conv_output_0[CH0][IN0_H][IN0_W],
    data_t cells_0_preprocess1_op_op_1_Conv_output_0[CH0][IN0_H][IN0_W],
    data_t cells_0_Concat_output_0[4 * CH0][IN0_H / 2][IN0_W / 2]
) {

    static data_t relu_buf[2][CH0][IN0_H][IN0_W];
    static data_t relu_half_buf[2][CH0][IN0_H / 2][IN0_W / 2];
    static data_t conv_buf[4][CH0][IN0_H / 2][IN0_W / 2];
    static data_t add_buf[4][CH0][IN0_H / 2][IN0_W / 2];
    static data_t concat_buf[2][CH0 / 2][IN0_H / 2][IN0_W / 2];
    static data_t concat_out[CH0][IN0_H / 2][IN0_W / 2];

    // 0.0
    relu_layer<CH0, IN0_H, IN0_W>(cells_0_preprocess1_op_op_1_Conv_output_0, relu_buf[0]);
    conv_layer<CH0, CH0, IN0_H, IN0_W, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_buf[0], cells_0_ops_by_node_0_0_op_1_weight, nullptr, conv_buf[0], 2, 4, 8, 2);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[0], onnx_Conv_316, onnx_Conv_317, conv_buf[1], 1, 0, 1, 1);

    // 0.1
    relu_layer<CH0, IN0_H, IN0_W>(cells_0_preprocess0_op_op_1_Conv_output_0, relu_buf[1]);
    conv_layer<CH0, CH0, IN0_H, IN0_W, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_buf[1], cells_0_ops_by_node_0_1_op_1_weight, nullptr, conv_buf[2], 2, 2, 8, 1);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[2], onnx_Conv_319, onnx_Conv_320, conv_buf[3], 1, 0, 1, 1);
    relu_layer<CH0, IN0_H / 2, IN0_W / 2>(conv_buf[3], relu_half_buf[0]);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_half_buf[0], cells_0_ops_by_node_0_1_op_5_weight, nullptr, conv_buf[2], 1, 2, 8, 1);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[2], onnx_Conv_322, onnx_Conv_323, conv_buf[3], 1, 0, 1, 1);
    add_layer<CH0, IN0_H / 2, IN0_W / 2>(conv_buf[1], conv_buf[3], add_buf[0]);

    // 1.0
    conv_layer<CH0, CH0 / 2, IN0_H, IN0_W, IN0_H / 2, IN0_W / 2, 1, 1>(
        relu_buf[0], cells_0_ops_by_node_1_0_conv1_weight, nullptr, concat_buf[0], 2, 0, 1, 1);
    conv_layer<CH0, CH0 / 2, IN0_H, IN0_W, IN0_H / 2, IN0_W / 2, 1, 1>(
        relu_buf[0], cells_0_ops_by_node_1_0_conv2_weight, nullptr, concat_buf[1], 2, 0, 1, 1);
    concat2_layer<CH0, IN0_H / 2, IN0_W / 2>(concat_buf[0], concat_buf[1], concat_out);
    batchnorm_layer<CH0, IN0_H / 2, IN0_W / 2>(
        concat_out, cells_0_ops_by_node_1_0_bn_weight, cells_0_ops_by_node_1_0_bn_bias,
        cells_0_ops_by_node_1_0_bn_running_mean, cells_0_ops_by_node_1_0_bn_running_var, conv_buf[0]);

    // 1.1
    relu_layer<CH0, IN0_H / 2, IN0_W / 2>(add_buf[0], relu_half_buf[0]);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_half_buf[0], cells_0_ops_by_node_1_1_op_1_weight, nullptr, conv_buf[1], 1, 2, 8, 1);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[1], onnx_Conv_325, onnx_Conv_326, conv_buf[2], 1, 0, 1, 1);
    relu_layer<CH0, IN0_H / 2, IN0_W / 2>(conv_buf[2], relu_half_buf[1]);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_half_buf[1], cells_0_ops_by_node_1_1_op_5_weight, nullptr, conv_buf[1], 1, 2, 8, 1);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[1], onnx_Conv_328, onnx_Conv_329, conv_buf[2], 1, 0, 1, 1);
    add_layer<CH0, IN0_H / 2, IN0_W / 2>(conv_buf[0], conv_buf[2], add_buf[1]);

    // 2.0
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 3, 8>(
        relu_half_buf[0], cells_0_ops_by_node_2_0_op_1_weight, nullptr, conv_buf[0], 1, 2, 8, 2);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[0], onnx_Conv_331, onnx_Conv_332, conv_buf[1], 1, 0, 1, 1);

    // 2.1
    relu_layer<CH0, IN0_H / 2, IN0_W / 2>(add_buf[1], relu_half_buf[0]);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_half_buf[0], cells_0_ops_by_node_2_1_op_1_weight, nullptr, conv_buf[2], 1, 2, 8, 1);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[2], onnx_Conv_334, onnx_Conv_335, conv_buf[3], 1, 0, 1, 1);
    relu_layer<CH0, IN0_H / 2, IN0_W / 2>(conv_buf[3], relu_half_buf[1]);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_half_buf[1], cells_0_ops_by_node_2_1_op_5_weight, nullptr, conv_buf[2], 1, 2, 8, 1);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[2], onnx_Conv_337, onnx_Conv_338, conv_buf[3], 1, 0, 1, 1);
    add_layer<CH0, IN0_H / 2, IN0_W / 2>(conv_buf[1], conv_buf[3], add_buf[2]);

    // 3.0
    relu_layer<CH0, IN0_H / 2, IN0_W / 2>(add_buf[2], relu_half_buf[0]);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_half_buf[0], cells_0_ops_by_node_3_0_op_1_weight, nullptr, conv_buf[0], 1, 4, 8, 2);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[0], onnx_Conv_340, onnx_Conv_341, conv_buf[1], 1, 0, 1, 1);

    // 3.1
    conv_layer<CH0, CH0 / 2, IN0_H, IN0_W, IN0_H / 2, IN0_W / 2, 1, 1>(
        relu_buf[0], cells_0_ops_by_node_3_1_conv1_weight, nullptr, concat_buf[0], 2, 0, 1, 1);
    conv_layer<CH0, CH0 / 2, IN0_H, IN0_W, IN0_H / 2, IN0_W / 2, 1, 1>(
        relu_buf[0], cells_0_ops_by_node_3_1_conv2_weight, nullptr, concat_buf[1], 2, 0, 1, 1);
    concat2_layer<CH0, IN0_H / 2, IN0_W / 2>(concat_buf[0], concat_buf[1], concat_out);
    batchnorm_layer<CH0, IN0_H / 2, IN0_W / 2>(
        concat_out, cells_0_ops_by_node_3_1_bn_weight, cells_0_ops_by_node_3_1_bn_bias,
        cells_0_ops_by_node_3_1_bn_running_mean, cells_0_ops_by_node_3_1_bn_running_var, conv_buf[2]);
    add_layer<CH0, IN0_H / 2, IN0_W / 2>(conv_buf[1], conv_buf[2], add_buf[3]);

    // concat
    concat_layer<CH0, IN0_H / 2, IN0_W / 2>(
        add_buf[0], add_buf[1], add_buf[2], add_buf[3], cells_0_Concat_output_0);
}
