#include "model.h"
#include "cell1_weights.h"

#include "relu_layer.h"
#include "add_layer.h"
#include "concat_layer.h"
#include "conv_layer.h"
#include "batchnorm_layer.h"
#include <iostream>

void cell1(
	    data_t cells_1_preprocess0_bn_BatchNormalization_output_0[CH1][IN1_H][IN1_W],
	    data_t cells_1_preprocess1_op_op_1_Conv_output_0[CH1][IN1_H][IN1_W],
	    data_t cells_1_Concat_output_0[4 * CH1][IN1_H / 2][IN1_W / 2]
) {
#pragma HLS INTERFACE m_axi port=cells_1_preprocess0_bn_BatchNormalization_output_0 bundle=gmem0
#pragma HLS INTERFACE m_axi port=cells_1_preprocess1_op_op_1_Conv_output_0 bundle=gmem1
#pragma HLS INTERFACE m_axi port=cells_1_Concat_output_0 bundle=gmem2

#pragma HLS INTERFACE s_axilite port=cells_1_preprocess0_bn_BatchNormalization_output_0 bundle=control
#pragma HLS INTERFACE s_axilite port=cells_1_preprocess1_op_op_1_Conv_output_0 bundle=control
#pragma HLS INTERFACE s_axilite port=cells_1_Concat_output_0 bundle=control

#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Block 0.0
    static data_t cells_1_ops_by_node_0_0_op_op_0_Relu_output_0[CH1][IN1_H][IN1_W];
    static data_t cells_1_ops_by_node_0_0_op_op_1_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_0_0_op_op_2_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];

    // Block 0.1
    static data_t cells_1_ops_by_node_0_1_op_op_0_Relu_output_0[CH1][IN1_H][IN1_W];
    static data_t cells_1_ops_by_node_0_1_op_op_1_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_0_1_op_op_2_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_0_1_op_op_4_Relu_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_0_1_op_op_5_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_0_1_op_op_6_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];

    // Add output 0
    static data_t cells_1_Add_output_0[CH1][IN1_H / 2][IN1_W / 2];

    // Block 1.0
    static data_t cells_1_ops_by_node_1_0_conv1_Conv_output_0[CH1 / 2][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_1_0_conv2_Conv_output_0[CH1 / 2][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_1_0_Concat_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_1_0_bn_BatchNormalization_output_0[CH1][IN1_H / 2][IN1_W / 2];

    // Block 1.1
    static data_t cells_1_ops_by_node_1_1_op_op_0_Relu_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_1_1_op_op_1_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_1_1_op_op_2_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_1_1_op_op_4_Relu_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_1_1_op_op_5_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_1_1_op_op_6_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];

    // Add output 1
    static data_t cells_1_Add_1_output_0[CH1][IN1_H / 2][IN1_W / 2];

    // Block 2.0
    static data_t cells_1_ops_by_node_2_0_op_op_1_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_2_0_op_op_2_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];

    // Block 2.1
    static data_t cells_1_ops_by_node_2_1_op_op_0_Relu_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_2_1_op_op_1_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_2_1_op_op_2_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_2_1_op_op_4_Relu_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_2_1_op_op_5_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_2_1_op_op_6_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];

    // Add output 2
    static data_t cells_1_Add_2_output_0[CH1][IN1_H / 2][IN1_W / 2];

    // Block 3.0
    static data_t cells_1_ops_by_node_3_0_op_op_0_Relu_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_3_0_op_op_1_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_3_0_op_op_2_Conv_output_0[CH1][IN1_H / 2][IN1_W / 2];

    // Block 3.1
    static data_t cells_1_ops_by_node_3_1_conv1_Conv_output_0[CH1 / 2][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_3_1_conv2_Conv_output_0[CH1 / 2][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_3_1_Concat_output_0[CH1][IN1_H / 2][IN1_W / 2];
    static data_t cells_1_ops_by_node_3_1_bn_BatchNormalization_output_0[CH1][IN1_H / 2][IN1_W / 2];

    // Add output 3
    static data_t cells_1_Add_3_output_0[CH1][IN1_H / 2][IN1_W / 2];

//cell0//
//0.0//
    //cells.0/ops_by_node.0.0/op/op.0/Relu    
    relu_layer<CH1, IN1_H, IN1_W>(
    	cells_1_preprocess1_op_op_1_Conv_output_0, cells_1_ops_by_node_0_0_op_op_0_Relu_output_0);
    //cells.0/ops_by_node.0.0/op/op.1/Conv
    conv_layer<CH1, CH1, IN1_H, IN1_W, IN1_H/2, IN1_W/2, 5, 16>(
    	cells_1_ops_by_node_0_0_op_op_0_Relu_output_0, cells_1_ops_by_node_0_0_op_1_weight, nullptr, cells_1_ops_by_node_0_0_op_op_1_Conv_output_0, 2, 4, 16, 2);
    //cells.0/ops_by_node.0.0/op/op.2/Conv
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 1, 1>(
    	cells_1_ops_by_node_0_0_op_op_1_Conv_output_0, onnx_Conv_346, onnx_Conv_347, cells_1_ops_by_node_0_0_op_op_2_Conv_output_0, 1, 0, 1, 1);
//0.1//
    //cells.0/ops_by_node.0.1/op/op.0/Relu
    relu_layer<CH1, IN1_H, IN1_W>(
    	cells_1_preprocess0_bn_BatchNormalization_output_0, cells_1_ops_by_node_0_1_op_op_0_Relu_output_0);
    //cells.0/ops_by_node.0.1/op/op.1/Conv
    conv_layer<CH1, CH1, IN1_H, IN1_W, IN1_H/2, IN1_W/2, 5, 16>(
    	cells_1_ops_by_node_0_1_op_op_0_Relu_output_0, cells_1_ops_by_node_0_1_op_1_weight, nullptr, cells_1_ops_by_node_0_1_op_op_1_Conv_output_0, 2, 2, 16, 1);
    //cells.0/ops_by_node.0.1/op/op.2/Conv
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 1, 1>(
    	cells_1_ops_by_node_0_1_op_op_1_Conv_output_0, onnx_Conv_349, onnx_Conv_350, cells_1_ops_by_node_0_1_op_op_2_Conv_output_0, 1, 0, 1, 1);
    //cells.0/ops_by_node.0.1/op/op.4/Relu
    relu_layer<CH1, IN1_H/2, IN1_W/2>(
    	cells_1_ops_by_node_0_1_op_op_2_Conv_output_0, cells_1_ops_by_node_0_1_op_op_4_Relu_output_0);
    //cells.0/ops_by_node.0.1/op/op.5/Conv
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 5, 16>(
    	cells_1_ops_by_node_0_1_op_op_4_Relu_output_0, cells_1_ops_by_node_0_1_op_5_weight, nullptr, cells_1_ops_by_node_0_1_op_op_5_Conv_output_0, 1, 2, 16, 1);
    //cells.0/ops_by_node.0.1/op/op.6/Conv
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 1, 1>(
    	cells_1_ops_by_node_0_1_op_op_5_Conv_output_0, onnx_Conv_352, onnx_Conv_353, cells_1_ops_by_node_0_1_op_op_6_Conv_output_0, 1, 0, 1, 1);
//Add 0.0 & 0.1 
    add_layer<CH1, IN1_H/2, IN1_W/2>(
    	cells_1_ops_by_node_0_0_op_op_2_Conv_output_0, cells_1_ops_by_node_0_1_op_op_6_Conv_output_0, cells_1_Add_output_0);

//1.0//
    conv_layer<CH1, CH1/2, IN1_H, IN1_W, IN1_H/2, IN1_W/2, 1, 1>(
    	cells_1_ops_by_node_0_0_op_op_0_Relu_output_0, cells_1_ops_by_node_1_0_conv1_weight, nullptr, cells_1_ops_by_node_1_0_conv1_Conv_output_0, 2, 0, 1, 1);
    conv_layer<CH1, CH1/2, IN1_H, IN1_W, IN1_H/2, IN1_W/2, 1, 1>(
    	cells_1_ops_by_node_0_0_op_op_0_Relu_output_0, cells_1_ops_by_node_1_0_conv2_weight, nullptr, cells_1_ops_by_node_1_0_conv2_Conv_output_0, 2, 0, 1, 1);
    concat2_layer<CH1, IN1_H/2, IN1_W/2>(
    	cells_1_ops_by_node_1_0_conv1_Conv_output_0, cells_1_ops_by_node_1_0_conv2_Conv_output_0, cells_1_ops_by_node_1_0_Concat_output_0);
    batchnorm_layer<CH1, IN1_H/2, IN1_W/2>(
    	cells_1_ops_by_node_1_0_Concat_output_0, cells_1_ops_by_node_1_0_bn_weight, cells_1_ops_by_node_1_0_bn_bias, cells_1_ops_by_node_1_0_bn_running_mean, cells_1_ops_by_node_1_0_bn_running_var, cells_1_ops_by_node_1_0_bn_BatchNormalization_output_0);

//1.1//
    relu_layer<CH1, IN1_H/2, IN1_W/2>(
    	cells_1_Add_output_0, cells_1_ops_by_node_1_1_op_op_0_Relu_output_0);
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 5, 16>(
    	cells_1_ops_by_node_1_1_op_op_0_Relu_output_0, cells_1_ops_by_node_1_1_op_1_weight, nullptr, cells_1_ops_by_node_1_1_op_op_1_Conv_output_0, 1, 2, 16, 1);
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 1, 1>(
    	cells_1_ops_by_node_1_1_op_op_1_Conv_output_0, onnx_Conv_355, onnx_Conv_356, cells_1_ops_by_node_1_1_op_op_2_Conv_output_0, 1, 0, 1, 1);
    relu_layer<CH1, IN1_H/2, IN1_W/2>(
    	cells_1_ops_by_node_1_1_op_op_2_Conv_output_0, cells_1_ops_by_node_1_1_op_op_4_Relu_output_0);
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 5, 16>(
    	cells_1_ops_by_node_1_1_op_op_4_Relu_output_0, cells_1_ops_by_node_1_1_op_5_weight, nullptr, cells_1_ops_by_node_1_1_op_op_5_Conv_output_0, 1, 2, 16, 1);
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 1, 1>(
    	cells_1_ops_by_node_1_1_op_op_5_Conv_output_0, onnx_Conv_358, onnx_Conv_359, cells_1_ops_by_node_1_1_op_op_6_Conv_output_0, 1, 0, 1, 1);
    add_layer<CH1, IN1_H/2, IN1_W/2>(
    	cells_1_ops_by_node_1_0_bn_BatchNormalization_output_0, cells_1_ops_by_node_1_1_op_op_6_Conv_output_0, cells_1_Add_1_output_0);

//2.0//
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 3, 16>(
    	cells_1_ops_by_node_1_1_op_op_0_Relu_output_0, cells_1_ops_by_node_2_0_op_1_weight, nullptr, cells_1_ops_by_node_2_0_op_op_1_Conv_output_0, 1, 2, 16, 2);
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 1, 1>(
    	cells_1_ops_by_node_2_0_op_op_1_Conv_output_0, onnx_Conv_361, onnx_Conv_362, cells_1_ops_by_node_2_0_op_op_2_Conv_output_0, 1, 0, 1, 1);

//2.1//
    relu_layer<CH1, IN1_H/2, IN1_W/2>(
        cells_1_Add_1_output_0, cells_1_ops_by_node_2_1_op_op_0_Relu_output_0);
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 5, 16>(
    	cells_1_ops_by_node_2_1_op_op_0_Relu_output_0, cells_1_ops_by_node_2_1_op_1_weight, nullptr, cells_1_ops_by_node_2_1_op_op_1_Conv_output_0, 1, 2, 16, 1);
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 1, 1>(
    	cells_1_ops_by_node_2_1_op_op_1_Conv_output_0, onnx_Conv_364, onnx_Conv_365, cells_1_ops_by_node_2_1_op_op_2_Conv_output_0, 1, 0, 1, 1);
    relu_layer<CH1, IN1_H/2, IN1_W/2>(
    	cells_1_ops_by_node_2_1_op_op_2_Conv_output_0, cells_1_ops_by_node_2_1_op_op_4_Relu_output_0);
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 5, 16>(
    	cells_1_ops_by_node_2_1_op_op_4_Relu_output_0, cells_1_ops_by_node_2_1_op_5_weight, nullptr, cells_1_ops_by_node_2_1_op_op_5_Conv_output_0, 1, 2, 16, 1);
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 1, 1>(
    	cells_1_ops_by_node_2_1_op_op_5_Conv_output_0, onnx_Conv_367, onnx_Conv_368, cells_1_ops_by_node_2_1_op_op_6_Conv_output_0, 1, 0, 1, 1);
    add_layer<CH1, IN1_H/2, IN1_W/2>(
    	cells_1_ops_by_node_2_0_op_op_2_Conv_output_0, cells_1_ops_by_node_2_1_op_op_6_Conv_output_0, cells_1_Add_2_output_0);

//3.0//
    relu_layer<CH1, IN1_H/2, IN1_W/2>(
    	cells_1_Add_2_output_0, cells_1_ops_by_node_3_0_op_op_0_Relu_output_0);
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 5, 16>(
    	cells_1_ops_by_node_3_0_op_op_0_Relu_output_0, cells_1_ops_by_node_3_0_op_1_weight, nullptr, cells_1_ops_by_node_3_0_op_op_1_Conv_output_0, 1, 4, 16, 2);
    conv_layer<CH1, CH1, IN1_H/2, IN1_W/2, IN1_H/2, IN1_W/2, 1, 1>(
    	cells_1_ops_by_node_3_0_op_op_1_Conv_output_0, onnx_Conv_370, onnx_Conv_371, cells_1_ops_by_node_3_0_op_op_2_Conv_output_0, 1, 0, 1, 1);

//3.1//
    conv_layer<CH1, CH1/2, IN1_H, IN1_W, IN1_H/2, IN1_W/2, 1, 1>(
    	cells_1_ops_by_node_0_0_op_op_0_Relu_output_0, cells_1_ops_by_node_3_1_conv1_weight, nullptr, cells_1_ops_by_node_3_1_conv1_Conv_output_0, 2, 0, 1, 1);
    conv_layer<CH1, CH1/2, IN1_H, IN1_W, IN1_H/2, IN1_W/2, 1, 1>(
        cells_1_ops_by_node_0_0_op_op_0_Relu_output_0, cells_1_ops_by_node_3_1_conv2_weight, nullptr, cells_1_ops_by_node_3_1_conv2_Conv_output_0, 2, 0, 1, 1);
    concat2_layer<CH1, IN1_H/2, IN1_W/2>(
    	cells_1_ops_by_node_3_1_conv1_Conv_output_0, cells_1_ops_by_node_3_1_conv2_Conv_output_0, cells_1_ops_by_node_3_1_Concat_output_0);
    batchnorm_layer<CH1, IN1_H/2, IN1_W/2>(
    	cells_1_ops_by_node_3_1_Concat_output_0, cells_1_ops_by_node_3_1_bn_weight, cells_1_ops_by_node_3_1_bn_bias, cells_1_ops_by_node_3_1_bn_running_mean, cells_1_ops_by_node_3_1_bn_running_var, cells_1_ops_by_node_3_1_bn_BatchNormalization_output_0);
    add_layer<CH1, IN1_H/2, IN1_W/2>(
    	cells_1_ops_by_node_3_0_op_op_2_Conv_output_0, cells_1_ops_by_node_3_1_bn_BatchNormalization_output_0, cells_1_Add_3_output_0);

//final
    concat_layer<CH1, IN1_H/2, IN1_W/2>(
    	cells_1_Add_output_0, cells_1_Add_1_output_0, cells_1_Add_2_output_0, cells_1_Add_3_output_0, cells_1_Concat_output_0);
}
