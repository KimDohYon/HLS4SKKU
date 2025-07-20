#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include "model.h"
#include "external_weights.h"
#include "cell1_weights.h"
#include "conv_layer.h"
#include "relu_layer.h"
#include "global_pooling.h"
#include "add_layer.h"
#include "concat_layer.h"
#include "batchnorm_layer.h"
#include "gemm_layer.h"

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "xcl2.hpp"

#define CHECK_CALL(err, msg) \
    if (err != CL_SUCCESS) { std::cerr << msg << " Error: " << err << std::endl; exit(EXIT_FAILURE); }

std::vector<unsigned char> read_binary_file(const std::string& xclbin_file_name) {
    std::ifstream file(xclbin_file_name, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open " << xclbin_file_name << std::endl;
        exit(EXIT_FAILURE);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "Failed to read " << xclbin_file_name << std::endl;
        exit(EXIT_FAILURE);
    }
    return buffer;
}

void read_input_data(const char* filename, data_t input[2][32][3072]) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Unable to open input file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    for (int c = 0; c < 2; c++)
        for (int h = 0; h < 32; h++)
            for (int w = 0; w < 3072; w++)
                file.read(reinterpret_cast<char*>(&input[c][h][w]), sizeof(float));
    file.close();
}

void print_inference_results(data_t output[13]) {
    float max_val = -std::numeric_limits<float>::infinity();
    float sum_exp = 0.0f;
    float probs[13];
    int max_class = -1;
    float max_prob = -1.0f;

    // Find max for numerical stability in softmax
    for (int i = 0; i < 13; ++i)
        max_val = std::max(max_val, output[i]);

    // Compute exponentials and their sum
    for (int i = 0; i < 13; ++i) {
        probs[i] = std::exp(output[i] - max_val);
        sum_exp += probs[i];
    }

    std::cout << "\n=== Inference Results ===\n";
    
    // Print probabilities and determine the max class
    for (int i = 0; i < 13; ++i) {
        float prob = probs[i] / sum_exp;
        std::cout << "Class " << i << ": " << std::fixed << std::setprecision(4) << prob * 100 << "%\n";
        if (prob > max_prob) {
            max_prob = prob;
            max_class = i;
        }
    }

    std::cout << "Predicted class: " << max_class << "\n";
}

int main(int argc, char** argv) {
    static data_t input[2][32][3072];
    static data_t stem_out[12][32][3072];
    static data_t relu_out[12][32][3072];
    static data_t conv0_out[8][32][3072];
    static data_t conv1_out[8][32][3072];
    static data_t cell0_out[32][16][1536];
    static data_t relu1_out[32][16][1536];
    static data_t conv1_0_out[16][16][1536];
    static data_t conv1_1_out[16][16][1536];
    static data_t concat1_out[16][16][1536];
    static data_t bn_out[16][16][1536];
    static data_t conv1_final[16][16][1536];
    static data_t cell1_out[64][8][768];
    static data_t global_out[64];
    static data_t output[13];

    read_input_data("train06_trimmed.fc32", input);

    conv_layer<2, 12, 32, 3072, 32, 3072, 3, 1>(
        input, onnx_Conv_307, onnx_Conv_308, stem_out, 1, 1, 1, 1);
    relu_layer<12, 32, 3072>(stem_out, relu_out);
    conv_layer<12, 8, 32, 3072, 32, 3072, 1, 1>(
        relu_out, onnx_Conv_310, onnx_Conv_311, conv0_out, 1, 0, 1, 1);
    conv_layer<12, 8, 32, 3072, 32, 3072, 1, 1>(
        relu_out, onnx_Conv_313, onnx_Conv_314, conv1_out, 1, 0, 1, 1);

    std::string binaryFile = "drone.xclbin";
    auto devices = xcl::get_xil_devices();
    auto device = devices[0];
    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    auto fileBuf = read_binary_file(binaryFile);
    cl::Program::Binaries bins;
    bins.push_back(std::make_pair(fileBuf.data(), fileBuf.size()));
    cl::Program program(context, {device}, bins);
    cl::Kernel kernel(program, "cell0");

    size_t in_size = CH0 * IN0_H * IN0_W;
    size_t out_size = 4 * CH0 * (IN0_H / 2) * (IN0_W / 2);

    cl::Buffer buf_in0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * in_size, conv0_out);
    cl::Buffer buf_in1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * in_size, conv1_out);
    cl::Buffer buf_out(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(data_t) * out_size, cell0_out);

    kernel.setArg(0, buf_in0);
    kernel.setArg(1, buf_in1);
    kernel.setArg(2, buf_out);

    q.enqueueMigrateMemObjects({buf_in0, buf_in1}, 0);

    cl::Event event;
    q.enqueueTask(kernel, nullptr, &event);

    q.enqueueMigrateMemObjects({buf_out}, CL_MIGRATE_MEM_OBJECT_HOST);

    q.finish();

    cl_ulong start_time, end_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);

    double duration_ms = (end_time - start_time) * 1e-6;  // 나노초 → 밀리초
    std::cout << "Kernel execution time: " << std::fixed << std::setprecision(3) << duration_ms << " ms\n";


    relu_layer<32, 16, 1536>(cell0_out, relu1_out);
    conv_layer<32, 16, 16, 1536, 16, 1536, 1, 1>(
        relu1_out, onnx_Conv_343, onnx_Conv_344, conv1_final, 1, 0, 1, 1);

    conv_layer<12, 8, 32, 3072, 16, 1536, 1, 1>(
        relu_out, cells_1_preprocess0_conv1_weight, nullptr, conv1_0_out, 2, 0, 1, 1);
    conv_layer<12, 8, 32, 3072, 16, 1536, 1, 1>(
        relu_out, cells_1_preprocess0_conv2_weight, nullptr, conv1_1_out, 2, 0, 1, 1);
    concat2_layer<16, 16, 1536>(conv1_0_out, conv1_1_out, concat1_out);
    batchnorm_layer<16, 16, 1536>(
        concat1_out, cells_1_preprocess0_bn_weight, cells_1_preprocess0_bn_bias,
        cells_1_preprocess0_bn_running_mean, cells_1_preprocess0_bn_running_var, bn_out);

    cell1(bn_out, conv1_final, cell1_out);

    global_avg_pool<64, 8, 768>(cell1_out, global_out);
    gemm_layer_1d<13, 64>(global_out, classifier_weight, classifier_bias, output);

    print_inference_results(output);
    return 0;
}
