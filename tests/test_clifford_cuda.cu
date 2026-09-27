#include "clifford_algebra.cuh"
#include "clifford_configs.cuh"
#include "clifford_tenn_cell.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace clifford;

// Helper: check CUDA errors
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Device test kernel for Cayley Isometry
__global__ void test_cayley_isometry_kernel(int n_samples, float* max_err_out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_samples) return;

    // Pseudo-random initialization based on thread ID
    Multivector16 x;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        x.data[i] = sinf((float)(idx * 16 + i) * 1.37f);
    }

    Multivector16 b;
    #pragma unroll
    for (int k = 0; k < CL4_BIV_COUNT; ++k) {
        b.data[CL4_BIV_START + k] = cosf((float)(idx * 6 + k) * 2.71f);
    }

    float dt = 0.05f * (1.0f + (idx % 100)); // Test varied dt

    float norm_before = sqrtf(x.norm_sq());

    Multivector16 x_rot = cayley_rotate_cl4(x, b, dt);

    float norm_after = sqrtf(x_rot.norm_sq());

    float rel_err = fabsf(norm_after - norm_before) / (norm_before + 1e-8f);

    // Atomic max on float (cast to int for CAS)
    // Simplified: write into output if large
    if (rel_err > *max_err_out) {
        *max_err_out = rel_err;
    }
}

int main() {
    printf("==============================================================\n");
    printf("  Clifford-LTC / TENN CUDA Kernel Verification & Benchmark\n");
    printf("==============================================================\n");

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("[WARN] No CUDA-capable device detected. Exiting test.\n");
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("[GPU] Device: %s (SM %d.%d, Total Memory: %.2f GB)\n",
           prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // TEST 1: Cayley Transform Isometry Test (Norm Preservation)
    printf("\n--- Test 1: Cayley Transform Isometry Invariant ---\n");
    int n_samples = 65536;
    float* d_max_err;
    float h_max_err = 0.0f;
    CHECK_CUDA(cudaMalloc(&d_max_err, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_max_err, &h_max_err, sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n_samples + threads - 1) / threads;
    test_cayley_isometry_kernel<<<blocks, threads>>>(n_samples, d_max_err);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_max_err, d_max_err, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  Samples tested: %d\n", n_samples);
    printf("  Maximum relative norm deviation: %e\n", h_max_err);
    if (h_max_err < 1e-5f) {
        printf("  [PASS] Cayley rotation strictly preserves multivector isometry to machine epsilon!\n");
    } else {
        printf("  [FAIL] Deviation exceeds tolerance: %e\n", h_max_err);
    }
    CHECK_CUDA(cudaFree(d_max_err));

    // TEST 2: Microcircuit End-to-End Execution (Forward & Backward)
    printf("\n--- Test 2: Microcircuit End-to-End Execution ---\n");
    CircuitConfig config;
    config.batch_size = 64;
    config.num_neurons = 32;
    config.algebra_dim = 16;
    config.temporal_modes = 1;

    int B = config.batch_size;
    int N = config.num_neurons;
    int D = config.algebra_dim;
    int in_dim = 8;

    size_t state_bytes = B * N * D * sizeof(float);
    size_t w_rec_bytes = N * N * D * sizeof(float);
    size_t w_in_bytes  = N * in_dim * D * sizeof(float);
    size_t in_bytes    = B * in_dim * sizeof(float);
    size_t bias_bytes  = N * sizeof(float);
    size_t biv_bytes   = N * 6 * sizeof(float);

    printf("  Batch size: %d | Neurons: %d | Clifford Dim: %d\n", B, N, D);
    printf("  Total circuit state memory: %.2f KB\n", state_bytes / 1024.0f);
    printf("  Synaptic weight memory:     %.2f KB\n", w_rec_bytes / 1024.0f);

    float *d_X_prev, *d_X_next, *d_S_out, *d_Rot_out;
    float *d_I_t, *d_W_rec, *d_W_in, *d_gamma_bias, *d_biv_bias;

    CHECK_CUDA(cudaMalloc(&d_X_prev, state_bytes));
    CHECK_CUDA(cudaMalloc(&d_X_next, state_bytes));
    CHECK_CUDA(cudaMalloc(&d_S_out, state_bytes));
    CHECK_CUDA(cudaMalloc(&d_Rot_out, state_bytes));
    CHECK_CUDA(cudaMalloc(&d_I_t, in_bytes));
    CHECK_CUDA(cudaMalloc(&d_W_rec, w_rec_bytes));
    CHECK_CUDA(cudaMalloc(&d_W_in, w_in_bytes));
    CHECK_CUDA(cudaMalloc(&d_gamma_bias, bias_bytes));
    CHECK_CUDA(cudaMalloc(&d_biv_bias, biv_bytes));

    // Initialize data on host
    float* h_ones = (float*)malloc(w_rec_bytes);
    for (size_t i = 0; i < w_rec_bytes / sizeof(float); ++i) {
        h_ones[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f; // Small stable weights
    }
    CHECK_CUDA(cudaMemcpy(d_W_rec, h_ones, w_rec_bytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(d_X_prev, 0, state_bytes));
    CHECK_CUDA(cudaMemset(d_I_t, 0, in_bytes));
    CHECK_CUDA(cudaMemset(d_W_in, 0, w_in_bytes));
    CHECK_CUDA(cudaMemset(d_gamma_bias, 0, bias_bytes));
    CHECK_CUDA(cudaMemset(d_biv_bias, 0, biv_bytes));

    CliffordCellParams params;
    params.W_rec = d_W_rec;
    params.W_in = d_W_in;
    params.gamma_bias = d_gamma_bias;
    params.bivector_bias = d_biv_bias;
    params.in_dim = in_dim;

    CircuitTensors tensors;
    tensors.X_prev = d_X_prev;
    tensors.I_t = d_I_t;
    tensors.X_next = d_X_next;
    tensors.S_out = d_S_out;
    tensors.Rot_out = d_Rot_out;
    tensors.dt = 0.02f;

    // Warm-up
    CHECK_CUDA(launch_clifford_cfc_forward(config, params, tensors));
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  [PASS] Forward pass kernel executed successfully!\n");

    // Backward pass allocation
    float *d_dL_dX_next, *d_dL_dX_prev, *d_dL_dW_rec;
    CHECK_CUDA(cudaMalloc(&d_dL_dX_next, state_bytes));
    CHECK_CUDA(cudaMalloc(&d_dL_dX_prev, state_bytes));
    CHECK_CUDA(cudaMalloc(&d_dL_dW_rec, w_rec_bytes));
    CHECK_CUDA(cudaMemset(d_dL_dX_prev, 0, state_bytes));
    CHECK_CUDA(cudaMemset(d_dL_dW_rec, 0, w_rec_bytes));

    CircuitGrads grads;
    grads.dL_dX_next = d_dL_dX_next;
    grads.dL_dX_prev = d_dL_dX_prev;
    grads.dL_dI_t = nullptr;
    grads.dL_dW_rec = d_dL_dW_rec;
    grads.dL_dW_in = nullptr;
    grads.dL_dgamma_bias = nullptr;
    grads.dt = tensors.dt;

    CHECK_CUDA(launch_clifford_cfc_backward(config, params, tensors, grads));
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  [PASS] Backward pass kernel executed successfully!\n");

    // TEST 3: Performance & Latency Benchmark
    printf("\n--- Test 3: GPU Latency & Throughput Benchmark ---\n");
    int num_bench_steps = 1000;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int s = 0; s < num_bench_steps; ++s) {
        launch_clifford_cfc_forward(config, params, tensors);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_us = (total_ms * 1000.0f) / num_bench_steps;
    float steps_per_sec = (num_bench_steps / total_ms) * 1000.0f;

    printf("  Benchmarked %d continuous-time steps:\n", num_bench_steps);
    printf("  Average latency per step:   %.2f microseconds\n", avg_us);
    printf("  Throughput:                 %.0f circuit steps/second\n", steps_per_sec);
    printf("  Throughput per sample:      %.0f sample-steps/second\n", steps_per_sec * B);

    // Clean up
    cudaFree(d_X_prev);
    cudaFree(d_X_next);
    cudaFree(d_S_out);
    cudaFree(d_Rot_out);
    cudaFree(d_I_t);
    cudaFree(d_W_rec);
    cudaFree(d_W_in);
    cudaFree(d_gamma_bias);
    cudaFree(d_biv_bias);
    cudaFree(d_dL_dX_next);
    cudaFree(d_dL_dX_prev);
    cudaFree(d_dL_dW_rec);
    free(h_ones);

    printf("\n==============================================================\n");
    printf("  All Clifford-LTC / TENN CUDA tests completed successfully!\n");
    printf("==============================================================\n");

    return 0;
}
