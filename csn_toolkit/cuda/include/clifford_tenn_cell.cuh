#pragma once

#include "clifford_configs.cuh"
#include "clifford_algebra.cuh"
#include "clifford_mat4.cuh"
#include <cuda_runtime.h>

namespace clifford {

/**
 * @struct CliffordCellParams
 * @brief Memory pointers for Clifford-LTC / TENN recurrent circuit weights
 */
struct CliffordCellParams {
    const float* W_rec;         // [num_neurons, num_neurons, 16] - recurrent Clifford synapses
    const float* W_in;          // [num_neurons, in_dim, 16]      - input Clifford projection
    const float* gamma_bias;    // [num_neurons]                  - baseline leak conductance
    const float* bivector_bias; // [num_neurons, 6]               - baseline bivector torque
    int in_dim;                 // Dimension of input vector
};

/**
 * @struct CircuitTensors
 * @brief State tensors for forward propagation
 */
struct CircuitTensors {
    const float* X_prev;        // [batch_size, num_neurons, 16] - input state at t
    const float* I_t;           // [batch_size, in_dim]          - external input at t
    float* X_next;              // [batch_size, num_neurons, 16] - updated state at t + dt
    float* S_out;               // [batch_size, num_neurons, 16] - intermediate interference field (for backward)
    float* Rot_out;             // [batch_size, num_neurons, 16] - rotated state (for backward)
    float dt;                   // Continuous time step
};

/**
 * @struct CircuitGrads
 * @brief Gradient tensors for backward pass
 */
struct CircuitGrads {
    const float* dL_dX_next;    // [batch_size, num_neurons, 16] - incoming loss gradient
    float* dL_dX_prev;          // [batch_size, num_neurons, 16] - gradient w.r.t. previous state
    float* dL_dI_t;             // [batch_size, in_dim]          - gradient w.r.t. input
    float* dL_dW_rec;           // [num_neurons, num_neurons, 16] - gradient w.r.t. recurrent weights
    float* dL_dW_in;            // [num_neurons, in_dim, 16]      - gradient w.r.t. input weights
    float* dL_dgamma_bias;      // [num_neurons]                  - gradient w.r.t. gamma bias
    float dt;
};

// Host API Declarations
cudaError_t launch_clifford_cfc_forward(
    const CircuitConfig& config,
    const CliffordCellParams& params,
    const CircuitTensors& tensors,
    cudaStream_t stream = 0
);

cudaError_t launch_clifford_cfc_backward(
    const CircuitConfig& config,
    const CliffordCellParams& params,
    const CircuitTensors& fwd_tensors,
    const CircuitGrads& grads,
    cudaStream_t stream = 0
);

} // namespace clifford
