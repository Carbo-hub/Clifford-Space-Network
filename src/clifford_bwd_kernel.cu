#include "clifford_algebra.cuh"
#include "clifford_tenn_cell.cuh"
#include <cuda_runtime.h>
#include <math.h>

namespace clifford {

/**
 * @brief Numerically stable sigmoid: 1 / (1 + exp(-x))
 */
__device__ inline float bwd_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief CUDA Backward Kernel for Clifford-LTC / TENN Circuit
 * Computes exact adjoint gradients w.r.t states, weights, and continuous-time parameters.
 */
__global__ void clifford_cfc_backward_kernel(
    CircuitConfig config,
    CliffordCellParams params,
    CircuitTensors fwd_tensors,
    CircuitGrads grads
) {
    int b = blockIdx.x; // Batch index
    int i = threadIdx.x; // Neuron index
    int N = config.num_neurons;

    if (b >= config.batch_size || i >= N) return;

    int offset_i = (b * N * 16) + (i * 16);

    // 1. Load incoming gradient dL / dX_next for neuron i
    Multivector16 dL_dX_next;
    #pragma unroll
    for (int c = 0; c < 16; ++c) {
        dL_dX_next.data[c] = grads.dL_dX_next[offset_i + c];
    }

    // Load forward cached values
    Multivector16 S_i;
    Multivector16 X_rot;
    #pragma unroll
    for (int c = 0; c < 16; ++c) {
        S_i.data[c]   = fwd_tensors.S_out[offset_i + c];
        X_rot.data[c] = fwd_tensors.Rot_out[offset_i + c];
    }

    // Recompute forward liquid decay
    float gamma_base = (params.gamma_bias) ? params.gamma_bias[i] : 0.0f;
    float gamma_arg = gamma_base + S_i.data[CL4_SCALAR_IDX];
    float gamma_i = (gamma_arg > 20.0f) ? gamma_arg : log1pf(expf(gamma_arg));
    float decay = bwd_sigmoid(-gamma_i * grads.dt);

    // 2. Gradients through state interpolation:
    //    X_next = decay * X_rot + (1 - decay) * S_i
    Multivector16 dL_dX_rot = dL_dX_next * decay;
    Multivector16 dL_dS_i   = dL_dX_next * (1.0f - decay);

    // Gradient w.r.t decay scalar: sum_c (dL_dX_next[c] * (X_rot[c] - S_i[c]))
    float dL_d_decay = 0.0f;
    #pragma unroll
    for (int c = 0; c < 16; ++c) {
        dL_d_decay += dL_dX_next.data[c] * (X_rot.data[c] - S_i.data[c]);
    }

    // Gradient through decay = sigmoid(-gamma * dt):
    float d_decay_d_arg = decay * (1.0f - decay);
    float dL_d_gamma = dL_d_decay * d_decay_d_arg * (-grads.dt);

    // Gradient through softplus: d(softplus(x))/dx = sigmoid(x)
    float d_gamma_d_arg = bwd_sigmoid(gamma_arg);
    float dL_d_gamma_in = dL_d_gamma * d_gamma_d_arg;

    // Accumulate to scalar component of S_i
    dL_dS_i.data[CL4_SCALAR_IDX] += dL_d_gamma_in;

    if (grads.dL_dgamma_bias && b == 0) {
        atomicAdd(&grads.dL_dgamma_bias[i], dL_d_gamma_in);
    }

    // 3. Adjoint through Isometric Cayley Rotation:
    //    Because Cayley rotation is an exact isometry (unitary),
    //    the adjoint w.r.t the rotated multivector X_prev is simply
    //    rotation in the opposite direction (-B_i)!
    Multivector16 B_i;
    #pragma unroll
    for (int k = 0; k < CL4_BIV_COUNT; ++k) {
        float bias = (params.bivector_bias) ? params.bivector_bias[i * 6 + k] : 0.0f;
        B_i.data[CL4_BIV_START + k] = S_i.data[CL4_BIV_START + k] + bias;
    }

    Multivector16 neg_B = B_i * (-1.0f);
    Multivector16 dL_dX_self_rot = cayley_rotate_cl4(dL_dX_rot, neg_B, grads.dt);

    // Write gradient w.r.t self state
    #pragma unroll
    for (int c = 0; c < 16; ++c) {
        atomicAdd(&grads.dL_dX_prev[offset_i + c], dL_dX_self_rot.data[c]);
    }

    // 4. Gradient through Synaptic Interference: S_i = sum_j (W_rec[i, j] * X_j)
    //    Adjoint of geometric product:
    //    If S = W * X, then:
    //    dL/dX = W_rev * dL/dS
    //    dL/dW = dL/dS * X_rev
    for (int j = 0; j < N; ++j) {
        const float* W_rec_ij = params.W_rec + (i * N * 16) + (j * 16);
        Multivector16 w_ij;
        #pragma unroll
        for (int c = 0; c < 16; ++c) w_ij.data[c] = W_rec_ij[c];

        Multivector16 x_j;
        int offset_j = (b * N * 16) + (j * 16);
        #pragma unroll
        for (int c = 0; c < 16; ++c) x_j.data[c] = fwd_tensors.X_prev[offset_j + c];

        // Gradient to previous state of neuron j
        Multivector16 dL_dX_j = geometric_product(w_ij.reversion(), dL_dS_i);
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
            atomicAdd(&grads.dL_dX_prev[offset_j + c], dL_dX_j.data[c]);
        }

        // Gradient to synaptic weight W_rec[i, j]
        if (grads.dL_dW_rec) {
            Multivector16 dL_dW_ij = geometric_product(dL_dS_i, x_j.reversion());
            int w_offset = (i * N * 16) + (j * 16);
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                atomicAdd(&grads.dL_dW_rec[w_offset + c], dL_dW_ij.data[c]);
            }
        }
    }
}

// Host Launch Wrapper
cudaError_t launch_clifford_cfc_backward(
    const CircuitConfig& config,
    const CliffordCellParams& params,
    const CircuitTensors& fwd_tensors,
    const CircuitGrads& grads,
    cudaStream_t stream
) {
    dim3 grid(config.batch_size);
    dim3 block(config.num_neurons);

    clifford_cfc_backward_kernel<<<grid, block, 0, stream>>>(
        config,
        params,
        fwd_tensors,
        grads
    );

    return cudaGetLastError();
}

} // namespace clifford
