#include "clifford_algebra.cuh"
#include "clifford_tenn_cell.cuh"
#include <cuda_runtime.h>
#include <math.h>

namespace clifford {

/**
 * @brief Numerically stable softplus: log(1 + exp(x))
 */
__device__ inline float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

/**
 * @brief Numerically stable sigmoid: 1 / (1 + exp(-x))
 */
__device__ inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief CUDA Forward Kernel for Clifford-LTC / TENN Microcircuit
 * 
 * Grid:   (batch_size, 1, 1)
 * Block:  (num_neurons, 1, 1)  (num_neurons <= 64)
 */
__global__ void clifford_cfc_forward_kernel(
    CircuitConfig config,
    CliffordCellParams params,
    CircuitTensors tensors
) {
    int b = blockIdx.x; // Batch index
    int i = threadIdx.x; // Neuron index (0 .. num_neurons - 1)
    int N = config.num_neurons;

    if (b >= config.batch_size || i >= N) return;

    // Shared memory for previous states of all neurons in this batch element
    // Max 64 neurons * 16 floats * 4 bytes = 4 KB shared memory
    __shared__ float sh_X[64][16];

    // Pointer to this batch's previous state
    const float* batch_X_prev = tensors.X_prev + (b * N * 16);

    // Cooperatively load X_prev into shared memory
    #pragma unroll
    for (int c = 0; c < 16; ++c) {
        sh_X[i][c] = batch_X_prev[i * 16 + c];
    }
    __syncthreads();

    // 1. Compute Full Clifford Synaptic Interference for neuron i:
    //    S_i = sum_j (W_rec[i, j] * X_j) + sum_k (W_in[i, k] * I_t[k])
    Multivector16 S_i; // Initialized to zero

    // Recurrent Clifford Interference (All-to-All)
    const float* W_rec_row = params.W_rec + (i * N * 16);
    for (int j = 0; j < N; ++j) {
        Multivector16 w_ij;
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
            w_ij.data[c] = W_rec_row[j * 16 + c];
        }

        Multivector16 x_j;
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
            x_j.data[c] = sh_X[j][c];
        }

        // Non-commutative geometric product: W_ij * X_j
        Multivector16 prod = geometric_product(w_ij, x_j);
        S_i = S_i + prod;
    }

    // Input Clifford Projection
    if (params.W_in && tensors.I_t) {
        const float* W_in_row = params.W_in + (i * params.in_dim * 16);
        const float* batch_I  = tensors.I_t + (b * params.in_dim);

        for (int k = 0; k < params.in_dim; ++k) {
            float in_val = batch_I[k];
            if (in_val != 0.0f) {
                Multivector16 w_ik;
                #pragma unroll
                for (int c = 0; c < 16; ++c) {
                    w_ik.data[c] = W_in_row[k * 16 + c];
                }
                S_i = S_i + (w_ik * in_val);
            }
        }
    }

    // 2. Liquid Time-Constant Conductance from Scalar Grade
    float gamma_base = (params.gamma_bias) ? params.gamma_bias[i] : 0.0f;
    float gamma_i = softplus(gamma_base + S_i.data[CL4_SCALAR_IDX]);

    // 3. Torque / Rotation Generator from Bivector Grade (6 components)
    Multivector16 B_i;
    #pragma unroll
    for (int k = 0; k < CL4_BIV_COUNT; ++k) {
        float bias = (params.bivector_bias) ? params.bivector_bias[i * 6 + k] : 0.0f;
        B_i.data[CL4_BIV_START + k] = S_i.data[CL4_BIV_START + k] + bias;
    }

    // 4. Exact Isometric Cayley Transform Rotation:
    Multivector16 x_self;
    #pragma unroll
    for (int c = 0; c < 16; ++c) {
        x_self.data[c] = sh_X[i][c];
    }

    Multivector16 X_rot = cayley_rotate_cl4(x_self, B_i, tensors.dt);

    // 5. Closed-Form Continuous-Time Integration (CfC + TENN step):
    //    decay = sigmoid(-gamma * dt)
    //    X_next = decay * X_rot + (1 - decay) * S_i
    float decay = sigmoid(-gamma_i * tensors.dt);

    Multivector16 X_next;
    #pragma unroll
    for (int c = 0; c < 16; ++c) {
        X_next.data[c] = decay * X_rot.data[c] + (1.0f - decay) * S_i.data[c];
    }

    // 6. Write out results to global memory
    int out_offset = (b * N * 16) + (i * 16);

    #pragma unroll
    for (int c = 0; c < 16; ++c) {
        tensors.X_next[out_offset + c] = X_next.data[c];
        if (tensors.S_out)   tensors.S_out[out_offset + c]   = S_i.data[c];
        if (tensors.Rot_out) tensors.Rot_out[out_offset + c] = X_rot.data[c];
    }
}

// Host Launch Wrapper
cudaError_t launch_clifford_cfc_forward(
    const CircuitConfig& config,
    const CliffordCellParams& params,
    const CircuitTensors& tensors,
    cudaStream_t stream
) {
    dim3 grid(config.batch_size);
    dim3 block(config.num_neurons);

    clifford_cfc_forward_kernel<<<grid, block, 0, stream>>>(
        config,
        params,
        tensors
    );

    return cudaGetLastError();
}

} // namespace clifford
