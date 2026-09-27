#pragma once

#include "clifford_configs.cuh"
#include <cuda_runtime.h>
#include <math.h>

namespace clifford {

/**
 * @brief Canonical bitmask mapping for Cl(4, 0)
 * Maps consecutive indices 0..15 to canonical basis bitmasks.
 */
__device__ __host__ constexpr uint8_t CL4_BLADE_BITS[16] = {
    0x0, // 0:  1
    0x1, // 1:  e1
    0x2, // 2:  e2
    0x4, // 3:  e3
    0x8, // 4:  e4
    0x3, // 5:  e12
    0x5, // 6:  e13
    0x9, // 7:  e14
    0x6, // 8:  e23
    0xa, // 9:  e24
    0xc, // 10: e34
    0x7, // 11: e123
    0xb, // 12: e124
    0xd, // 13: e134
    0xe, // 14: e234
    0xf  // 15: e1234
};

/**
 * @brief Reverse lookup: bitmask to canonical index 0..15
 */
__device__ __host__ constexpr uint8_t get_blade_index_cl4(uint8_t mask) {
    switch (mask) {
        case 0x0: return 0;
        case 0x1: return 1;
        case 0x2: return 2;
        case 0x4: return 3;
        case 0x8: return 4;
        case 0x3: return 5;
        case 0x5: return 6;
        case 0x9: return 7;
        case 0x6: return 8;
        case 0xa: return 9;
        case 0xc: return 10;
        case 0x7: return 11;
        case 0xb: return 12;
        case 0xd: return 13;
        case 0xe: return 14;
        case 0xf: return 15;
        default:  return 0;
    }
}

/**
 * @brief Compute sign for multiplying two canonical bitmasks in Cl(4, 0) (Euclidean e_i^2 = +1)
 */
__device__ __host__ constexpr float compute_blade_sign_cl4(uint8_t a, uint8_t b) {
    // Count swaps needed to reorder basis vectors canonically
    int swaps = 0;
    for (int i = 0; i < 4; ++i) {
        if (a & (1 << i)) {
            // Count bits in b with index less than i
            for (int j = 0; j < i; ++j) {
                if (b & (1 << j)) {
                    swaps++;
                }
            }
        }
    }
    return (swaps % 2 == 0) ? 1.0f : -1.0f;
}

/**
 * @struct Multivector16
 * @brief 16-component multivector representing elements of Cl(4, 0)
 */
struct Multivector16 {
    float data[16];

    __device__ __host__ Multivector16() {
        #pragma unroll
        for (int i = 0; i < 16; ++i) data[i] = 0.0f;
    }

    __device__ __host__ static Multivector16 zero() {
        return Multivector16();
    }

    __device__ __host__ static Multivector16 scalar(float s) {
        Multivector16 m;
        m.data[CL4_SCALAR_IDX] = s;
        return m;
    }

    __device__ __host__ float& operator[](int idx) { return data[idx]; }
    __device__ __host__ const float& operator[](int idx) const { return data[idx]; }

    __device__ __host__ Multivector16 operator+(const Multivector16& o) const {
        Multivector16 res;
        #pragma unroll
        for (int i = 0; i < 16; ++i) res.data[i] = data[i] + o.data[i];
        return res;
    }

    __device__ __host__ Multivector16 operator-(const Multivector16& o) const {
        Multivector16 res;
        #pragma unroll
        for (int i = 0; i < 16; ++i) res.data[i] = data[i] - o.data[i];
        return res;
    }

    __device__ __host__ Multivector16 operator*(float s) const {
        Multivector16 res;
        #pragma unroll
        for (int i = 0; i < 16; ++i) res.data[i] = data[i] * s;
        return res;
    }

    /**
     * @brief Clifford Reversion involution: reverse order of basis vectors.
     * Grades 0, 1, 4: sign +1
     * Grades 2, 3:    sign -1
     */
    __device__ __host__ Multivector16 reversion() const {
        Multivector16 r;
        r.data[0] = data[0]; // Scalar (+)
        #pragma unroll
        for (int i = 1; i <= 4; ++i) r.data[i] = data[i]; // Vectors (+)
        #pragma unroll
        for (int i = 5; i <= 10; ++i) r.data[i] = -data[i]; // Bivectors (-)
        #pragma unroll
        for (int i = 11; i <= 14; ++i) r.data[i] = -data[i]; // Trivectors (-)
        r.data[15] = data[15]; // Pseudoscalar (+)
        return r;
    }

    /**
     * @brief Extract squared Euclidean norm of the multivector: sum of squares
     */
    __device__ __host__ float norm_sq() const {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) sum += data[i] * data[i];
        return sum;
    }

    /**
     * @brief Extract bivector squared norm: sum of squares of grade 2 components
     */
    __device__ __host__ float bivector_norm_sq() const {
        float sum = 0.0f;
        #pragma unroll
        for (int i = CL4_BIV_START; i < CL4_BIV_START + CL4_BIV_COUNT; ++i) {
            sum += data[i] * data[i];
        }
        return sum;
    }
};

/**
 * @brief Pure Clifford Geometric Product for Cl(4, 0): C = A * B
 * Completely unrolled into 256 multiply-accumulate operations.
 */
__device__ __host__ inline Multivector16 geometric_product(const Multivector16& a, const Multivector16& b) {
    Multivector16 c;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint8_t mask_a = CL4_BLADE_BITS[i];
        float val_a = a.data[i];
        if (val_a == 0.0f) continue;

        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            uint8_t mask_b = CL4_BLADE_BITS[j];
            float val_b = b.data[j];
            if (val_b == 0.0f) continue;

            uint8_t target_mask = mask_a ^ mask_b;
            uint8_t target_idx = get_blade_index_cl4(target_mask);
            float sign = compute_blade_sign_cl4(mask_a, mask_b);

            c.data[target_idx] += sign * val_a * val_b;
        }
    }
    return c;
}

/**
 * @brief Fast Rotor-Sandwich Cayley Transform for Cl(4, 0)
 * Rotates state X by bivector B over time step dt with exact norm preservation.
 * 
 * Formula:
 *   mu = dt / 2
 *   R_num = (1 + mu * B)
 *   R_rev = (1 - mu * B)
 *   denom = 1 + mu^2 * ||B||^2
 *   X_rot = (R_num * X * R_rev) / denom
 */
__device__ __host__ inline Multivector16 cayley_rotate_cl4(
    const Multivector16& x,
    const Multivector16& b,
    float dt
) {
    float mu = 0.5f * dt;

    // Construct R_num = 1 + mu * B (scalar=1, bivectors=mu*b)
    Multivector16 r_num;
    r_num.data[CL4_SCALAR_IDX] = 1.0f;
    #pragma unroll
    for (int i = CL4_BIV_START; i < CL4_BIV_START + CL4_BIV_COUNT; ++i) {
        r_num.data[i] = mu * b.data[i];
    }

    // Construct R_rev = 1 - mu * B
    Multivector16 r_rev;
    r_rev.data[CL4_SCALAR_IDX] = 1.0f;
    #pragma unroll
    for (int i = CL4_BIV_START; i < CL4_BIV_START + CL4_BIV_COUNT; ++i) {
        r_rev.data[i] = -mu * b.data[i];
    }

    // Compute Denominator: 1 + mu^2 * ||B||^2
    float b_norm_sq = b.bivector_norm_sq();
    float denom = 1.0f + (mu * mu) * b_norm_sq;
    float inv_denom = 1.0f / denom;

    // Numerator: (R_num * X) * R_rev
    Multivector16 temp = geometric_product(r_num, x);
    Multivector16 num  = geometric_product(temp, r_rev);

    // Exact isometric result
    return num * inv_denom;
}

} // namespace clifford
