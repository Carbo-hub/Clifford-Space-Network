#pragma once

#include <cuda_runtime.h>

namespace clifford {

/**
 * @struct Mat4x4
 * @brief 4x4 real matrix representation for accelerated Clifford isomorphic operations.
 * Allows executing Clifford products directly as 64-FMA matrix multiplications on GPU.
 */
struct Mat4x4 {
    float m[4][4];

    __device__ __host__ Mat4x4() {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                m[r][c] = 0.0f;
            }
        }
    }

    __device__ __host__ static Mat4x4 identity() {
        Mat4x4 res;
        res.m[0][0] = 1.0f;
        res.m[1][1] = 1.0f;
        res.m[2][2] = 1.0f;
        res.m[3][3] = 1.0f;
        return res;
    }

    __device__ __host__ Mat4x4 operator+(const Mat4x4& o) const {
        Mat4x4 res;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                res.m[r][c] = m[r][c] + o.m[r][c];
            }
        }
        return res;
    }

    __device__ __host__ Mat4x4 operator-(const Mat4x4& o) const {
        Mat4x4 res;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                res.m[r][c] = m[r][c] - o.m[r][c];
            }
        }
        return res;
    }

    __device__ __host__ Mat4x4 operator*(float s) const {
        Mat4x4 res;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                res.m[r][c] = m[r][c] * s;
            }
        }
        return res;
    }

    /**
     * @brief Transpose: corresponds to matrix adjoint
     */
    __device__ __host__ Mat4x4 transpose() const {
        Mat4x4 res;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                res.m[r][c] = m[c][r];
            }
        }
        return res;
    }

    /**
     * @brief Frobenious norm squared
     */
    __device__ __host__ float norm_sq() const {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                sum += m[r][c] * m[r][c];
            }
        }
        return sum;
    }
};

/**
 * @brief Fully unrolled 4x4 matrix multiplication: C = A * B
 * Exactly 64 FMA operations in registers.
 */
__device__ __host__ inline Mat4x4 matmul4x4(const Mat4x4& a, const Mat4x4& b) {
    Mat4x4 c;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                sum += a.m[i][k] * b.m[k][j];
            }
            c.m[i][j] = sum;
        }
    }
    return c;
}

/**
 * @brief Skew-symmetric bivector matrix generator in 4x4 space:
 * 6 independent components corresponding to the 6 bivector planes e_ij
 */
__device__ __host__ inline Mat4x4 bivector_to_skew4x4(const float b[6]) {
    Mat4x4 s;
    // Upper triangle
    s.m[0][1] =  b[0]; // e12
    s.m[0][2] =  b[1]; // e13
    s.m[0][3] =  b[2]; // e14
    s.m[1][2] =  b[3]; // e23
    s.m[1][3] =  b[4]; // e24
    s.m[2][3] =  b[5]; // e34

    // Skew-symmetric reflection: s = -s^T
    s.m[1][0] = -b[0];
    s.m[2][0] = -b[1];
    s.m[3][0] = -b[2];
    s.m[2][1] = -b[3];
    s.m[3][1] = -b[4];
    s.m[3][2] = -b[5];

    return s;
}

/**
 * @brief Cayley Transform on 4x4 matrix space:
 * Computes Q = (I - Omega)^(-1) * (I + Omega)
 * For small Omega = (dt / 2) * B, uses closed-form rotor expansion:
 * Q = (I + Omega) * (I - Omega) / (1 + ||Omega||^2 / 4)
 */
__device__ __host__ inline Mat4x4 cayley_rotor4x4(const Mat4x4& omega) {
    Mat4x4 eye = Mat4x4::identity();
    Mat4x4 r_num = eye + omega;
    Mat4x4 r_rev = eye - omega;

    float omega_norm_sq = 0.5f * omega.norm_sq(); // Sum of squares of independent elements
    float denom = 1.0f + 0.25f * omega_norm_sq;
    float inv_denom = 1.0f / denom;

    Mat4x4 q = matmul4x4(r_num, r_rev);
    return q * inv_denom;
}

/**
 * @brief Sandwich similarity rotation: X_rot = Q * X * Q^T
 */
__device__ __host__ inline Mat4x4 rotate_state4x4(const Mat4x4& x, const Mat4x4& q) {
    Mat4x4 q_t = q.transpose();
    Mat4x4 temp = matmul4x4(q, x);
    return matmul4x4(temp, q_t);
}

} // namespace clifford
