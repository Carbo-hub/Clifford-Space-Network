#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

/**
 * @file clifford_configs.cuh
 * @brief Algebraic dimension constants, signatures, and grade metadata.
 */

namespace clifford {

// Supported Algebra Dimensions (2^n)
constexpr int DIM_COMPLEX   = 2;   // Cl(0, 1) or Cl(1, 0)
constexpr int DIM_QUAT      = 4;   // Cl+(3, 0) ~ H (quaternions / spinors)
constexpr int DIM_CL3       = 8;   // Cl(3, 0) 3D Euclidean
constexpr int DIM_CL4       = 16;  // Cl(4, 0) 4D Euclidean / Spacetime Cl(1, 3)
constexpr int DIM_CL5       = 32;  // Cl(5, 0) 5D Conformal

/**
 * @brief Grade masks and components for Cl(4, 0)
 * 16 components:
 * - Grade 0 (1):  Scalar [0]
 * - Grade 1 (4):  Vectors e1, e2, e3, e4 [1, 2, 3, 4]
 * - Grade 2 (6):  Bivectors e12, e13, e14, e23, e24, e34 [5, 6, 7, 8, 9, 10]
 * - Grade 3 (4):  Trivectors e123, e124, e134, e234 [11, 12, 13, 14]
 * - Grade 4 (1):  Pseudoscalar e1234 [15]
 */
constexpr int CL4_SCALAR_IDX = 0;
constexpr int CL4_VEC_START  = 1;
constexpr int CL4_VEC_COUNT  = 4;
constexpr int CL4_BIV_START  = 5;
constexpr int CL4_BIV_COUNT  = 6;
constexpr int CL4_TRI_START  = 11;
constexpr int CL4_TRI_COUNT  = 4;
constexpr int CL4_PSEUDO_IDX = 15;

/**
 * @brief Grade masks and components for Cl(3, 0)
 * 8 components:
 * - Grade 0 (1):  Scalar [0]
 * - Grade 1 (3):  Vectors e1, e2, e3 [1, 2, 3]
 * - Grade 2 (3):  Bivectors e12, e23, e31 [4, 5, 6]
 * - Grade 3 (1):  Pseudoscalar e123 [7]
 */
constexpr int CL3_SCALAR_IDX = 0;
constexpr int CL3_VEC_START  = 1;
constexpr int CL3_VEC_COUNT  = 3;
constexpr int CL3_BIV_START  = 4;
constexpr int CL3_BIV_COUNT  = 3;
constexpr int CL3_PSEUDO_IDX = 7;

/**
 * @brief Grade counts for Quaternions H ~ Cl+(3,0)
 * 4 components: 1 scalar + 3 bivectors (i, j, k)
 */
constexpr int QUAT_SCALAR_IDX = 0;
constexpr int QUAT_BIV_START  = 1;
constexpr int QUAT_BIV_COUNT  = 3;

/**
 * @brief Execution configuration for GPU circuits
 */
struct CircuitConfig {
    int batch_size;
    int num_neurons;       // Typically 16, 32, or 64
    int algebra_dim;       // 16 for Cl(4,0)
    int temporal_modes;    // TENN context modes (e.g. 4 or 8)
};

} // namespace clifford
