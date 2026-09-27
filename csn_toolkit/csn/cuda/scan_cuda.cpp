#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA kernel launch wrappers
torch::Tensor scan_fwd_cuda_d4(torch::Tensor M, torch::Tensor C);
std::vector<torch::Tensor> scan_bwd_cuda_d4(torch::Tensor M, torch::Tensor X, torch::Tensor grad_X);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-Performance Fused Clifford Associative Scan CUDA Extension";
    m.def("scan_fwd_cuda_d4", &scan_fwd_cuda_d4, "Clifford Fused Forward Scan D=4 (CUDA)");
    m.def("scan_bwd_cuda_d4", &scan_bwd_cuda_d4, "Clifford Fused Backward Adjoint Scan D=4 (CUDA)");
}
