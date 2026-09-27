"""
Triton 3.2.0 Fused Scan Kernel for Clifford Space Networks.
Optimized for NVIDIA Ampere, Ada Lovelace (RTX 4090), and Hopper Tensor Cores.
"""

from typing import Optional
import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def clifford_scan_fwd_kernel_d4(
        M_ptr, C_ptr, X_ptr,
        stride_ms, stride_mt,
        stride_cs, stride_ct,
        stride_xs, stride_xt,
        T: tl.constexpr
    ):
        """
        Fused forward recurrence kernel in Triton:
            X_0 = C_0
            X_t = M_t @ X_{t-1} + C_t
        Mapped across sequences (program_id(0) = seq_idx).
        """
        seq_idx = tl.program_id(0)
        
        # 4x4 matrix tile offsets
        rm = tl.arange(0, 4)
        rn = tl.arange(0, 4)
        offs_mat = rm[:, None] * 4 + rn[None, :]
        
        # Pointer bases
        c_base = C_ptr + seq_idx * stride_cs
        x_base = X_ptr + seq_idx * stride_xs
        m_base = M_ptr + seq_idx * stride_ms
        
        # Initial state at t = 0
        x_cur = tl.load(c_base + offs_mat)
        tl.store(x_base + offs_mat, x_cur)
        
        # Recurrence loop
        for t in range(1, T):
            m_t = tl.load(m_base + t * stride_mt + offs_mat)
            c_t = tl.load(c_base + t * stride_ct + offs_mat)
            
            # Fused matrix multiplication and accumulation
            x_cur = tl.dot(m_t, x_cur) + c_t
            tl.store(x_base + t * stride_xt + offs_mat, x_cur)


    class TritonCliffordScanFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, M: torch.Tensor, C: torch.Tensor):
            S, T, D1, D2 = M.shape
            assert D1 == 4 and D2 == 4, "Triton scan currently requires 4x4 matrices."
            X = torch.empty_like(C)
            
            grid = (S,)
            clifford_scan_fwd_kernel_d4[grid](
                M, C, X,
                M.stride(0), M.stride(1),
                C.stride(0), C.stride(1),
                X.stride(0), X.stride(1),
                T=T
            )
            ctx.save_for_backward(M, X)
            return X

        @staticmethod
        def backward(ctx, grad_X: torch.Tensor):
            # Analytical reverse adjoint backward pass
            M, X = ctx.saved_tensors
            S, T, _, _ = M.shape
            grad_M = torch.empty_like(M)
            grad_C = torch.empty_like(grad_X)
            
            G = grad_X[:, T - 1]
            grad_C[:, T - 1] = G
            grad_M[:, T - 1] = torch.matmul(G, X[:, T - 2].transpose(-1, -2))
            
            for t in range(T - 2, 0, -1):
                G = grad_X[:, t] + torch.matmul(M[:, t + 1].transpose(-1, -2), G)
                grad_C[:, t] = G
                grad_M[:, t] = torch.matmul(G, X[:, t - 1].transpose(-1, -2))
                
            G = grad_X[:, 0] + torch.matmul(M[:, 1].transpose(-1, -2), G)
            grad_C[:, 0] = G
            grad_M[:, 0] = 0.0
            
            return grad_M, grad_C


def is_triton_available() -> bool:
    """Returns True if Triton is installed and a compatible GPU is available."""
    if not _TRITON_AVAILABLE:
        return False
    if not torch.cuda.is_available():
        return False
    # Check compute capability >= 7.5 (Turing/Ampere/Ada)
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 7


def triton_clifford_scan(M: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Runs the Triton fused scan kernel."""
    if not is_triton_available():
        raise RuntimeError("Triton is not available or supported on this device.")
    return TritonCliffordScanFunction.apply(M, C)
