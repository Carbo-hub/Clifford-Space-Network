"""
CSN CUDA Extension Loader & Autograd Integration.
Provides seamless JIT compilation and cached execution of fused Clifford scan kernels.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import torch
from torch.autograd import Function

_CUDA_MODULE = None
_CUDA_ATTEMPTED = False


def load_cuda_scan_extension():
    """
    Dynamically compiles and loads the fused Clifford scan CUDA extension.
    Uses PyTorch's load_inline or torch.utils.cpp_extension.load with persistent caching.
    """
    global _CUDA_MODULE, _CUDA_ATTEMPTED
    if _CUDA_MODULE is not None:
        return _CUDA_MODULE
    if _CUDA_ATTEMPTED:
        return None

    _CUDA_ATTEMPTED = True
    if not torch.cuda.is_available():
        return None

    try:
        from torch.utils.cpp_extension import load
        cuda_dir = Path(__file__).parent.resolve()
        cpp_src = cuda_dir / "scan_cuda.cpp"
        cu_src = cuda_dir / "scan_cuda.cu"

        if not (cpp_src.exists() and cu_src.exists()):
            return None

        # Build / load cached extension
        _CUDA_MODULE = load(
            name="csn_fast_scan_cuda",
            sources=[str(cpp_src), str(cu_src)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-Xcompiler", "-fPIC"],
            verbose=False,
        )
        return _CUDA_MODULE
    except Exception as e:
        print(f"[CSN CUDA WARNING] Failed to load CUDA fused scan extension: {e}")
        return None


class FusedCudaCliffordScanFunction(Function):
    """
    PyTorch Autograd Function backed by 100% register-resident CUDA kernels.
    Executes the entire temporal/spatial recurrence in 1 single kernel launch.
    """
    @staticmethod
    def forward(ctx, M: torch.Tensor, C: torch.Tensor):
        mod = load_cuda_scan_extension()
        if mod is None:
            raise RuntimeError("CUDA fused scan extension is not available.")
        
        M_c = M.contiguous()
        C_c = C.contiguous()
        X = mod.scan_fwd_cuda_d4(M_c, C_c)
        ctx.save_for_backward(M_c, X)
        return X

    @staticmethod
    def backward(ctx, grad_X: torch.Tensor):
        mod = load_cuda_scan_extension()
        if mod is None:
            raise RuntimeError("CUDA fused scan extension is not available.")
        
        M, X = ctx.saved_tensors
        grad_M, grad_C = mod.scan_bwd_cuda_d4(M, X, grad_X.contiguous())
        return grad_M, grad_C


def fused_cuda_clifford_scan(M: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
    High-performance entry point for fused CUDA Clifford scan.
    
    Args:
        M: [S, T, 4, 4] transition matrix sequence (Cayley rotors / damping)
        C: [S, T, 4, 4] injected multivector states
    Returns:
        X: [S, T, 4, 4] integrated state trajectory
    """
    return FusedCudaCliffordScanFunction.apply(M, C)


def is_fused_cuda_available() -> bool:
    """Returns True if the fast CUDA fused scan kernel is available and ready."""
    return load_cuda_scan_extension() is not None
