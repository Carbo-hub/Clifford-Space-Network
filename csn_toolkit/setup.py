import os
from setuptools import setup, find_packages

ext_modules = []
cmdclass = {}

# Check if CUDA is available for compiling native extensions
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    if torch.cuda.is_available():
        ext_modules.append(
            CUDAExtension(
                name="csn.cuda.csn_fast_scan_cuda",
                sources=[
                    "csn/cuda/scan_cuda.cpp",
                    "csn/cuda/scan_cuda.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3", "--use_fast_math", "-Xcompiler", "-fPIC"],
                }
            )
        )
        cmdclass["build_ext"] = BuildExtension
except Exception as e:
    print(f"[CSN SETUP] Warning: Building without pre-compiled CUDA extensions: {e}")

setup(
    name="csn_toolkit",
    version="1.1.0",
    description="Clifford Space Networks (CSN): Lie Algebra Rotors & Associative Scans with Hardware CUDA Kernels",
    author="Antigravity / Science AI Team",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "triton": ["triton>=3.0.0"],
        "cv": ["opencv-python", "albumentations"],
    }
)
