# Clifford Space Network (CSN) - Zenodo Research Artifact & Model Release (v1.0)

**Deposit Title:** Clifford Space Networks: Sub-Quadratic Non-Abelian Lie Rotor Scans for Geometric Sequence Modeling and Physical Systems  
**Author:** Mykyta Nosenko  
**Affiliation:** National Technical University "Kharkiv Polytechnic Institute" (NTU KhPI), Kharkiv, Ukraine  
**ORCID:** [0009-0006-1976-1557](https://orcid.org/0009-0006-1976-1557)  
**License:** Apache-2.0  
**DOI:** Pending Zenodo Registration  

---

## 1. Overview & Research Scope

This archive constitutes the open reproducible research deposit for the **Clifford Space Network (CSN)** framework. CSN replaces conventional non-linear activation layers and discrete matrix recurrence with continuous state-space trajectories governed by multivector bivector generators $\mathfrak{spin}(p, q)$ under adaptive dynamic step scaling $\Delta t(t)$.

> **Research Prototype Notice**: This release represents an **early-stage empirical feasibility prototype (v1.0)**. The empirical results demonstrate theoretical feasibility and specific niche advantages under defined conditions. Claims of overarching superiority across standard enterprise workloads are strictly avoided. Future revisions (v2.0+) will focus on scaling verification across larger multi-gigabyte datasets and broad multi-modal benchmarks.

---

## 2. Directory Layout & Archive Contents

```text
├── clifford_space_network.pdf          # Full 9-page research manuscript
├── README.md                           # Main project documentation and benchmark tables
├── README_ZENODO.md                    # This archive manifest and usage guide
├── LICENSE                             # Apache-2.0 open-source license
├── .zenodo.json                        # Zenodo repository deposition metadata
├── checkpoints/                        # Pre-trained PyTorch model weights (.pt)
│   ├── csn_v3_deep_unleash_best.pt     # Vision: MOT16 detector (tau=0.25, F1=65.06%) [21.4 MB]
│   ├── csn_v3_mot16_transfer_best.pt   # Vision: Post-COCO fine-tuned weights (Prec=76.2%) [10.8 MB]
│   ├── csn_v3_coco_best.pt             # Vision: Pre-trained COCO backbone [21.3 MB]
│   └── csn_lm_cl8_master.pt            # Language: Master Cl(8,0) neuron model (PPL=5.46) [~1.8 MB]
├── python/                             # Core Python/PyTorch implementations
│   ├── clifford_cl8.py                 # Monolithic Cl(8,0) 256-D neuron engine
│   ├── clifford_continuous.py          # Continuous Lie algebra dynamic integrator
│   ├── csn_v3_unified.py               # Complete CSN-V3 vision architecture
│   └── ...                             # Mathematical baselines and utilities
├── cuda/                               # Native CUDA / C++ kernel extensions
│   ├── clifford_cuda_kernel.cu         # Lie bracket and bivector commutator kernels
│   └── ...
├── tests/                              # Reproduction and evaluation scripts
│   ├── benchmark_cl8_scaling.py        # Cl(4,0) vs Cl(6,0) vs Cl(8,0) scaling benchmark
│   ├── benchmark_cl8_compact.py        # Single-neuron Pareto efficiency test
│   ├── evaluate_mot16_precision.py     # MOT16 precision/recall evaluation harness
│   └── ...
└── docs/paper/                         # LaTeX sources and bibliography for the manuscript
```

---

## 3. Pre-trained Checkpoint Summary

> **Weights Hosting**: All pre-trained PyTorch checkpoints listed below are hosted on [Google Drive](https://drive.google.com/drive/folders/1jzsZBEl-FwJkyJkHERvsGA2E3M-lVQeF?usp=sharing).

| File Name | Domain | Architecture Details | Parameters | Target Metric |
| :--- | :--- | :--- | :---: | :--- |
| `csn_v3_deep_unleash_best.pt` | Computer Vision | CSN-V3 MOT16 Deep Detector | 5.34M | F1 = 65.06% ($\tau=0.25$), Recall = 70.0% ($\tau=0.15$) |
| `csn_v3_mot16_transfer_best.pt` | Computer Vision | CSN-V3 Transfer-Trained on MOT16 | 2.69M | Precision = 76.2% ($\tau=0.25$) |
| `csn_v3_coco_best.pt` | Computer Vision | CSN-V3 COCO Pre-trained Backbone | 5.34M | Multi-scale geometric representation |
| `csn_lm_cl8_master.pt` | Language Modeling | Monolithic $\mathcal{Cl}(8, 0)$ Master Neuron | 158K | Val PPL = 5.46, Accuracy = 50.50% on TinyShakespeare |

---

## 4. Environment & Dependencies

To execute inference or benchmarks, Python 3.10+ and PyTorch 2.0+ with CUDA support are recommended:

```bash
pip install torch torchvision numpy scipy
```

Optional GPU compilation of custom CUDA kernels:
```bash
cd cuda
nvcc -O3 -shared -Xcompiler -fPIC clifford_cuda_kernel.cu -o libclifford_cuda.so
```

---

## 5. Verification & Reproduction Steps

### A. Language Modeling Benchmark & Checkpoint Evaluation
To evaluate the $\mathcal{Cl}(8, 0)$ master neuron model on continuous sequence modeling:
```bash
python tests/benchmark_cl8_scaling.py
```

### B. Computer Vision MOT16 Evaluation
To evaluate detection precision, recall, and F1 curves across classification thresholds $\tau \in [0.15, 0.40]$:
```bash
python tests/evaluate_mot16_precision.py --weights checkpoints/csn_v3_deep_unleash_best.pt
```

---

## 6. Operating Boundaries & Comparative Trade-offs

| Dimension | Clifford Space Network (CSN) | Standard Baselines (Transformer / SSM / Mamba) |
| :--- | :--- | :--- |
| **State Interaction Density** | $\mathcal{O}(D^3)$ Lie-algebraic coupling in associative algebra | $\mathcal{O}(D^2)$ standard linear state space coupling |
| **Transition Compute** | $\mathcal{O}(D^2)$ matrix commutator / bivector rotation | $\mathcal{O}(D)$ diagonalized recurrence / structured state spaces |
| **Advantage Conditions** | Ultra-dense high-dimensional interactions per neuron, parameter-constrained dynamic systems | Raw linear throughput on ultra-large sequence contexts without non-linear interaction |
| **Maturity Status** | Early-stage research prototype | Industrially mature, hardware-specialized compiler support |

---

## 7. Citation

```bibtex
@article{nosenko2026clifford,
  title={Clifford Space Network: Continuous Lie-Algebraic State Evolution in Multivector Clifford Algebras},
  author={Nosenko, Mykyta},
  journal={arXiv preprint / Zenodo Deposit},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.PENDING}
}
```
