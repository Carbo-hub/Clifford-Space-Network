# Clifford Space Networks (CSN)
### Sub-Quadratic Non-Abelian Lie Rotor Scans for Geometric Sequence Modeling and Physical Systems

[![arXiv](https://img.shields.io/badge/arXiv-2609.xxxxx-b31b1b.svg)](docs/paper/clifford_space_network.pdf)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-ee4c2c.svg)](https://pytorch.org/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--1976-1557-green.svg)](https://orcid.org/0009-0006-1976-1557)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)

---

## 📌 Executive Summary (v1 Feasibility Study)

**Clifford Space Networks (CSN)** formulate an analytical geometric computing paradigm that elevates state representations into **graded Clifford multivector algebras** $\mathcal{Cl}(p, q)$ and parameterizes recurrent state transitions via the continuous **non-abelian Spinor Lie group** $\mathrm{Spin}(n)$. 

Unlike existing Clifford neural networks designed as static PDE surrogates for 3D physics, CSN is architected primarily as an **analytical solver for non-commutative logic graphs and physical Hamiltonian systems**, enabling exact sub-quadratic state transitions ($\mathcal{O}(L \log L)$ parallel associative scan, $\mathcal{O}(1)$ autoregressive generation memory).

**Current v1 Status & Feasibility Proofs**:
- **Dynamic Sequence Modeling (CSN-LM)**: Validated on Tiny Shakespeare, reaching 50.50% Top-1 accuracy with **$19.7\times$ fewer parameters** (158k vs. 1.90M) and **$1.54\times$ faster token generation** at $L=2048$ with $\mathcal{O}(1)$ memory ($\approx 21$ MB). Conversely, standard Transformers maintain higher raw parallel training token throughput via vendor BLAS GEMMs, and diagonal SSMs (Mamba) have lower per-step arithmetic complexity. Large-scale evaluations on BabyLM 10M/100M and TinyStories are underway to evaluate regimes where attention mechanisms traditionally scale.
- **Static Spatial Perception (CSN-V3)**: Validated on MOT16 pedestrian detection with **$1.98\times$ fewer parameters** than YOLO26n and **70.0% pedestrian recall** when trained from scratch on limited domain data. When extensive MS COCO pretraining is available, however, calibrated YOLO26n achieves higher overall F1 (73.60% vs. 65.06%) and lower forward latency (46.78 ms vs. 54.98 ms). Hierarchical scaling to ImageNet-1k is under active development.
- **Interconnect Bottleneck Diagnosis**: We identify a core architectural challenge: the *interconnect bottleneck* between information-dense multivector neurons (up to 256 degrees of freedom) and flat Euclidean linear projection weights.
- **Structured Development Roadmap**: Version 1 establishes the mathematical feasibility proof; Version 2 focuses strictly on empirical scaling and benchmark formation; Port-Hamiltonian Interconnections (PHS) and Orbifold Geometries ($\mathcal{M}/\Gamma$) constitute long-term theoretical horizons.

```
                           Foundational Architectural Comparison
┌──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
│ Technical Property   │ Transformers (GPT)   │ State Space (Mamba)  │ Liquid (CfC / LTC)   │ Clifford Space (CSN) │
├──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Time Complexity      │ O(L^2) (Quadratic)   │ O(L) (Linear)        │ O(L) (Linear Recurr) │ O(L log L) / O(L)    │
│ Generation KV Memory │ O(L) (Cache Explode) │ O(1) (Fixed State)   │ O(1) (Fixed State)   │ O(1) (Fixed Multivec)│
│ State Algebra        │ Flat Dot-Products    │ Commutative Diagonal │ Scalar Euclidean ODE │ Non-Abelian Spin(n)  │
│ Continuous Pacing    │ Discrete Only        │ Discretized ZOH      │ Exponential Decay    │ Adaptive Δt(t) Lie   │
│ Energy Invariance    │ Unbounded Softmax    │ Decaying Norm        │ Decaying Norm        │ Isometric Sandwich   │
│ Parameter Footprint  │ 19.7x larger         │ 2.9x larger          │ 2.4x larger          │ Ultra-Compact (1.0x) │
└──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘
```

---

## 🔬 Distinction from Prior Clifford Neural Networks

The integration of Clifford algebras into deep learning has gained traction through Clifford Neural Layers for PDE modeling (Brandstetter et al., 2022/2023), Geometric Clifford Algebra Networks (Ruhe et al., 2023), and Geometric Algebra Transformers (Brehmer et al., 2023). However, foundational architectural differences set CSN apart:

1. **Target Modality**: Prior Clifford models focus on static spatial equivariance ($O(n), E(n)$) or continuous PDE surrogates (fluid dynamics, weather). CSN addresses **sub-quadratic dynamic sequence modeling and state space transitions**.
2. **Transition Mechanism**: While prior works utilize static multivector weight multiplication ($W \star x$), CSN formulates recurrent state updates via the **rational Cayley transform** on Lie algebras $\mathfrak{so}(n) \to \mathrm{Spin}(n)$.
3. **Associative Parallel Scans**: Prior multivector architectures do not formulate causal temporal recurrences or require $\mathcal{O}(L^2)$ attention (as in GATr). CSN derives an exact affine closed form admitting a **parallel prefix scan in $\mathcal{O}(L \log L)$** with register-level fused GPU kernels.

---

## 🌐 Primary Orientation: Analytical Graph and Physical Solvers

While standard deep learning architectures approximate tasks through statistical pattern matching on unconstrained Euclidean tensors, Clifford Space Networks were conceived as an **analytical engine**:

1. **Analytical Logic Graph Solvers**: In formal logic, causal dependency networks, and knowledge graphs, relations are non-commutative and structured. By mapping entities to multivectors and relations to Spinor Lie rotors $R_{ij} \in \mathrm{Spin}(n)$, logical inference along graph paths reduces to exact algebraic rotor composition without geometric distortion.
2. **Physical System Modeling and Gauge Invariants**: Physical conservation laws (Hamiltonian mechanics, Maxwell's electrodynamics $F = E + I c B$, and gauge field theories) are natively expressed via differential forms and bivectors. CSN operates directly on the Lie group $\mathrm{Spin}(n)$ with strict algebraic norm preservation, providing a natural inductive bias for physical dynamical systems.
3. **Sequence and Spatial Perception**: Compact, sub-quadratic representation of temporal sequence order and spatial orientations without quadratic attention overhead.

---

## ⚡ Mathematical Distinction: Exact Clifford Closed Form vs. Heuristic CfC

Closed-form Continuous-time (CfC) networks approximate continuous ODE dynamics via empirical heuristic bounding functions on flat coordinates:
$$h_t^{\text{CfC}} = \sigma(-f(x_t) \Delta t) \odot \tanh(W_1 x_t) + (1 - \sigma(-f(x_t) \Delta t)) \odot \tanh(W_2 x_t)$$
Because recurrent states are coupled inside non-linear saturating functions ($\tanh$), **CfC cannot be formulated as an associative matrix operation**, strictly constraining its training to slow sequential loops.

In contrast, CSN derives an **exact algebraic closed form** on the Lie group $\mathrm{Spin}(n)$ via the **rational Cayley transform** on the Lie algebra $\mathfrak{so}(n)$:
$$\Omega_t = \frac{1}{2} B_t \Delta t, \quad R_t = (I + \Omega_t)(I - \Omega_t)^{-1} \in \mathrm{Spin}(n)$$
$$h_t = \sqrt{\alpha_t} R_t h_{t-1} + (1 - \alpha_t) S_t \equiv M_t h_{t-1} + C_t$$
**Theoretical Properties of Clifford Closed Form:**
- **Exact Lie Group Isometry**: The Cayley transform is an exact algebraic bijection from $\mathfrak{so}(n)$ onto $\mathrm{Spin}(n)$, guaranteeing $\|R_t v\| = \|v\|$ and unit spectral radius by geometric construction (no vanishing/exploding gradients).
- **Strict Associativity**: $(M_j, C_j) \circ (M_i, C_i) = (M_j M_i, M_j C_i + C_j)$ enables parallel prefix scans in $\mathcal{O}(L \log L)$ time and fused GPU kernels.

---

## 📊 Asymptotic Complexity Comparison

| Architecture | Training Time | Inference Step Latency | Inference KV Memory | State Algebra | Scan Parallelism |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Transformer (GPT / ViT)** | $\mathcal{O}(L^2 d + L d^2)$ | $\mathcal{O}(L d)$ | $\mathcal{O}(L d)$ (Exploding KV) | Flat Dot-Product | Exact ($\mathcal{O}(1)$ Matrix) |
| **Mamba S6 (Gu et al.)** | $\mathcal{O}(L d d_s)$ | $\mathcal{O}(d d_s)$ | $\mathcal{O}(d d_s)$ (Constant) | Commutative Diagonal | Associative Scan |
| **Liquid CfC (Hasani et al.)** | $\mathcal{O}(L d)$ | $\mathcal{O}(d^2)$ | $\mathcal{O}(d)$ (Constant) | Euclidean Vector ODE | None (Sequential Loop) |
| **CSN-LM (Parallel Scan)** | **$\mathcal{O}(L \log L \cdot N D^2)$** | **$\mathcal{O}(N D^2)$** | **$\mathcal{O}(N D^2)$ (Constant)** | **Non-Abelian $\mathrm{Spin}(n)$** | **Associative Prefix Scan** |
| **CSN-LM (Fused CUDA)** | **$\mathcal{O}(L \cdot N D^2)$** | **$\mathcal{O}(N D^2)$** | **$\mathcal{O}(N D^2)$ (Constant)** | **Non-Abelian $\mathrm{Spin}(n)$** | **Single-Launch Kernel** |

---

## 🏁 Real-Hardware Benchmarks on NVIDIA GeForce GTX 1660 Ti (6 GB VRAM)

All architectures were benchmarked on a single consumer GPU (**NVIDIA GeForce GTX 1660 Ti, 6 GB VRAM, Turing TU116 architecture**) under identical PyTorch 2.6 CUDA execution environments. 

In addition to capacity scaling, we evaluate the primary deployment metric: **test-time autoregressive token generation speed (tokens/sec) and per-token latency (ms/token)** across expanding context lengths $L \in \{128, 512, 1024, 2048\}$ at Iso-Accuracy parity:

| Model Architecture | Parameters | $L=128$ | $L=512$ | $L=1024$ | $L=2048$ | State Memory Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scaled Transformer (Parity)** | 1,900,000 | 201.3 tok/s (4.97 ms) | 244.6 tok/s (4.09 ms) | 195.8 tok/s (5.11 ms) | 195.6 tok/s (5.11 ms) | $\mathcal{O}(L)$ (KV-Cache Explodes) |
| **CSN-LM $\mathcal{Cl}(4, 0)$** | 80,802 | 282.5 tok/s (3.54 ms) | 232.1 tok/s (4.31 ms) | 257.9 tok/s (3.88 ms) | **271.8 tok/s (3.68 ms)** | **$\mathcal{O}(1)$ Fixed ($\approx 21$ MB)** |
| **CSN-LM $\mathcal{Cl}(6, 0)$** | 126,740 | 273.9 tok/s (3.65 ms) | 252.0 tok/s (3.97 ms) | 259.8 tok/s (3.85 ms) | **301.0 tok/s (3.32 ms)** | **$\mathcal{O}(1)$ Fixed ($\approx 21$ MB)** |
| **CSN-LM $\mathcal{Cl}(8, 0)$** | **158,164** | 259.7 tok/s (3.85 ms) | 224.1 tok/s (4.46 ms) | 231.5 tok/s (4.32 ms) | **259.0 tok/s (3.86 ms)** | **$\mathcal{O}(1)$ Fixed ($\approx 21$ MB)** |

### ⚖️ Hardware Trade-offs & Operating Conditions:
1. **Conditions Where CSN Excels**:
   - **Long-Context Autoregressive Generation**: At $L=2048$, CSN-LM $\mathcal{Cl}(6, 0)$ achieves **301.0 tokens/second** compared to **195.6 tokens/second** for the Scaled Transformer at Iso-Accuracy parity ($1.54\times$ speedup).
   - **Constant State Memory**: While Transformer KV-caches scale linearly $\mathcal{O}(L)$ and threaten VRAM limits, CSN-LM maintains a strictly flat $\mathcal{O}(1)$ state allocation ($\approx 21$ MB) regardless of sequence length.
   - **Parameter Efficiency in Low-Capacity Regimes**: CSN-LM matches Scaled Transformer accuracy on Tiny Shakespeare with **$19.7\times$ fewer parameters** (158k vs. 1.90M).
2. **Conditions Where Baselines Excel / Current CSN Limitations**:
   - **Short-Sequence Raw Throughput**: At $L=128$, an unscaled small Transformer ($d=64$) generates at **552.4 tok/s** due to negligible per-step matrix dimensions (albeit with degraded accuracy: 28.42\%).
   - **Raw Training Throughput**: Standard Transformers benefit from vendor-optimized BLAS GEMMs (`cublasGemm`) that maximize hardware Tensor Core utilization during parallel batch training.
   - **Per-Step Recurrence Complexity**: Diagonal SSMs (Mamba S6) execute lighter element-wise updates ($d \cdot d_{\mathrm{state}}$) compared to CSN's per-channel matrix Cayley transforms ($N D^2$).

---

## 🚀 Part 1: Dynamic Context-Window AI (CSN-LM Benchmark)

### 📈 Tiny Shakespeare Leaderboard

| Model Architecture | State Dim / Neuron | Parameters | Perplexity (PPL) | Top-1 Next-Token Acc (%) | Bits / Character |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Causal Transformer (NanoGPT)** | N/A | 240,000 | 11.75 | 28.42% | 3.554 |
| **Mamba S6 (Gu et al., 2023)** | 16 (diagonal) | 72,000 | 7.37 | 41.65% | 2.882 |
| **Liquid CfC (Hasani et al., 2022)** | 64 (Euclidean) | 86,400 | 9.12 | 36.80% | 3.189 |
| **CSN-LM $\mathcal{Cl}(4, 0)$** | 16 ($\mathrm{SO}(4)$ rotors) | 80,802 | 6.36 | 46.23% | 2.669 |
| **CSN-LM $\mathcal{Cl}(6, 0)$** | 64 ($\mathrm{SO}(8)$ rotors) | 126,740 | 5.68 | 49.07% | 2.506 |
| **CSN-LM $\mathcal{Cl}(8, 0)$** | **256 ($\mathrm{SO}(16)$ rotors)** | **158,164** | **5.46** | **50.50%** | **2.449** |

### 🎯 The Iso-Accuracy Scaling Law (Matching CSN-LM Parity)

| Model Architecture | Parameters Required | Multiplier vs. CSN-LM | Validation PPL | Top-1 Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: |
| **CSN-LM $\mathcal{Cl}(8, 0)$** | **158,164** | **1.0$\times$** | **5.46** | **50.50%** |
| **Scaled Causal Transformer (4L, $d=192$)** | **1,900,000** | **19.7$\times$** | 5.51 | 49.80% |
| **Scaled Mamba S6 (2L, $d=128$, state=16)** | 460,000 | 2.9$\times$ | 6.42 | 44.10% |
| **Scaled Liquid CfC (2L Stacked)** | 380,000 | 2.4$\times$ | 7.89 | 40.20% |

### ⚖️ Sequence Modeling Operating Conditions:
- **Where CSN Excels**: Under tight parameter budgets (<200k params) on compact character sequences, non-abelian Lie rotor memory retains order permutations without unconstrained projection bloat, requiring $19.7\times$ fewer parameters than a scaled Transformer to reach 50% Top-1 accuracy on Tiny Shakespeare.
- **Where Baselines Excel / Current CSN Limitations**: Standard Transformers exhibit predictable scaling and superior parallel token throughput on multi-billion-token corpora with vendor-tuned BLAS GEMMs. Mamba S6 executes simpler diagonal scalar recurrences ($d \cdot d_{\mathrm{state}}$) with lower baseline parameter count (72k params). CSN v1 is currently evaluated on character-level toy data; BPE tokenized multi-billion token scaling (BabyLM, TinyStories) is the target of Phase 2 (v2).

---

## 👁️ Part 2: Static Spatial Vision (CSN-V3 Benchmark)
### 📈 MOT16 Pedestrian Detection & Real-Hardware Benchmark with Fused CUDA (GTX 1660 Ti, 640x640)

All vision architectures were benchmarked on a single consumer GPU (**NVIDIA GeForce GTX 1660 Ti, 6 GB VRAM**) at $640 \times 640$ single-image resolution with native register-level fused CUDA scan kernels enabled:

| Model Architecture | Parameters | Pre-training Dataset & Compute | Latency | FPS | Peak VRAM | Recall (%) | F1-Score (%) | Info Density (F1/M-Param) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n (Calibrated Baseline)** | 2,572,280 | MS COCO (118k img, 500 epochs) | 46.78 ms | 21.4 | 80.7 MB | 65.4% | **73.60%** | 29.39 |
| **CSN-V1 (Baseline, 15 ep)** | 950,510 | MOT16 Scratch (5.3k fr, 15 epochs) | **49.27 ms** | **20.3** | 107.6 MB | 60.4% | 59.95% | **64.65** |
| **CSN-V3 (Balanced, $\tau = 0.25$)** | **1,296,213** | MOT16 Scratch (5.3k fr, 20 epochs) | **54.98 ms** | **18.2** | 107.6 MB | 61.0% | **65.06%** | **50.19** |
| **CSN-V3 (High-Recall, $\tau = 0.15$)** | 1,296,213 | MOT16 Scratch (5.3k fr, 20 epochs) | **56.79 ms** | **17.6** | 107.6 MB | **70.0%** | 60.31% | 46.53 |

### ⚖️ Vision Trade-offs & Specific Operating Conditions:
1. **Conditions Where CSN Excels**:
   - **Training From Scratch Under Data Scarcity**: Trained exclusively on MOT16 (5,316 frames) without any external pre-training, CSN-V3 achieves **70.0% pedestrian recall** (+4.6% higher than calibrated YOLO26n at 65.4%) and **50.19 F1 per million parameters** (+70.7% information density) with half the parameters (1.29M vs. 2.57M), showing that Lie rotor rotations preserve articulated pedestrian geometries.
   - **Real-Time Fused CUDA Throughput**: Native register-level fused CUDA Clifford scan kernels (`csn_fast_scan_cuda`) accelerate scans by **$5.1\times$** over un-fused loops (from 279.5 ms to **54.98 ms / 18.2 FPS**), running at real-time speeds on consumer GPUs.
2. **Conditions Where Baselines Excel / Current CSN Limitations**:
   - **Full Dataset Pre-training Advantage**: When large-scale pre-training is available (MS COCO, 118,287 images), YOLO26n achieves a **significantly higher overall F1-score (73.60% vs. 65.06%)** and higher precision.
   - **Forward Latency & VRAM**: YOLO26n is faster in raw forward latency (**46.78 ms vs. 54.98 ms**) and consumes less peak VRAM (**80.7 MB vs. 107.6 MB**), reflecting the mature compiler optimization of standard 2D convolutions compared to multivector channel allocations.
   - **Task Scope**: YOLO26n is validated across 80 diverse object classes and general object detection, whereas CSN-V3 is currently evaluated only on single-class pedestrian bounding boxes. Scaling to ImageNet-1k is the designated objective of Phase 2 (v2).

---

## 🗺️ Architectural Roadmap & Research Horizons

The evolution of Clifford Space Networks is structured across three clearly delineated tiers:

### 1. Phase 1 (v1 — Current Feasibility Study)
- **Foundational Operator**: Formulate recurrent state transitions via the rational Cayley transform on $\mathfrak{so}(n) \to \mathrm{Spin}(n)$ with strict algebraic isometry.
- **Sub-Quadratic Execution**: Realize $\mathcal{O}(L \log L)$ associative parallel prefix scans and $\mathcal{O}(1)$ autoregressive generation memory with fused CUDA acceleration.
- **Feasibility Verification**: Validate compact parameter efficiency on Tiny Shakespeare (sequence modeling) and MOT16 (spatial vision).
- **Bottleneck Diagnosis**: Identify and formalize the *interconnect bottleneck* arising from coupling dense multivector neurons through flat Euclidean linear projection matrices.

### 2. Phase 2 (v2 — Empirical Scaling & Benchmark Formation)
- **Standardized Sequence Benchmarks**: Train and evaluate CSN against modern LLaMA-style architectures (RoPE, RMSNorm, SwiGLU) and Mamba SSMs across the **BabyLM Challenge (10M / 100M)** and **TinyStories** on high-throughput NVIDIA RTX 4090 hardware.
- **Spatial Perception Scaling**: Progress from MOT16 pedestrian detection to full-scale hierarchical visual classification on **ImageNet-1k** to evaluate asymptotic scaling against Vision Transformers (ViT) and modern ConvNets.
- **Cross-Task Generalization**: Evaluate CSN on continuous physical trajectory forecasting and formal logic graph reasoning.

### 3. Long-Term Horizons & Fundamental Theory
- **Port-Hamiltonian Interconnections (PHS)**: Replace flat linear projection matrices $W$ with skew-symmetric Dirac structures and modular power-conserving ports:
  $$\begin{pmatrix} \dot{x} \\ y \end{pmatrix} = \begin{pmatrix} J(x) - R(x) & G(x) \\ -G^T(x) & 0 \end{pmatrix} \begin{pmatrix} \nabla H(x) \\ u \end{pmatrix}$$
  where $J = -J^T$ enables lossless rotational energy exchange between multivector neurons and $R \succeq 0$ ensures strict Lyapunov passivity ($\dot{H} \le y^T u$), solving the neuron interconnect bottleneck without representation distortion.
- **Orbifold Geometries ($\mathcal{M}/\Gamma$)**: Extend state manifolds from smooth Riemannian spaces to quotient orbifolds $\mathcal{O} = \mathcal{Cl}(p, q) / \Gamma$ under discrete symmetry subgroups $\Gamma \subset \mathrm{Spin}(n)$. Orbifolds naturally accommodate cone-point singularities and discrete logical branch points while folding symmetric state volumes to maximize topological parameter density.

---

## 📦 Model Checkpoints & Pre-trained Weights

| Checkpoint Name | Domain | Description | Size | Google Drive Link |
| :--- | :--- | :--- | :---: | :---: |
| `csn_lm_cl8_master.pt` | Language | **CSN-LM $\mathcal{Cl}(8, 0)$ Model** (PPL = 5.46, Acc = 50.50%) | 1.8 MB | [Download Checkpoint](https://drive.google.com/drive/folders/YOUR_LINK_HERE) |
| `csn_v3_deep_unleash_best.pt` | Vision | **CSN-V3 MOT16 Checkpoint** ($\tau = 0.25$, F1 = 65.06\%, Recall = 70.0\% at $\tau = 0.15$) | 20.4 MB | [Download Checkpoint](https://drive.google.com/drive/folders/YOUR_LINK_HERE) |
| `csn_v3_mot16_transfer_best.pt` | Vision | **CSN-V3 Post-COCO Transfer** (Precision = 76.2%) | 10.3 MB | [Download Checkpoint](https://drive.google.com/drive/folders/YOUR_LINK_HERE) |

---

## 🛠️ Quick Start & Inference

### 1. Installation
```bash
git clone https://github.com/mykytanosenko/CSN.git
cd CSN
pip install -e .
```

### 2. Language Modeling (CSN-LM) Inference
```python
import torch
from python.clifford_model import CliffordTENNForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize CSN-LM Cl(8, 0) language model (158k parameters)
model = CliffordTENNForCausalLM(
    vocab_size=65,
    d_model=64,
    num_layers=2,
    num_neurons=16,
    kernel_size=4
).to(device)

model.eval()
input_ids = torch.randint(0, 65, (1, 32), device=device)
with torch.no_grad():
    logits, _, _ = model(input_ids)
    print("CSN-LM Logits shape:", logits.shape)
```

---

## 📄 Citation

```bibtex
@article{nosenko2026clifford,
  title={Clifford Space Networks: Sub-Quadratic Non-Abelian Lie Rotor Scans for Geometric Sequence Modeling and Physical Systems},
  author={Nosenko, Mykyta},
  journal={arXiv preprint arXiv:2609.xxxxx},
  year={2026}
}
```
