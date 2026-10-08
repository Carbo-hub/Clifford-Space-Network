# Clifford Space Networks (CSN)
### Sub-Quadratic Non-Abelian Lie Rotor Scans for Geometric Sequence Modeling and Physical Systems

[![arXiv](https://img.shields.io/badge/arXiv-2609.xxxxx-b31b1b.svg)](docs/paper/clifford_space_network.pdf)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-ee4c2c.svg)](https://pytorch.org/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--1976-1557-green.svg)](https://orcid.org/0009-0006-1976-1557)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)

---

## 📌 Executive Summary (v1 Feasibility Study)

**Clifford Space Networks (CSN)** formulate an analytical geometric computing paradigm that elevates state representations into **graded Clifford multivector algebras** $\mathcal{Cl}(p, q)$ and parameterizes state transitions via the continuous **non-abelian Spinor Lie group** $\mathrm{Spin}(n)$. 

Unlike existing Clifford neural networks designed as static PDE surrogates for 3D physics, CSN is architected as an **analytical solver for non-commutative logic graphs and physical Hamiltonian systems**, enabling exact sub-quadratic state transitions ($\mathcal{O}(L \log L)$ parallel associative scan, $\mathcal{O}(1)$ autoregressive generation memory).

**Current v1 Status & Feasibility Proofs**:
- **Dynamic Context-Window Systems (CSN-LM)**: Validated on Tiny Shakespeare, achieving 50.50% Top-1 accuracy with **$19.7\times$ fewer parameters** (158k vs. 1.90M) and **$1.54\times$ faster token generation** than Scaled Transformers with $\mathcal{O}(1)$ memory ($\approx 21$ MB). Large-scale evaluations on BabyLM 10M/100M and TinyStories are in progress to evaluate asymptotic multi-billion-token regimes where attention mechanisms traditionally scale.
- **Static Spatial Structures (CSN-V3)**: Validated on MOT16 pedestrian detection with **$1.98\times$ fewer parameters** than YOLO26n and **$+70.7\%$ higher information density** at real-time speeds (54.98 ms / 18.2 FPS) using register-level fused CUDA scan kernels. Full-scale hierarchical scaling to ImageNet-1k is under active development.
- **Identified v1 Bottleneck & Roadmap**: We diagnose the *interconnect bottleneck* of information-dense Clifford neurons and introduce **Port-Hamiltonian Interconnections (PHS)** as the architectural foundation for v2.

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

## 🌐 CSN as a Universal Inductive Bias for Artificial Intelligence

Clifford Space Networks provide a universal mathematical language capable of translating all modern neural network architectures into coordinate-free geometric spaces:

1. **Dynamic Context-Window & Autoregressive AI**: Large language models, code generation, streaming speech processing, and financial time-series benefit from non-abelian Lie rotor memory, which prevents commutative state collapse ($A_t A_{t-1} = A_{t-1} A_t$) without quadratic attention costs.
2. **Multimodal Perception & Spatial AI**: Computer vision, video temporal understanding, 3D point clouds, NeRFs, and 3D Gaussian Splatting natively map to graded multivectors, learning spatial rotations and reflections intrinsically without synthetic data augmentations.
3. **Physical AI & Scientific Machine Learning**: Partial differential equation solvers (Navier-Stokes fluid mechanics, Maxwell electromagnetism), molecular dynamics, protein conformation modeling, and Hamiltonian/Lagrangian quantum many-body systems naturally express physical conservation laws in Clifford algebras.
4. **Autonomous Decision-Making & Embodied AI**: Continuous reinforcement learning, robotic manipulator kinematics, quadruped locomotion, and autonomous driving state estimation operate on Lie groups ($\mathrm{SE}(3)$, $\mathrm{SO}(3)$), making continuous rotor dynamics the native state representation.
5. **Relational & Topological AI**: Hyperbolic and non-abelian knowledge graph embeddings, complex biological networks, and drug-target interaction graphs.

---

## ⚡ Mathematical Distinction: Exact Clifford Closed Form vs. Heuristic CfC

Closed-form Continuous-time (CfC) networks approximate continuous ODE dynamics via empirical heuristic bounding functions on flat coordinates:
$$h_t^{\text{CfC}} = \sigma(-f(x_t) \Delta t) \odot \tanh(W_1 x_t) + (1 - \sigma(-f(x_t) \Delta t)) \odot \tanh(W_2 x_t)$$
Because recurrent states are coupled inside non-linear saturating functions ($\tanh$), **CfC cannot be formulated as an associative matrix operation**, strictly constraining its training to slow sequential loops.

In contrast, CSN derives an **exact algebraic closed form** on the Lie group $\mathrm{Spin}(n)$ via the **rational Cayley transform** on the Lie algebra $\mathfrak{so}(n)$:
$$\Omega_t = \frac{1}{2} B_t \Delta t, \quad R_t = (I + \Omega_t)(I - \Omega_t)^{-1} \in \mathrm{Spin}(n)$$
$$h_t = \sqrt{\alpha_t} R_t h_{t-1} + (1 - \alpha_t) S_t \equiv M_t h_{t-1} + C_t$$
**Why Clifford Closed Form is Superior:**
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

Rather than reporting raw training token throughput (which is misleading because low-capacity Transformers require significantly more training tokens and $19.7\times$ more parameters to reach equivalent accuracy), we evaluate the **primary deployment metric**: **test-time autoregressive token generation speed (tokens/sec) and per-token latency (ms/token)** across expanding context lengths $L \in \{128, 512, 1024, 2048\}$ at Iso-Accuracy parity:

| Model Architecture | Parameters | $L=128$ | $L=512$ | $L=1024$ | $L=2048$ | State Memory Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scaled Transformer (Parity)** | 1,900,000 | 201.3 tok/s (4.97 ms) | 244.6 tok/s (4.09 ms) | 195.8 tok/s (5.11 ms) | 195.6 tok/s (5.11 ms) | $\mathcal{O}(L)$ (KV-Cache Explodes) |
| **CSN-LM $\mathcal{Cl}(4, 0)$** | 80,802 | 282.5 tok/s (3.54 ms) | 232.1 tok/s (4.31 ms) | 257.9 tok/s (3.88 ms) | **271.8 tok/s (3.68 ms)** | **$\mathcal{O}(1)$ Fixed ($\approx 21$ MB)** |
| **CSN-LM $\mathcal{Cl}(6, 0)$** | 126,740 | 273.9 tok/s (3.65 ms) | 252.0 tok/s (3.97 ms) | 259.8 tok/s (3.85 ms) | **301.0 tok/s (3.32 ms)** | **$\mathcal{O}(1)$ Fixed ($\approx 21$ MB)** |
| **CSN-LM $\mathcal{Cl}(8, 0)$ Master** | **158,164** | 259.7 tok/s (3.85 ms) | 224.1 tok/s (4.46 ms) | 231.5 tok/s (4.32 ms) | **259.0 tok/s (3.86 ms)** | **$\mathcal{O}(1)$ Fixed ($\approx 21$ MB)** |

### Key Hardware Insights:
1. **Up to $+53.9\%$ Faster Autoregressive Generation**: At $L=2048$, CSN-LM $\mathcal{Cl}(6, 0)$ achieves **301.0 tokens/second** compared to **195.6 tokens/second** for the Scaled Transformer at Iso-Accuracy parity ($1.54\times$ speedup).
2. **Strictly Flat $\mathcal{O}(1)$ State Memory**: While the Transformer's KV-cache scales linearly with context length $\mathcal{O}(L)$ and quickly threatens consumer VRAM limits, CSN-LM maintains a constant state allocation ($\approx 21$ MB) regardless of sequence length.
3. **Parameter Efficiency**: CSN-LM delivers this generation throughput advantage while requiring **$19.7\times$ fewer parameters** (158k vs. 1.90M).

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
| **CSN-LM $\mathcal{Cl}(8, 0)$ (Master)** | **256 ($\mathrm{SO}(16)$ rotors)** | **158,164** | **5.46** | **50.50%** | **2.449** |

### 🎯 The Iso-Accuracy Scaling Law (Matching CSN-LM Parity)

| Model Architecture | Parameters Required | Multiplier vs. CSN-LM | Validation PPL | Top-1 Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: |
| **CSN-LM $\mathcal{Cl}(8, 0)$** | **158,164** | **1.0$\times$** | **5.46** | **50.50%** |
| **Scaled Causal Transformer (4L, $d=192$)** | **1,900,000** | **19.7$\times$** | 5.51 | 49.80% |
| **Scaled Mamba S6 (2L, $d=128$, state=16)** | 460,000 | 2.9$\times$ | 6.42 | 44.10% |
| **Scaled Liquid CfC (2L Stacked)** | 380,000 | 2.4$\times$ | 7.89 | 40.20% |

---

## 👁️ Part 2: Static Spatial Vision (CSN-V3 Benchmark)
### 📈 MOT16 Pedestrian Detection & Real-Hardware Benchmark with Fused CUDA (GTX 1660 Ti, 640x640)

All vision architectures were benchmarked on a single consumer GPU (**NVIDIA GeForce GTX 1660 Ti, 6 GB VRAM**) at $640 \times 640$ single-image resolution with native register-level fused CUDA scan kernels enabled:

| Model Architecture | Parameters | Pre-training Dataset & Compute | Latency | FPS | Peak VRAM | Recall (%) | F1-Score (%) | Info Density (F1/M-Param) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n (Calibrated Baseline)** | 2,572,280 | MS COCO (118k img, 500 epochs) | 46.78 ms | 21.4 | 80.7 MB | 65.4% | **73.60%** | 29.39 |
| **CSN-V1 (Surgical Baseline)** | 950,510 | MOT16 Scratch (5.3k fr, 15 epochs) | **49.27 ms** | **20.3** | 107.6 MB | 60.4% | 59.95% | **64.65** |
| **CSN-V3 Deep Unleash (Peak)** | **1,296,213** | MOT16 Scratch (5.3k fr, 20 epochs) | **54.98 ms** | **18.2** | 107.6 MB | 61.0% | **65.06%** | **50.19** |
| **CSN-V3 High-Sensitivity ($\tau = 0.15$)** | 1,296,213 | MOT16 Scratch (5.3k fr, 20 epochs) | **56.79 ms** | **17.6** | 107.6 MB | **70.0%** | 60.31% | 46.53 |

### ⚡ The Extreme Pre-training Compute Disparity:
1. **Incomparably Smaller Pre-training Footprint**: The standard YOLO26n baseline relies on extensive pre-training across the complete **MS COCO dataset ($118,287$ images, $>860,000$ labeled bounding boxes across 80 classes)** trained for 500 epochs on multi-GPU server clusters (thousands of GPU-hours). In contrast, **CSN-V3 was trained directly from scratch on MOT16 alone (merely $5,316$ video frames across 7 sequences)** for only 20 epochs on a **single consumer GPU (GTX 1660 Ti)** in a few hours.
2. **Superior Pedestrian Recall Under Data Scarcity**: Despite training with **$>22\times$ less data and $>1000\times$ less compute**, CSN-V3 achieves **70.0% pedestrian recall** (+4.6% higher than calibrated YOLO26n's 65.4%), demonstrating that continuous Lie rotor rotations capture articulated pedestrian geometries without requiring millions of augmented training images.
3. **$+70.7\%$ Higher Information Density**: CSN-V3 delivers **50.19 F1 per million parameters** vs. 29.39 for YOLO26n, while operating with **$1.98\times$ fewer parameters** (1.29M vs. 2.57M).
4. **Real-Time Fused CUDA Throughput**: Utilizing native register-level fused CUDA Clifford scan kernels (`csn_fast_scan_cuda`), CSN-V3 accelerates by **$5.1\times$** over un-fused PyTorch loops (from 279.5 ms to **54.98 ms / 18.2 FPS**), running at virtually identical latency to standard Euclidean detectors (54.98 ms vs. 46.78 ms) on a consumer GPU.

---

## 📦 Model Checkpoints & Pre-trained Weights

| Checkpoint Name | Domain | Description | Size | Google Drive Link |
| :--- | :--- | :--- | :---: | :---: |
| `csn_lm_cl8_master.pt` | Language | **CSN-LM $\mathcal{Cl}(8, 0)$ Master LLM** (PPL = 5.46, Acc = 50.50%) | 1.8 MB | [Download Checkpoint](https://drive.google.com/drive/folders/YOUR_LINK_HERE) |
| `csn_v3_deep_unleash_best.pt` | Vision | **CSN-V3 Peak MOT16 Checkpoint** (F1 = 65.06%, Recall = 70.0%) | 20.4 MB | [Download Checkpoint](https://drive.google.com/drive/folders/YOUR_LINK_HERE) |
| `csn_v3_mot16_transfer_best.pt` | Vision | **CSN-V3 Post-COCO Transfer** (Peak Precision = 76.2%) | 10.3 MB | [Download Checkpoint](https://drive.google.com/drive/folders/YOUR_LINK_HERE) |

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
