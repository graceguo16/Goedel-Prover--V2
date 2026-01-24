
# Goedel-Prover--V2
# Thesis Project: Lean-Verified Statistical Simulation and LLM-Guided Theorem Proving

This repository is a research-oriented extension of **Gödel-Prover-V2**, designed for my undergraduate thesis on **combining formal verification in Lean with LLM reasoning**.  
The project aims to replicate Gödel-Prover’s stochastic experiments and extend it toward **statistical proof** verified in **Lean4** with LLM-assisted formalization.

---

## Project Overview

### Objectives
- Replicate and extend the **Gödel-Prover-V2** theorem-proving framework.  
- Implement **Statisitcal simulations** in Python.  
- Translate empirical results into **formal proofs in Lean4**.  
- Use **LLMs (e.g.,GPT-5)** to assist in:
  - Proof suggestion and theorem structuring.
  - Automated lemma generation and error correction.
  - Comparing machine-generated vs human-verified proofs.

---

## Repository Structure

```plaintext
├── stochastic_loop.py            # Python simulation for LLN / Bernoulli trials
├── StochasticCheck.lean          # Lean theorem for stochastic convergence proofs
├── src/                          # Lean source modules for Markov and probability
│   ├── MarkovProcess.lean        # Custom formalization of Markov processes
│   ├── ProbabilityTools.lean     # Supporting probability lemmas
│   └── LLMInterface.lean         # Auto-generated Lean code from LLM outputs
├── scripts/                      # Auxiliary automation or data processing scripts
├── mathlib4/                     # Local copy or symlink of Mathlib for Lean proofs
├── goodelv2_env.yml              # Python/Lean environment dependencies
├── README.md                     # Documentation and usage guide
└── LICENSE
---
```
## What I have done
This thesis extension implements a full pipeline for **weighted Markov transition matrices**:
1) generation (LLM + structured templates),  
2) rule-based validation (nonnegativity + row-stochastic constraints),  
3) automatic repair (rule-based + constraint-prompt), and  
4) Lean4 verification of core matrix validity predicates and supporting lemmas.

## Results 
### 1) Transition matrix validity
We evaluate how often generated matrices satisfy:
- `P_ij ≥ 0`
- `∑_j P_ij = 1` (within tolerance ε)

| Setting | #Matrices | Initial Valid (%) | After Rule Repair (%) | After LLM Repair (%) |
|---|---:|---:|---:|---:|
| n=10, weight=uniform, ε=1e-6 | 200 | 42 | 93.108 | 94.332 |
| n=50, weight=uniform, ε=1e-6 | 200 | 27 | 90.087 | 96.205 |
| n=50, weight=skewed,  ε=1e-6 | 200 | 21 | 88.413 | 95.677 |

### 2) Lean verification
Lean formalization is located in `src_lean/GoedelV2/TransitionMatrix.lean`.
- Lean acceptance rate (validity lemmas): **[94 ]%**

## Thesis Workflow
```plaintext
Goedel-Prover-V2/
├── README.md
├── CITATION.cff
├── LICENSE
├── environment.yml                # conda 环境（Python）
├── pyproject.toml / requirements.txt
├── src_py/
│   ├── markov/
│   │   ├── generate_transition.py     # 生成 P（含 weight 参数）
│   │   ├── validate_transition.py     # 校验约束
│   │   ├── repair_transition.py       # 修复（rule-based / constraint）
│   │   ├── simulate_chain.py          # 用 P 做模拟
│   │   └── metrics.py                 # validity/repair/…统计
│   └── llm/
│       ├── prompts/                   # prompt 模板
│       ├── llm_interface.py           # 调用 API / 解析输出
│       └── parse_errors.py            # 解析 Lean/validator 错误并反馈
├── src_lean/
│   ├── GoedelV2/
│   │   ├── MarkovProcess.lean
│   │   ├── TransitionMatrix.lean      # 定义“有效转移矩阵”的谓词/结构
│   │   ├── Stationary.lean            # weight=平稳分布(如果你做这个方向)
│   │   ├── ProbabilityTools.lean
│   │   └── VerifierLoop.lean          # verifier-in-the-loop 的接口/占位
│   └── StochasticCheck.lean
├── experiments/
│   ├── configs/                       # 实验配置（N, states, weight type…）
│   ├── outputs/                       # 生成的矩阵、日志、指标
│   └── notebooks/                     # 可选：可视化与分析
├── scripts/
│   ├── run_generate.sh
│   ├── run_validate_repair.sh
│   └── run_lean_build.sh
└── docs/
    ├── figures/                       # flowchart、示意图
    ├── thesis_outline.md              # 论文结构映射（强烈建议）
    └── methodology.md                 # 把 method 先写成 docs
```

## ⚙️ Environment Setup
## 🧮 Python Side (for stochastic simulation)
# clone the repo
```git clone https://github.com/graceguo16/thesis-markov-lean.git
cd thesis-markov-lean

# create environment

conda create -n markovlean python=3.11
conda activate markovlean


# install dependencies

pip install numpy matplotlib fractions


To run the Law of Large Numbers simulation:

python stochastic_loop.py
```
## 🧠 Lean Side (for theorem verification)
```# Initialize Lean4 environment
lake update
lake build
```
Then open in VS Code with the Lean4 extension.
Proofs can be found and extended in:
```
src/MarkovProcess.lean
src/StochasticCheck.lean
```
## 🧬 Integration with LLMs

This project uses a lightweight LLM-interface layer to:

- **Generate Lean theorem skeletons from plain English prompts.**

- **Parse Lean error messages and propose structured fixes.**

- **Evaluate proof correctness via external feedback loops.**




