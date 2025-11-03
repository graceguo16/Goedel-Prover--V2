
# Goedel-Prover--V2
# 🧠 Thesis Project: Lean-Verified Statistical Simulation and LLM-Guided Theorem Proving

This repository is a research-oriented extension of **Gödel-Prover-V2**, designed for my undergraduate thesis on **combining formal verification in Lean with LLM reasoning**.  
The project aims to replicate Gödel-Prover’s stochastic experiments and extend it toward **statistical proof** verified in **Lean4** with LLM-assisted formalization.

---

## 🧩 Project Overview

### 🎯 Objectives
- Replicate and extend the **Gödel-Prover-V2** theorem-proving framework.  
- Implement **Statisitcal simulations** in Python.  
- Translate empirical results into **formal proofs in Lean4**.  
- Use **LLMs (e.g.,GPT-5)** to assist in:
  - Proof suggestion and theorem structuring.
  - Automated lemma generation and error correction.
  - Comparing machine-generated vs human-verified proofs.

---

## 🧱 Repository Structure

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




