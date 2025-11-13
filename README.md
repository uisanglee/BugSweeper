# BugSweeper: Smart Contract Vulnerability Detection Pipeline

**BugSweeper** provides an end-to-end workflow to convert Solidity contracts into function-level AST graphs (FLAGs) and train/evaluate a Graph Neural Network (GNN) using PyTorch Geometric aimed at vulnerability detection.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Components](#components)
* [Installation & Requirements](#installation--requirements)
* [Directory Structure](#directory-structure)
* [Usage](#usage)

  * [1) Data Preprocessing](#1-data-preprocessing)
  * [2) Model Training](#2-model-training)
  * [3) Model Evaluation](#3-model-evaluation)
* [Parameters & Configuration](#parameters--configuration)
* [References](#references)

---

## Project Overview

BugSweeper:

1. **AST Preprocessing**: Uses `solc` to generate AST JSON, parses and cleans with NetworkX.
2. **FLAG Generation & Merging**: Extracts function-level subgraphs and merges related code with recursive depth control (`coverage`).
3. **Dataset Construction**: Converts each FLAG into a `torch_geometric.data.Data` object and wraps in `MyDataset`.
4. **Training & Evaluation**: Trains a GNN with logit adjustment for class imbalance, validates via macro F1, and tests with detailed per-class metrics.

---

## Components

| File               | Description                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| `preprocess.py`    | AST JSON generation, Graph Constructor                                   |
| `train.py`         | Model definition, training loop, validation, checkpointing, and testing. |
| `utils.py`         | `MyDataset` wrapper and helper functions.                                |
| `config.py`        | Global settings: dataset paths, regex patterns, class mappings.          |
| `requirements.txt` | Python package dependencies.                                             |

---

## Installation & Requirements

```bash
git clone https://github.com/yourusername/BugSweeper.git
cd BugSweeper
pip install -r requirements.txt
```

* **Python**: 3.8+
* **PyTorch**: >=1.12
* **PyTorch Geometric**, **networkx**, **chardet**, **pandas**, **scikit-learn**, **solc-select**

---

## Directory Structure

```plaintext
BugSweeper/
├── config.py
├── utils.py
├── preprocess.py          # AST preprocessing pipeline
├── train.py               # Training & evaluation script
├── requirements.txt
├── models/                # Saved model checkpoints
├── datasets/              # Raw & processed datasets
│   ├── function/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── contract/
└── README.md              # This file
```

---

## Usage

### 1) Data Preprocessing

```bash
python preprocess.py \
  --coverage 4 \            # Recursive FLAG merging depth (default: 4)   \
  --mode train \            # Mode: train | valid | test | DApp    \
  --level function          # Level: function | contract(not available)
```

* Outputs: `datasets/<level>/<mode>/<coverage>/raw/data_list.pkl` and `processed/data.pt`

### 2) Model Training

```bash
python train.py \
  --coverage 4 \           # Must match preprocessing coverage  \
  --model POOL \           # Options: POOL  \
  --loss ce \              # Loss function: ce (CrossEntropy)   \
  --lr 0.001 \             # Learning rate   \
  --wd 5e-4 \              # Weight decay   \
  --epochs 50 \            # Total epochs   \
  --batch_size 64           # Batch size
```

* Best checkpoint saved as `models/<coverage>_<epochs>_<loss>.pth` based on validation macro-F1.

### 3) Model Evaluation

```bash
python train.py \
  --coverage 4 \           # Match coverage      \
  --load models/4_50_ce.pth \  # Pretrained checkpoint path    \
  --mode test              # Run in test mode
```

* Prints macro-averaged and per-class precision, recall, and F1 scores.

---

## Parameters & Configuration

* `coverage`: Integer depth for recursive FLAG merging.
* `mode`: `train` | `valid` | `test` | `DApp`.
* `level`: `function` | `contract`.
* `model`: GNN architecture choice.
* `loss`: Loss type (`ce`).
* `lr`, `wd`, `epochs`, `batch_size`: Training hyperparameters.

---

## References

1. Zhuang et al., TMP: Smart Contract Vulnerability Detection using GNN, ASIA CCS 2021.
2. Liu et al., AME: Attention-based Smart Contract Security, CCS 2021.
3. Wu et al., Peculiar: GraphCodeBERT-based Vulnerability Detection, EMNLP 2021.
4. Zhang et al., ReVulDL: Reentrancy Detection with Deep Learning, ICSE 2022.
5. Wang, Wenhan, et al., Detecting code clones with graph neural network and flow-augmented abstract syntax tree, SANER(IEEE), 2020.

---

