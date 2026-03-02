# Benchmark Matrix Template (Graph Classification for Code Tasks)

## 1) Datasets

| Domain | Dataset | Task | Label Space | Graph View | Split Protocol | Primary Metrics |
|---|---|---|---|---|---|---|
| C/C++ vulnerability | Devign | Function-level vulnerability classification | Binary | AST+CFG/Code graph | Official / Reproduced | F1, Acc, AUC |
| C/C++ vulnerability | Big-Vul (MSR) | Vulnerable vs clean function classification | Binary | AST/CFG/PDG variants | Time-based / random split | F1, Precision, Recall |
| Smart-contract vulnerability | Smart-contract corpus (reentrancy / unchecked calls / time manipulation) | Multi-class graph classification | 4-way | Solidity AST graph | train/valid/test | Macro-F1, per-class F1 |

## 2) Model Family Comparison Matrix

| Family | Representative Models | Input | Pooling Strategy | Pros | Cons |
|---|---|---|---|---|---|
| Pure GNN | GCN/GAT/GraphSAGE + readout | Graph only | global mean/max | Simple, fast | Weak long-range semantics |
| Hierarchical GNN | DiffPool/TopK/SAGPool, deterministic hierarchy | Graph only | learned or rule-based hierarchical pooling | Better structure abstraction | More complexity / hyperparams |
| Hybrid Code+Graph | GraphCodeBERT variants + GNN fusion | Tokens + graph | mixed | Strong accuracy | Heavy compute |
| Proposed (this repo) | AST deterministic hierarchical pooling + GNN | AST graph (+ tokenized node text) | deterministic multi-level pooling | Interpretable, stable, low variance | Needs good AST construction |

## 3) Experimental Grid

| Axis | Values |
|---|---|
| Seed | 42, 7, 13 |
| Pool mode | single, hier |
| Max pool levels | 0(all), 2, 4 |
| Hidden channels | 256, 512, 1024 |
| GNN depth | 2, 3, 4 |
| Batch size | 32, 64 |
| LR | 1e-4, 5e-5 |

## 4) Fair Comparison Checklist

- Same split, same seed set, same metric implementation
- Same tokenizer/vocab policy
- Same training budget (epochs, early stopping)
- Report mean ± std over seeds
- Include per-class F1 for imbalanced vulnerability labels
