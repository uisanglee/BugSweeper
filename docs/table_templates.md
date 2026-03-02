# Paper-ready Table Templates

## Table 1. Benchmark comparison (macro-level)

| Dataset | Model | Input Graph | Pooling | Macro-F1 (%) | Precision (%) | Recall (%) | Accuracy (%) | Params (M) |
|---|---|---|---|---:|---:|---:|---:|---:|
| Devign | GAT + global mean | AST/CFG | global mean |  |  |  |  |  |
| Devign | DiffPool-GNN | AST/CFG | learned hierarchical |  |  |  |  |  |
| Devign | **Proposed** | AST | deterministic hierarchical |  |  |  |  |  |
| Big-Vul | GAT + global mean | AST/CFG | global mean |  |  |  |  |  |
| Big-Vul | SAGPool-GNN | AST/CFG | learned top-k |  |  |  |  |  |
| Big-Vul | **Proposed** | AST | deterministic hierarchical |  |  |  |  |  |
| Smart-contract | GAT + global mean | Solidity AST | global mean |  |  |  |  |  |
| Smart-contract | TopKPool-GNN | Solidity AST | learned top-k |  |  |  |  |  |
| Smart-contract | **Proposed** | Solidity AST | deterministic hierarchical |  |  |  |  |  |

## Table 2. Ablation study on proposed model

| Setting ID | Pool Mode | Max Pool Levels | Macro-F1 (%) | Precision (%) | Recall (%) | Accuracy (%) | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| A1 | single | 0 |  |  |  |  | Function-level only |
| A2 | hier | 2 |  |  |  |  | + shallow hierarchy |
| A3 | hier | 4 |  |  |  |  | + deeper hierarchy |
| A4 | hier | 0 |  |  |  |  | all levels |

## Table 3. Per-class breakdown (smart-contract)

| Model | F1-safe | F1-reentrancy | F1-unchecked_low_calls | F1-time_manipulation | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Baseline-global |  |  |  |  |  |
| Proposed-single |  |  |  |  |  |
| Proposed-hier |  |  |  |  |  |
