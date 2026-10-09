# Analysis notes

## Temporal reconstruction

- Timestamped interaction events: 129,604
- Unique reconstructed directed pairs: 109,219
- Pair overlap with supplied network: 109,217
- Reconstructed-pair precision: 99.9982%
- Supplied-pair coverage: 99.8638%
- Pearson correlation between reconstructed pair counts and supplied Weight: 0.9822
- Spearman correlation: 0.9155
- Weekly snapshots: 25
- Mean nodes/week: 4,083.76
- Mean directed edges/week: 4,994.40

## Weekly prediction cohort

- User-week label records: 77,247
- Ties excluded: 1,500
- User-weeks missing historical graph features: 59,455
- Prediction-ready records: 16,292
- Rolling test weeks: 15 (2022-12-19 to 2023-03-27)

## Main baseline results

| Model | Macro-F1 | Class-0 AP | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Majority | 0.3460 | 0.5292 | 0.5000 |
| Persistence | 0.5396 | 0.5575 | 0.5403 |
| Logistic regression | 0.4742 | 0.5529 | 0.5121 |
| MLP | 0.4522 | 0.5494 | 0.5066 |
| GRU | 0.4487 | 0.5464 | 0.5055 |

## Graph-model comparisons

- Best graph model by macro-F1: reversed GAT = 0.4943
- Reversed GAT - Persistence: -0.0453, 95% CI [-0.0551, -0.0349]
- Reversed GAT - MLP: +0.0421, 95% CI [0.0211, 0.0640]
- Reversed GAT - matched feature-only control: +0.0231, 95% CI [0.0135, 0.0342]
- Temporal GCN - aligned GRU: +0.0020, 95% CI [-0.0033, 0.0068]
- Temporal GCN - best static graph: -0.0047, 95% CI [-0.0159, 0.0069]

## History-aware matched experiment

- History-only logistic macro-F1: 0.5529
- History + structural MLP macro-F1: 0.5496
- History + reversed GAT macro-F1: 0.5184
- GAT - MLP: -0.0312
- Ordinary bootstrap 95% CI: [-0.0444, -0.0208]
- 3-week moving-block 95% CI: [-0.0375, -0.0221]
- 4-week moving-block 95% CI: [-0.0372, -0.0232]
- Holm-adjusted p-value: 0.00122

These are aggregate findings from completed local runs. Seed-level outputs remain in the local analysis project unless explicitly committed later.
