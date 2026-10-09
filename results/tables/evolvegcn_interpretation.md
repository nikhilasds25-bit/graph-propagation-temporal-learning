## Interpretation

This notebook predicts the next-month dominant sentiment class for each node using the static directed Source -> Target graph and observed monthly node features. The task is intentionally separate from the Stage 7 diffusion simulation: Stage 7 models mechanistic contagion, while Stage 8 predicts observed monthly activity patterns.

- March 2023 contains 2,871 negative samples (16.31%) and 14,727 positive samples (83.69%).
- Both models predicted all 17,598 test nodes as positive. Negative-class precision, recall, and F1 were all 0.0.
- The static GCN baseline uses the directed graph and month-level node features. It does not invent timestamps or dynamic edges.
- The EvolveGCN-style model adds a lightweight temporal GRU over the monthly history so node states can evolve across adjacent months while still using the same static graph.
- For the March 2023 test month, GCN and EvolveGCN had identical predictions and identical metrics: accuracy=0.8368564609614729, positive_class_f1=0.9111832946635731, negative_class_f1=0.0, macro_f1=0.45559164733178653, weighted_f1=0.7625296272593727.
- There was no direct March test leakage. The model was trained and selected using data up to January 2023 and then evaluated once on the held-out March target.
- The result should be interpreted as failure to detect the minority negative class under the current task/data setup, not as evidence of strong balanced classification performance.
- EvolveGCN did not demonstrate an improvement over the static GCN in this experiment.
- The graph has only six monthly snapshots, so the temporal model is limited by the short observation window. There are no edge timestamps, so the task remains static-graph prediction rather than true dynamic temporal graph learning.
- The output should be interpreted as predictive modeling of observed activity, not causal evidence that negative information spreads more quickly.
