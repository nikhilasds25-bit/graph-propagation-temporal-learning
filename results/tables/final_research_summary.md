# Final research summary

## 1. Dataset
- The project retained the processed datastreams used for the diffusion and temporal tasks, with the Stage 7 diffusion population fixed at 17,598 nodes, 29,715 directed edges, and 37,115 processed tweets.
- The Stage 7 diffusion start month was 2022-11.
- For Stage 8, the March 2023 held-out target was strongly imbalanced: 2,871 negative samples (16.31%) and 14,727 positive samples (83.69%).

## 2. Network construction
- The directed network was built from the processed Stage 6 edge list without inventing timestamps.
- The final Stage 7 simulator kept the Source -> Target direction and did not convert the graph to undirected form.
- Edge weights were normalized using the primary max-normalization rule, with log normalization retained only as a sensitivity comparison.
- Self-loops were retained from the Stage 6 data but were handled safely by preventing infection from re-assigning state on already infected or recovered nodes.

## 3. Diffusion population
- The full diffusion population remained 17,598 users; no 10K subset was introduced.
- The Stage 7 model used a fixed population and directed edge set throughout the primary and corrected runs.
- The simulation was built around a static topology because no edge timestamps were available.

## 4. Trust and PageRank features
- Trust and PageRank were loaded from the Stage 6 node-level outputs rather than estimated during the diffusion run.
- Trust was clipped into the [0, 1] range and used as a transmission multiplier.
- PageRank was min-max normalized and used as a source influence multiplier.
- These values affected transmission probability but were not themselves learned within the diffusion process.

## 5. Competitive diffusion model
- The Stage 7 simulator uses a static S/I1/I2/R compartment model with one state per node.
- The primary transmission probabilities were fixed at 0.14 for both positive and negative information.
- Recovery probability was fixed at 0.25.
- Competition when both positive and negative exposure arrived in the same step used the proportional exposure rule by default.
- The primary experiment used 25 seeds and 30 stochastic runs per configuration.

## 6. Primary Stage 7 findings
- Negative-only baseline: mean negative cascade size was 26.6; positive cascade size was 0.0.
- Positive-only baseline: mean positive cascade size was 26.03; negative cascade size was 0.0.
- Full competitive random model: mean negative cascade size was 25.07 and mean positive cascade size was 25.13; all 30 runs were mixed_close.
- The high_trust, high_pagerank, and high_negative_ratio seed strategies all produced near-balanced outcomes with mean negative/positive cascade sizes around 25.1-25.2.
- The main conclusion from the primary results is that, under the existing model and configuration, competitive propagation remained close to balanced rather than showing a strong negative-dominance effect.

## 7. Ablation findings
- The ablation suite showed that removing edge weights produced the largest tested change: mean negative cascade size 27.7 and mean positive cascade size 28.1.
- The full model remained near-balanced at approximately 25.0 negative and 25.2 positive cascade size.
- Trust, PageRank, and sentiment ablations did not produce dramatic deviations from the full model under the tested setup.

## 8. Sensitivity findings
- The parameter sensitivity grid showed that transmission probability, recovery probability, and seed count changed cascade magnitude in the expected direction.
- Max and log edge normalization produced broadly similar patterns in the edge-normalization comparison, indicating that the qualitative interpretation was not highly sensitive to the normalizer used.
- Competition-rule comparison did not materially change the qualitative conclusion under the tested settings.

## 9. Stage 8 GCN vs EvolveGCN findings
- Stage 8 retained the chronological protocol: training through the January 2023 target, February validation, and March 2023 held-out testing.
- The original result was a majority-class collapse: both models predicted all March nodes as positive, with negative F1 = 0, macro F1 = 0.4556, and weighted F1 = 0.7625.
- The corrected protocol used training-only class weighting and selected each model's threshold on February validation. The selected thresholds were frozen for March.
- On March, the corrected GCN achieved balanced accuracy 0.4978, negative F1 0.0968, macro F1 0.4877, weighted F1 0.7510, and PR-AUC 0.8371.
- On March, the corrected EvolveGCN-style model achieved balanced accuracy 0.5008, negative F1 0.1489, macro F1 0.5000, weighted F1 0.7366, and PR-AUC 0.8364.
- EvolveGCN-style therefore has modestly higher negative-class F1 and macro F1, while GCN has slightly higher PR-AUC. Balanced accuracy remains close to 0.5 for both, so these results do not support strong temporal-model superiority.
- Stage 8 is future-label prediction and remains distinct from the Stage 7 stochastic competitive diffusion simulation.

## 10. Major limitations
- The Stage 7 diffusion network is static because edge timestamps are unavailable; it does not model dynamic edge formation or decay.
- The Stage 8 task is a strongly imbalanced classification problem, so majority-class prediction can produce high weighted F1 even when the minority class is entirely missed.
- The Stage 7 model is a stylized simulator rather than a validated reconstruction of real-world Twitter diffusion.
- The corrected Stage 8 models still have balanced accuracy near 0.5 on the held-out March period, and the six-month observation window provides limited evidence for temporal-model advantage.
- The single chronological test period does not justify statistical significance, confidence intervals, causal effects, or claims about observed diffusion.

## 11. Future work
- Add temporal edge information or dynamic graph structure if available.
- Evaluate additional chronological periods if future data become available, while retaining leakage-safe class weighting and validation threshold selection.
- Treat the Stage 7 diffusion model as a simulation framework rather than a causal reconstruction of observed social diffusion.
- Validate model assumptions against observed temporal data before making stronger causal or predictive claims.

## Final conclusion
The completed results support a conservative interpretation: user influence, trust, and information orientation matter within the simulation as transmission modifiers, but the primary competitive diffusion configuration remained near-balanced rather than negative-dominant. In Stage 8, imbalance correction recovered limited negative-class detection; EvolveGCN-style showed modestly higher negative F1 and macro F1, while GCN showed slightly higher PR-AUC. Balanced accuracy remained close to 0.5 for both, so the evidence provides only limited support for temporal processing and does not support a strong causal, observed-diffusion, or temporal-model-superiority claim.
