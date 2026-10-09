# Final interpretation

This project combines two distinct tasks: a static competitive diffusion simulation (Stage 7) and a temporal graph classification task (Stage 8). The results should be interpreted separately and conservatively.

## Stage 7 interpretation
- The Stage 7 diffusion model is built on a static directed graph because no edge timestamps were available. The topology is therefore fixed for the simulation, and the results should not be interpreted as evidence of temporal edge dynamics or causal social influence over time.
- The primary configuration used a transmission probability of 0.14 for both positive and negative information and a recovery probability of 0.25, with 25 seeds and 30 stochastic runs per configuration.
- Under the primary competitive setup, the negative and positive cascades stayed close to balanced. This does not support a claim that negative information dominates in the network under this model; it indicates that the modeled competitive propagation remained near-balanced in the completed runs.
- Removing edge weights produced the largest tested change among the ablations. This suggests that edge-weight structure materially affects cascade magnitude in the simulator, though the effect remained within the same overall near-balanced regime.
- Transmission probability, recovery probability, and seed count changed cascade magnitude across the tested sensitivity grid. The exact magnitudes varied, but the broad qualitative interpretation remained one of a stylized competitive diffusion process rather than a real-world reconstruction.
- Max and log edge normalization produced broadly similar patterns, suggesting that the qualitative result was not highly sensitive to the chosen normalization family in this setup.

## Stage 8 interpretation
- The March 2023 test target was strongly imbalanced: 2,871 negative samples (16.31%) and 14,727 positive samples (83.69%).
- The original result was a majority-class collapse: both models predicted every March test node as positive. Negative-class F1 was 0.0, macro F1 was 0.4556, and weighted F1 was 0.7625. The high weighted F1 reflected majority-class dominance rather than strong balanced performance.
- The corrected protocol used class weights calculated from training labels only and selected each threshold on February validation. The thresholds were frozen before the single March evaluation; no March labels were used for training or threshold selection.
- Corrected GCN March results were balanced accuracy 0.4978, negative F1 0.0968, macro F1 0.4877, weighted F1 0.7510, and PR-AUC 0.8371.
- Corrected EvolveGCN-style March results were balanced accuracy 0.5008, negative F1 0.1489, macro F1 0.5000, weighted F1 0.7366, and PR-AUC 0.8364.
- EvolveGCN-style has higher negative-class F1 and macro F1 than GCN, but the improvement is modest. GCN has slightly higher PR-AUC, and balanced accuracy remains close to 0.5 for both.
- These results do not support strong temporal-model superiority. Temporal processing provides only limited evidence of improvement under the current task and six-month observation window.
- Stage 7 is a stochastic competitive diffusion simulation, whereas Stage 8 is future-label prediction. The Stage 8 results are not evidence of observed diffusion, causal effects, or faster negative-information spread.

## Scientific conclusion
The completed results support a conservative interpretation of the research question: user influence, trust, and information orientation are represented in the Stage 7 diffusion model as transmission modifiers, and the ablation results suggest that the edge-weight structure is among the most influential tested components. The primary competitive propagation remained near-balanced rather than negative-dominant. In Stage 8, imbalance correction produced limited minority-class detection; EvolveGCN-style improved negative F1 and macro F1 modestly, while GCN retained a slightly higher PR-AUC. The near-0.5 balanced accuracy for both models means the evidence does not establish robust temporal-model superiority and remains a model-based result rather than a causal claim about actual Twitter diffusion.
