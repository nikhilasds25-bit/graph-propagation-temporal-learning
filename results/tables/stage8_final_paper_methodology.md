# Stage 8 Methodology, Results, and Interpretation

## Methodology

Stage 8 predicts the next-month dominant sentiment label for each node using the static directed Source -> Target graph and the existing monthly node features. The temporal protocol is chronological: training uses the existing periods through the January 2023 target, February 2023 is the validation period, and March 2023 is the held-out test period. No random temporal split was used.

Both the static GCN baseline and the EvolveGCN-style model use training-only class weighting. For training examples $i$ with observed class $y_i$, the weighted cross-entropy is

$$
L = -\sum_i w_{y_i} \log p_{i,y_i}
$$

where class weights are calculated from the training data only. February labels are used for validation and threshold selection, not for model training. For each model, the classification threshold is selected by

$$
τ^* = argmax_τ MacroF1(validation, τ)
$$

The selected $\tau^*$ remains fixed for the March 2023 test evaluation. March labels are not used for training, threshold selection, or model selection. The March test set contains 17,598 nodes: 2,871 negative and 14,727 positive labels.

## Why the correction was necessary

The original Stage 8 result was a majority-class collapse. Because positive labels comprised 83.69% of the March test set, both models predicted every test node as positive. This produced negative-class F1 = 0 and macro F1 = 0.4556, while weighted F1 remained 0.7625 because it was dominated by the majority class. The corrected protocol addresses the training imbalance with training-only class weighting and replaces the arbitrary fixed decision rule with a threshold selected on February validation data.

## Final interpretation

The corrected March evaluation shows nonzero negative-class detection for both models. EvolveGCN-style has higher negative-class F1 (0.1489 vs. 0.0968) and slightly higher macro F1 (0.5000 vs. 0.4877) than the static GCN. The improvement is modest. GCN has slightly higher PR-AUC (0.8371 vs. 0.8364), while balanced accuracy remains close to 0.5 for both models (0.4978 and 0.5008). These results do not support a claim of strong temporal-model superiority. Temporal processing provides only limited evidence of improvement under the current task.

The results are predictive rather than causal. Stage 7 is a stochastic competitive diffusion simulation; Stage 8 is future-label prediction. Neither stage should be interpreted as direct evidence that observed negative information diffuses faster or more widely.

## Limitations

The Stage 8 task has only six monthly snapshots, a strongly imbalanced target, and a static graph because edge timestamps are unavailable. Balanced accuracy near 0.5 indicates that the corrected models still provide limited balanced discrimination on the held-out March period. The EvolveGCN-style architecture does not establish temporal-model superiority from this single chronological test period. No statistical significance, confidence intervals, causal effects, or observed diffusion claims are inferred from these results.
