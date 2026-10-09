# Reviewer correction evidence report

Analysis only. Original files/model results preserved; no LaTeX edited. Classifications refer to claims described in the request because the current manuscript itself was not supplied in this project. Numerical statements that depend on matching rule or node universe must retain that qualifier. Canonical outputs below supersede generated interim validation drafts.

## 1. Exact vs normalized username matching

**Issue:** Exact vs normalized username matching
**Reviewer concern:** Inconsistent literal and case-insensitive populations
**What was checked:** Raw tweet/network endpoint sets, induced graph with all matched nodes including isolates
**Result:** Exact WCC 17598 users/29715 edges; normalized 17636 users/29825 edges. WCC changes; existing model cohort remains preserved.
**Current statement classification:** needs methodological correction
**Exact recommended replacement:** “Literal matching selected 17,598 users and 29,715 directed edges in the largest weak component; trimming whitespace, removing one leading @ and casefolding selected 17,636 users and 29,825 edges, so the previously fitted models retain the original cohort while the matching sensitivity is reported separately.”
**Supporting output:** `results/tables/matching_sensitivity_comparison.csv`

## 2. 99.8638% coverage denominator

**Issue:** 99.8638% coverage denominator
**Reviewer concern:** Raw rows and unique pairs mixed
**What was checked:** Literal and normalized unique pairs, reconstructed overlap, per-row membership
**Result:** 109217/109366 = 99.86376022%; precision 109217/109219 = 99.99816882%.
**Current statement classification:** correct
**Exact recommended replacement:** “Reconstructed interactions overlap 109,217 of 109,366 normalized unique supplied directed pairs (99.8638% pair coverage), and 109,217 of 109,219 reconstructed pairs (99.9982% precision-style overlap); the supplied file contains 109,443 rows.”
**Supporting output:** `results/tables/reconstruction_overlap_reconciliation.csv`

## 3. 94.67% vs 94.61% lexical agreement

**Issue:** 94.67% vs 94.61% lexical agreement
**Reviewer concern:** Disputed percentage and zero-polarity definition
**What was checked:** TextBlob recomputed on every raw Tweet using exact saved formula, saved counts checked
**Result:** 80937/85495 = 94.66869408%; rounds 94.67%, not 94.61%.
**Current statement classification:** correct
**Exact recommended replacement:** “Agreement between the binary Target label and indicator(TextBlob polarity > 0), with zero polarity assigned to the nonpositive class, is 80,937/85,495 (94.67%); Target-0 and Target-1 row agreements are 97.62% and 91.68%, respectively.”
**Supporting output:** `results/tables/target_textblob_agreement_recheck.csv`

## 4. Edge-direction semantics

**Issue:** Edge-direction semantics
**Reviewer concern:** Recorded action edges mistaken for verified content flow
**What was checked:** Reconstruction author/reference semantics, layer controls and controlled weight reallocation
**Result:** Retweet reversal is plausible content flow; mention and inferred reply have no verified content direction. Structural adjacency/spectral transpose invariance does not imply reach invariance.
**Current statement classification:** needs wording correction
**Exact recommended replacement:** “Recorded edges point from tweet author to a text-referenced account; retweet edges are reversed for a plausible content-flow sensitivity, whereas mention and inferred-reply direction remain uncertain and are tested in both orientations rather than treated as verified information transmission.”
**Supporting output:** `results/tables/typed_direction_assumptions.txt`

## 5. Typed retweet/reply/mention layers

**Issue:** Typed retweet/reply/mention layers
**Reviewer concern:** Pooling hides interaction semantics
**What was checked:** Separate weighted graphs and node-universe/layer overlaps
**Result:** 16 retweet events (14 pairs/20 endpoints), 41,024 inferred replies and 88,564 mentions. Native metadata absent; inferred reply is a leading-handle heuristic.
**Current statement classification:** needs methodological correction
**Exact recommended replacement:** “The 129,604 reconstructed events comprise 16 text-supported retweets, 41,024 inferred replies and 88,564 mentions; the sparse retweet layer and absent native reply/retweet metadata preclude broad conclusions about retweet propagation or verified reply behavior.”
**Supporting output:** `results/tables/typed_layer_network_statistics.csv`

## 6. Layer orientation mixing

**Issue:** Layer orientation mixing
**Reviewer concern:** Pooled association or global shuffle may reflect degree
**What was checked:** 2000 degree/strength-stratified and 2000 global permutations per layer; separate Holm families
**Result:** Primary Holm p: retweet=0.6777, reply_inferred=0.6777, mention=0.2984. Retweet global null has undefined draws and p=NA.
**Current statement classification:** needs wording correction
**Exact recommended replacement:** “None of the three layer-wise orientation correlations is supported after Holm correction under the degree/strength-stratified permutation null; these descriptive full-history associations do not establish political homophily.”
**Supporting output:** `results/tables/layer_orientation_permutation_tests.csv`

## 7. Static vs time-respecting reach

**Issue:** Static vs time-respecting reach
**Reviewer concern:** Static paths may violate event chronology
**What was checked:** 500 seeds per stratum where nodes available; non-decreasing timestamp earliest arrival
**Result:** High-out-strength mean temporal/static reach 0.748735 recorded (0.725722, 0.771174); 0.523778 layer-aware (0.498249, 0.549556). Stratum-dependent reductions; many static sinks ratio=NA.
**Current statement classification:** needs methodological correction
**Exact recommended replacement:** “For the 500 highest-out-strength seeds, time-respecting reach retains on average 74.87% of static descendants in recorded direction and 52.38% under the retweet-reversed layer-aware convention, with substantial variation by seed stratum and no reach ratio assigned to static sinks.”
**Supporting output:** `results/tables/time_respecting_reachability_summary.csv`

## 8. Net-of-seed propagation reach

**Issue:** Net-of-seed propagation reach
**Reviewer concern:** Final totals dominated by initial seeds
**What was checked:** 2000 paired Monte Carlo runs per configuration/direction; identity-based seed exclusion
**Result:** Full balanced recorded mean secondary 0.269500 (95% MC CI 0.246461, 0.292539); amplification 0.005390. Weighted baseline secondary 2.503500.
**Current statement classification:** needs wording correction
**Exact recommended replacement:** “With 25 initial seeds per type, beta_0=beta_1=0.14 and recovery 0.25, the full recorded-direction model generates a mean of 0.2695 secondary adoptions (95% Monte Carlo CI 0.2465–0.2925) within 25 steps, corresponding to 0.00539 secondary adoptions per initial seed; seed-inclusive cascade totals should not be interpreted as amplification.”
**Supporting output:** `results/tables/secondary_reach_simulation_results.csv`

## 9. Amount of actual competition

**Issue:** Amount of actual competition
**Reviewer concern:** Exclusive adoption does not imply many contested nodes
**What was checked:** Positive-probability susceptible opportunities vs successful simultaneous resolution
**Result:** Full balanced recorded mean ever-both fraction 0.00058964; actual resolution events/run 0.075500. Exposure denominator excludes 50 seeds.
**Current statement classification:** needs wording correction
**Exact recommended replacement:** “In the balanced full-model recorded-direction simulation, the mean fraction of non-seed nodes ever offered both contagions before adoption is 0.0590%, and the mean number of actual simultaneous competition-resolution events is 0.0755 per run; these are model exposure opportunities, not observed Twitter contacts.”
**Supporting output:** `results/tables/competition_exposure_overlap.csv`

## 10. Seed-inclusive vs secondary competition outcome

**Issue:** Seed-inclusive vs secondary competition outcome
**Reviewer concern:** Seed allocation can determine outcome statistic
**What was checked:** Signed (C0-C1)/(C0+C1) and original 5% mixed-close rule on totals and secondary counts
**Result:** Both definitions saved; zero secondary-adoption denominator is NA, not zero or tie. No equivalence margin defined.
**Current statement classification:** needs methodological correction
**Exact recommended replacement:** “Competitive outcomes are reported primarily using non-seed Target-0 and Target-1 adoptions; runs with no secondary adoption have undefined outcomes, and seed-inclusive outcomes are retained only as a sensitivity because initial allocation can dominate them.”
**Supporting output:** `results/tables/competition_outcome_definition_comparison.csv`

## 11. Spectral-threshold validity

**Issue:** Spectral-threshold validity
**Reviewer concern:** Tiny recurrent core and self-loops may dominate
**What was checked:** Exact SCC-block spectra, loop removal, three directions, explicit normalization
**Result:** Largest SCC 17; radius 20.000000 -> 9.043208; loop-dominated singleton ictcimpact. Diagnostic beta 0.425000 -> 0.939932; all-reversed invariant.
**Current statement classification:** needs wording correction
**Exact recommended replacement:** “The adjacency spectral radius is dominated by a self-loop on a singleton SCC and decreases from 20.0000 to 9.0432 after loop removal; because the largest SCC has only 17 nodes and epidemic-model assumptions were not validated, the associated critical-beta calculation is a loop-sensitive spectral diagnostic rather than an established epidemic threshold.”
**Supporting output:** `results/tables/spectral_threshold_audit.csv`

## 12. Forecasting near-chance interpretation

**Issue:** Forecasting near-chance interpretation
**Reviewer concern:** Accuracy may reflect imbalance and incomparable cohorts
**What was checked:** Saved forecasts only, 156 summary means independently reproduced from saved weekly means
**Result:** Graph-only balanced accuracy .5006–.5091; Persistence .5403; history logistic .5535 and history+structural MLP .5536. No corrected macro-F1 superiority over Persistence established.
**Current statement classification:** needs wording correction
**Exact recommended replacement:** “Graph-only weekly models achieve balanced accuracy close to 0.50, whereas Persistence and history-based models show modest signal; historical label information, rather than demonstrated incremental graph-message value, accounts for the strongest saved results, and the original monthly majority-like accuracy must not be presented as strong balanced forecasting.”
**Supporting output:** `results/tables/all_forecasting_models_reconciled.csv`

## 13. Aligned GRU definition

**Issue:** Aligned GRU definition
**Reviewer concern:** Different GRUs conflated
**What was checked:** Saved architecture/configuration and cutoff construction compared
**Result:** Original GRU direct six-feature GRU differs from graph-aligned projected raw-feature GRU. Temporal GCN fixed graph weights, not parameter-evolving EvolveGCN.
**Current statement classification:** needs wording correction
**Exact recommended replacement:** “The aligned GRU replaces the Temporal GCN graph aggregation with raw weekly structural feature sequences while retaining its projection, GRU head and evaluation pipeline; it is distinct from the original lagged node-only GRU and is reported separately.”
**Supporting output:** `results/tables/aligned_GRU_definition.txt`

## 14. Static vs temporal node universes

**Issue:** Static vs temporal node universes
**Reviewer concern:** Population denominators differ across analyses
**What was checked:** Exact/normalized static cohorts, reconstructed endpoint union, weekly prediction eligibility
**Result:** Original WCC 17,598; normalized WCC 17,636; full event endpoint universe 60,219; weekly prediction uses eligible user-weeks, not every static node.
**Current statement classification:** needs wording correction
**Exact recommended replacement:** “The original static diffusion cohort contains 17,598 users, the normalized-matching sensitivity contains 17,636 largest-WCC users, and the full reconstructed temporal network contains 60,219 endpoint accounts; weekly forecasting evaluates an eligible user-week cohort and these denominators are not interchangeable.”
**Supporting output:** `results/tables/forecasting_population_and_information_boundary.csv`

## 15. Publication figure corrections

**Issue:** Publication figure corrections
**Reviewer concern:** Mixed units, hidden zeros/core, seed-inclusive outcome and missing uncertainty
**What was checked:** New figures only; source CSVs, SVG/400-DPI PNG, readable unit/scale/CI labels
**Result:** Separated population units and weekly panels, visible exact polarity zero, CCDFs without power-law claims, log lollipop bow-tie, net-of-seed competition, panel-specific heatmap scales and forecast references.
**Current statement classification:** needs methodological correction
**Exact recommended replacement:** “Corrected figures separate population units and activity/class-share panels, display exact zero polarity, report degree/strength CCDFs without a power-law claim, and use secondary-adoption outcomes and explicitly labelled uncertainty and chance references.”
**Supporting output:** `results/tables/../figures/reviewer_corrected/`

## 16. Reproducibility and governance gaps

**Issue:** Reproducibility and governance gaps
**Reviewer concern:** Environment, assumptions, provenance and hashes incomplete
**What was checked:** Protected-file baseline/postcheck, scripts/configuration and output hashes packaged
**Result:** Raw tweet text not included; original package versions/dataset provenance, validated native reply direction, causal adoption and public licence/DOI unavailable.
**Current statement classification:** needs methodological correction
**Exact recommended replacement:** “The correction analyses are accompanied by scripts, fixed random seeds, configuration, software versions, protected-input SHA-256 verification, figure source data and output manifests; the source tweet dataset is excluded and no public repository, DOI, redistribution licence or verified native interaction metadata is claimed.”
**Supporting output:** `results/tables/reproducibility_manifest.csv`

## Remaining limits

Native reply/retweet metadata, true information exposure/adoption, causal edge direction and completeness of the text-supported interaction network cannot be recovered from these inputs. Retweet layer inference rests on 16 events, with only nine orientation-labelled unique pairs; some global-shuffle correlations are undefined. The original manuscript spectral operator/version and dataset acquisition provenance are unavailable. The original TextBlob environment version was not saved; current recomputation exactly matches saved sign counts. No scientifically justified equivalence margin was supplied or invented. Finite 25-step simulations are conditional sensitivity analyses. Full-history orientation is descriptive, not a prior-time causal label. Temporal seed-bootstrap intervals describe seeds in this observed network, not network-population uncertainty. No new forecasting, model training or manuscript editing occurred.
