# Manuscript integration change log

Date: 9 October 2026

Final source: `final_manuscript/main.tex`  
Final PDF: `final_manuscript/main.pdf`  
Overleaf package: `final_manuscript_overleaf.zip`

## Scope and preservation

This was manuscript integration only. No experiment, simulation, model training, forecasting, permutation test, or statistical estimation was rerun. Existing canonical analysis outputs were read and copied into a new manuscript folder. Figure reflow uses saved figure-source values and changes presentation only.

The supplied `main.tex` was copied before editing. It remains unchanged at `manuscript_integration/main.tex`; `main_before_integration.tex` records that input. The revised deliverable is the separate `final_manuscript/main.tex`. `source_main.tex` and `source_references.bib` remain untouched. The title, Nikhil A S's name, MSc Data Science affiliation, Digital University Kerala address, email, and October 2026 date were retained. `final_manuscript/references.bib` is byte-identical to the supplied bibliography and retains all 22 entries.

The manuscript retains its overall sequence: Introduction; Related Work; Data and Population Construction; Label Validation; Directed Structure; Propagation Model; Spectral/Sensitivity/Competition; Component/Direction Controls; Weekly Forecasting; Discussion; Limitations; Ethics; Reproducibility; Conclusion; author/funding/interests statements; References. Time-respecting reachability is a prominent subsection within directed structure. Two section titles were adjusted to describe the supported evidence: the threshold section now identifies spectral diagnostics, and the component-control section reports the canonical direction and probability-factor controls. Unsupported numerical claims from the source's mean-preserving-control section were not carried into the final manuscript.

## Evidence integration

| Issue | Final manuscript treatment | Canonical evidence |
|---|---|---|
| Population matching | Literal WCC: 17,598 users / 29,715 edges. Normalized sensitivity: 17,636 / 29,825. No experiment cohort reassignment. | `matching_sensitivity_comparison.csv` |
| Raw rows versus pairs | 109,443 supplied rows; 109,366 unique normalized supplied pairs; 109,219 reconstructed pairs; 109,217 overlap. Coverage 99.8638%; precision-style overlap 99.9982%, with both denominators explicit. | `reconstruction_overlap_reconciliation.csv` |
| Lexical agreement | 80,937 / 85,495 = 94.67%; row agreements 97.62% and 91.68%. All six class/sign counts replaced with canonical counts; exact zero remains visible. | `target_textblob_agreement_recheck.csv` |
| Typed interaction evidence | 16 text-supported RT events, 41,024 inferred replies, 88,564 mentions. Inferred replies are explicitly a leading-handle heuristic. | `typed_layer_network_statistics.csv` |
| Direction semantics | Author to referenced account is recorded interaction direction. Only RT shares reverse in the layer-aware sensitivity; mentions/replies have unverified flow direction. | `typed_direction_assumptions.txt` |
| Orientation mixing | Unique-pair units, full-history ratios, missing-endpoint exclusions, 2,000 stratified/global permutations per layer, separate Holm families. Primary Holm p values 0.677661, 0.677661, 0.298351 support no homophily claim. Sparse RT and undefined global-null inference documented. | `layer_orientation_permutation_tests.csv` |
| Temporal reachability | Non-decreasing timestamps, waiting/equal-time closure, seed exclusion, undefined sink ratios, four strata, seed-bootstrap interpretation. High-out-strength recorded mean 0.7487353198 [0.7257221343, 0.7711735446]; layer-aware 0.5237778023 [0.4982491182, 0.5495557165]. | `time_respecting_reachability_summary.csv` |
| Temporal hop interpretation | Earliest-arrival witness depth is descriptive, neither maximum temporal walk nor guaranteed fewest-hop foremost path. | `temporal_hop_interpretation_clarification.txt` |
| Simulation methods | Saved max-weight normalization by 34, structural prominence, PageRank and historical orientation/polarity factors, one union adoption draw followed by conditional class resolution, synchronous quiescence. Static hypothetical process with November features; no observed-adoption calibration claim. | `reproducibility_package/simulator_settings.txt` and archived reference scripts |
| Secondary propagation | Full balanced recorded mean 0.2695 [0.246461, 0.292539]; amplification 0.00539 [0.00492923, 0.00585077]. Eight recorded configurations in a saved-value table. Seed-inclusive totals are excluded from amplification. | `secondary_reach_simulation_results.csv` |
| Competition intensity | Mean non-seed contested fraction 0.05896398%; actual resolution mean 0.0755/run. Non-seed, exposed-union, and conditional-adopter denominators are separated. | `competition_exposure_overlap.csv` |
| Non-seed outcomes | Primary signed outcome uses secondary adoptions. 1,532 zero-secondary runs are undefined; 468 defined runs remain. Seed-inclusive comparison is a sensitivity. No equivalence claim. | `secondary_reach_simulation_results.csv`, `competition_outcome_definition_comparison.csv` |
| Spectral diagnostic | Radius 20.000000 with loops, 9.0432080504 without; largest SCC 17. Operator normalization, singleton loop dominance, and transpose invariance explained. No established epidemic-threshold claim. | `spectral_threshold_audit.csv` |
| Forecasting cohorts | Exact primary test count: 12,126 eligible user-weeks, 6,889 users, 15 weeks. History-available and monthly tasks remain separate. | `forecasting_population_and_information_boundary.csv` |
| Complete forecasting table | All 27 saved reconciled rows, six metrics, available 95% intervals, explicit NA/CI-unavailable entries. Includes all required weekly models and separate cohort/task sensitivities. | `all_forecasting_models_reconciled.csv` |
| GRU definitions | Original direct-feature GRU and graph-aligned projected GRU defined separately. Temporal GCN is fixed-weight GRU-over-GCN; monthly saved model identifier is displayed by its fixed-graph architecture. | `aligned_GRU_definition.txt`, saved model methodology |
| Forecasting interpretation | Graph-only balanced accuracy near 0.50; Persistence and history supply modest signal. Metric-specific Holm results retained; no broad graph superiority or temporal improvement claim. Matched history plus reversed GAT is inferior to history plus structural MLP. | `history_graph_findings_addendum.txt` and reconciled results |

All canonical filenames above refer to `results/tables/` unless another directory is specified. The master report and issue index guided integration, while numerical values came from their linked saved evidence.

## Writing, figures, and package

- Replaced the abstract with a 193-word evidence-based abstract; no Keywords.
- Rebuilt the Discussion around directed recurrence, chronology, tiny secondary spread/rare competition, and difficult forecasting with modest history signal.
- Removed revision-history narration and unsupported claims about negative-information dominance, credibility, causal contagion, political homophily, and graph-model superiority.
- Kept susceptible state `U`, prominence `H`, simulation outcome `B`, paired model differences `d`, Spearman `rho_s`, and adjacency radius `rho(A)` distinct.
- Used quiescence probability consistently and distinguished initial seed nodes from pseudorandom seeds.
- Added all requested limitations and the exact funding sentence: "No external funding was received for this study."
- Removed the duplicate References heading and used the supplied bibliography once.
- Used 17 corrected figures with cited captions and copied source CSVs. Population units and weekly panels stay separate; zero polarity and the 17-node SCC remain visible; no unsupported power-law fit, dual y-axis, or 3D plot is introduced.
- Typeset three manuscript-only figure copies from their saved source grids/intervals: sensitivity (quiescence labels and independent scales), forecasting (stacked readable panels), and paired direction controls (readable labels). Saved numerical values were retained. Each has SVG and 400-DPI PNG output, with PDF used where helpful.
- Described the reproducibility package accurately: it contains scripts/configuration/version/seed/hash/assumption records and links canonical derived tables and figure source CSVs through manifests. These linked outputs are not all physically embedded in the existing reproducibility folder.
- The clean ZIP includes primary source, five table/numeric fragments, the unchanged bibliography, referenced figures, copied derived evidence, README, and a SHA-256/provenance manifest. It excludes raw tweets, checkpoints, compiler binaries/cache, LaTeX intermediates/logs, manuscript backups, and the supplied draft PDF. The compiled final PDF is delivered separately.

## Final validation

- Final source compiled successfully twice with Tectonic 0.17.0: `qa/compile_final_1.txt` and `qa/compile_final_2.txt`. The TeX bundle cache and compiler tools stay inside the project.
- Final PDF: 25 pages. All pages rendered with Poppler and visually reviewed, with enlarged checks of the temporal table, sensitivity labels, forecasting figure, and both landscape forecasting-table pages.
- Final LaTeX log: no errors, undefined references/citations, missing characters, overfull boxes, or warnings. A system-font configuration notice in the first final invocation was resolved for the second; the final TeX log and final compile output are clean.
- Source audit: all 17 figures and seven tables cited, all referenced files/source CSVs present, no undefined source references.
- All 27 forecasting rows' displayed means and intervals independently checked against the canonical CSV. Simulation table and numerical bindings were typeset directly from saved results.
- PDF extraction check: one References heading, zero unresolved `??` markers, zero characters outside page bounds.
- Requested language search inspected. Remaining "original" identifies the required original-GRU model or canonical asset filename; "earlier" denotes prior-time inputs; "corrected" occurs only in the canonical directory path; epidemic-threshold occurrences explicitly state unvalidated assumptions. The EvolveGCN name survives only as the supplied literature reference, not a model-performance claim.
- SHA-256 verification confirms all 1,066 protected result/figure/reproducibility/source-backup files remain byte-identical. The primary supplied `main.tex` also matches its pre-integration copy. No analysis output was modified.

Audit records: `qa/manuscript_source_audit.json`, `qa/forecast_table_audit.json`, `qa/pdf_layout_audit.json`, and `protected_integration_hashes.json`. ZIP integrity is verified at packaging.
