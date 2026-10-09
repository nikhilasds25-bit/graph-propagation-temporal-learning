# Final Overleaf manuscript

**Title:** Graph-Based Information Propagation and Temporal Learning in a Directed Indian Political Twitter Network  
**Author:** Nikhil A S

This folder is the clean Overleaf-ready version of the paper. `main.tex` is the editable source and `main.pdf` is the compiled manuscript.

## What is included in the final paper

The manuscript uses the completed analysis rather than the earlier monthly GNN experiment. The final results now cover:

- the 17,598-user directed WCC and its 17-node largest SCC;
- orientation-label checks and the full-data TextBlob comparison;
- direction-aware centrality and orientation-mixing null tests;
- threshold, sensitivity, asymmetric-competition, and mean-preserving diffusion controls;
- reconstruction of 129,604 timestamped interactions across 109,219 unique directed pairs;
- 25 weekly dynamic graph snapshots from 2022-10-10 to 2023-03-27;
- leakage-controlled rolling-origin forecasting over 15 test weeks;
- Majority, Persistence, Logistic Regression, MLP, GRU, GCN, GraphSAGE, GAT, and Temporal GCN comparisons;
- matched history-aware controls showing that history + reversed GAT does not outperform history + structural MLP;
- ordinary and moving-block bootstrap intervals and Holm-adjusted testing for the central history-aware comparison.

The paper deliberately uses **Temporal GCN** for the GRU-over-GCN implementation and does not call it EvolveGCN-O/H.

## Package layout

- `main.tex` - final LaTeX source
- `main.pdf` - compiled manuscript
- `figures_revised/` - publication figures used by the manuscript
- `results_tables_revised/` - numerical source summaries and figure data available in this package
- `analysis/` - scripts for the propagation/network analyses already included in the project
- `requirements.txt` - recorded Python environment for those packaged scripts
- `data/raw/README.txt` - expected raw filenames and checksums; raw tweet text is not redistributed
- `references.bib` - reusable bibliography entries

## Reproducibility note

The newest weekly forecasting experiments were executed in the local `sentiment_graph_diffusion` project. The Overleaf archive includes the aggregate values used in the manuscript and derived summary CSVs. For a public research release, the seed-level forecasting outputs and the corresponding local forecasting scripts should be deposited alongside this manuscript.

## Originality note

The manuscript has been rewritten in original academic prose and source claims are attributed. A local exact-phrase check against the reference article and the three review PDFs found no long verbatim overlap outside titles and bibliographic citations. This is not a substitute for an institutional similarity service such as Turnitin or iThenticate, and no AI detector score can be guaranteed.
