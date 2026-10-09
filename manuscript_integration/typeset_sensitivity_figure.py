"""Typeset a copied saved figure-source grid; no simulation or estimation."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'final_manuscript'
df=pd.read_csv(OUT/'supporting_evidence/sensitivity_amplification_independent_scales_source.csv')
plt.rcParams.update({'font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(13,3.4),layout='constrained')
for ax,seeds in zip(axes,[5,10,25]):
    z=df[df.seed_count==seeds]
    matrix=z.pivot(index='recovery',columns='base_transmission',values='amplification')
    im=ax.imshow(matrix.to_numpy(),cmap='cividis',vmin=0,vmax=matrix.to_numpy().max(),aspect='auto')
    ax.set_xticks(range(len(matrix.columns)),[f'{v:g}' for v in matrix.columns])
    ax.set_yticks(range(len(matrix.index)),[f'{v:g}' for v in matrix.index])
    ax.set_xlabel(r'$\beta_0=\beta_1$')
    ax.set_ylabel('Quiescence probability')
    ax.set_title(f'{seeds} seeds per class\nrange 0 to {matrix.to_numpy().max():g}')
    fig.colorbar(im,ax=ax,label='Mean amplification',fraction=.045,pad=.04)
fig.savefig(OUT/'figures/sensitivity_amplification_independent_scales.pdf',bbox_inches='tight')
fig.savefig(OUT/'figures/sensitivity_amplification_independent_scales.png',dpi=400,bbox_inches='tight')
fig.savefig(OUT/'figures/sensitivity_amplification_independent_scales.svg',bbox_inches='tight')
plt.close(fig)
print('Typeset saved sensitivity grid with quiescence labels and panel-specific ranges.')
