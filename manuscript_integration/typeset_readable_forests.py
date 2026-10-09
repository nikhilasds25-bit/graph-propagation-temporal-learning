"""Reflow copied vector plots at manuscript size; use only saved source CSVs."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent/'final_manuscript'
plt.rcParams.update({'font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
def save(fig,stem):
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/'figures'/(stem+'.'+ext),dpi=400,bbox_inches='tight')
    plt.close(fig)

z=pd.read_csv(OUT/'supporting_evidence/forecasting_reconciled_uncertainty_source.csv')
fig,axes=plt.subplots(2,1,figsize=(6.8,8.1),layout='constrained')
labels=z.model.str.replace('Temporal GCN / GRU-over-GCN','Temporal GCN (GRU-over-GCN)',regex=False).str.replace('History Logistic','History-only logistic',regex=False)
for ax,metric,title in zip(axes,['balanced_accuracy','class_0_PR_AUC'],['Balanced accuracy','Class-0 average precision']):
    for y,(_,r) in enumerate(z.iterrows()):
        ax.plot([r[metric+'_ci95_low'],r[metric+'_ci95_high']],[y,y],color='#0072B2',lw=1.2)
        ax.plot(r[metric],y,'o',color='#0072B2',ms=3.5)
    ax.set_yticks(range(len(z)),labels,fontsize=8.8)
    ax.set_xlabel(title+' (saved 95% interval)')
    ref=.5 if metric=='balanced_accuracy' else z.iloc[0].no_skill_class_0_PR_AUC
    pers=z[z.model=='Persistence'].iloc[0][metric]
    ax.axvline(ref,color='#666666',ls=':',lw=1.2,label='Chance / prevalence')
    ax.axvline(pers,color='#D55E00',ls='--',lw=1.2,label='Persistence mean')
    ax.legend(loc='lower right',fontsize=7.5)
    ax.grid(axis='x',alpha=.15)
save(fig,'forecasting_reconciled_uncertainty')

z=pd.read_csv(OUT/'supporting_evidence/simulation_direction_forest_source.csv')
fig,ax=plt.subplots(figsize=(6.8,5.8),layout='constrained')
names={'baseline_beta_005':'Edge-only, beta 0.05','symmetric_baseline':'Edge-only, beta 0.14',
       'baseline_beta_030':'Edge-only, beta 0.30','full_balanced':'Full, balanced',
       'Target_0_advantage':'Target-0 advantage','Target_1_advantage':'Target-1 advantage',
       'asymmetric_40_10':'Seeds 40 / 10','asymmetric_10_40':'Seeds 10 / 40'}
labels=[]
for y,(_,r) in enumerate(z.iterrows()):
    comparison='Layer-aware' if 'layer_aware' in r.comparison else 'Reversed'
    labels.append(names[r.configuration]+'; '+comparison+'\nminus recorded')
    color='#0072B2' if comparison=='Layer-aware' else '#D55E00'
    ax.plot([r.ci95_low,r.ci95_high],[y,y],color=color,lw=1.2)
    ax.plot(r['mean'],y,'o',color=color,ms=3.5)
ax.set_yticks(range(len(z)),labels,fontsize=8.5)
ax.axvline(0,color='#666666',ls=':',lw=1)
ax.set_xlabel('Paired difference in secondary adoptions (95% Monte Carlo CI)')
ax.grid(axis='x',alpha=.15)
save(fig,'simulation_direction_forest')
print('Reflowed saved forecast/direction intervals for readable manuscript-size labels.')
