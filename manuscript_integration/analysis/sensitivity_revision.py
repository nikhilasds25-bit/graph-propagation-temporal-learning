from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT; TAB=OUT/'results_tables_revised'; FIG=OUT/'figures_revised'
BLUE='#0072B2'
nodes=pd.read_csv(TAB/'node_features_revision.csv',index_col=0); nodes.index=nodes.index.astype(str); idx={n:i for i,n in enumerate(nodes.index)}; N=len(nodes)
ed=pd.read_csv(ROOT/'data'/'raw'/'network_edges.csv'); ed.Source=ed.Source.astype(str); ed.Target=ed.Target.astype(str); ed=ed[ed.Source.isin(idx)&ed.Target.isin(idx)].copy(); src=ed.Source.map(idx).to_numpy(int); dst=ed.Target.map(idx).to_numpy(int); w=ed.Weight.to_numpy(float); m=src!=dst; src,dst,w=src[m],dst[m],w[m]
wfac=np.log1p(w)/np.log1p(w.max()); infl=nodes.structural_influence_score.fillna(.5).to_numpy(float); rneg=nodes.target0_ratio.fillna(.5).to_numpy(float); q=wfac*(.5+.5*infl[src])*(.5+.5*rneg[src])
adj=[[] for _ in range(N)]
for s,d,qq in zip(src,dst,q): adj[s].append((d,float(qq)))
seed_fracs=[.001,.01]; gammas=[.10,.25,.40]; betas=[.1,.25,.5,1.0,1.5]
def sim(beta,gamma,seed_count,seed):
 r=np.random.default_rng(seed); st=np.zeros(N,np.int8); seeds=r.choice(N,seed_count,replace=False); st[seeds]=1; active=list(map(int,seeds)); ever=np.zeros(N,bool); ever[seeds]=1
 for t in range(120):
  if not active: break
  new=set()
  for u in active:
   for d,qq in adj[u]:
    if st[d]==0 and r.random()<min(1,beta*qq): new.add(d)
  nxt=[]
  for u in active:
   if r.random()<gamma: st[u]=2
   else:nxt.append(u)
  for d in new:
   if st[d]==0: st[d]=1;ever[d]=1;nxt.append(d)
  active=nxt
 return ever.sum()
rows=[]
for sf in seed_fracs:
 sc=max(1,round(sf*N))
 for g in gammas:
  for b in betas:
   for run in range(20): rows.append({'seed_fraction':sf,'gamma':g,'beta':b,'run':run,'final_fraction':sim(b,g,sc,int(1e6+sf*1e6+g*1e5+b*1e4+run))/N})
res=pd.DataFrame(rows); res.to_csv(TAB/'sensitivity_grid_runs.csv',index=False)
summary=res.groupby(['seed_fraction','gamma','beta']).final_fraction.agg(['mean','std','count']).reset_index();summary['ci95']=1.96*summary['std']/np.sqrt(summary['count']); summary.to_csv(TAB/'sensitivity_grid_summary.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),sharey=True)
for ax,sf in zip(axs,seed_fracs):
 pv=summary[summary.seed_fraction==sf].pivot(index='gamma',columns='beta',values='mean').reindex(index=gammas,columns=betas); im=ax.imshow(pv.values,origin='lower',aspect='auto',cmap='viridis',extent=[-.5,len(betas)-.5,-.5,len(gammas)-.5]); ax.set_xticks(range(len(betas)));ax.set_xticklabels(betas);ax.set_yticks(range(len(gammas)));ax.set_yticklabels(gammas);ax.set_xlabel('Transmission multiplier β');ax.set_title(f'Seed fraction {100*sf:.1f}%')
 for_i=0
axs[0].set_ylabel('Recovery probability γ'); cb=fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.025,pad=.03);cb.set_label('Mean final reached fraction'); fig.suptitle('Sensitivity of reach to transmission, recovery, and seed fraction',fontsize=10);fig.tight_layout(rect=[0,0,.94,.95]);fig.savefig(FIG/'fig_sensitivity_grid.pdf',bbox_inches='tight');fig.savefig(FIG/'fig_sensitivity_grid.png',dpi=220,bbox_inches='tight');plt.close(fig)
print(summary.to_string(index=False))
