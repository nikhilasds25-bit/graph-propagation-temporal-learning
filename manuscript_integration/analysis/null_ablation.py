from pathlib import Path
import numpy as np,pandas as pd,json
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT; TAB=OUT/'results_tables_revised'; FIG=OUT/'figures_revised'
BLUE='#0072B2'; VERM='#D55E00'; GREY='#7F7F7F'; GREEN='#009E73'; ORANGE='#E69F00'
nodes=pd.read_csv(TAB/'node_features_revision.csv',index_col=0); nodes.index=nodes.index.astype(str); idx={n:i for i,n in enumerate(nodes.index)}; N=len(nodes)
edges=pd.read_csv(ROOT/'data'/'raw'/'network_edges.csv'); edges.Source=edges.Source.astype(str); edges.Target=edges.Target.astype(str); edges=edges[edges.Source.isin(idx)&edges.Target.isin(idx)].copy(); src=edges.Source.map(idx).to_numpy(int); dst=edges.Target.map(idx).to_numpy(int); w=edges.Weight.to_numpy(float); keep=src!=dst; src,dst,w=src[keep],dst[keep],w[keep]
wfac=np.log1p(w)/np.log1p(w.max()); infl=nodes.structural_influence_score.fillna(.5).to_numpy(float); rneg=nodes.target0_ratio.fillna(.5).to_numpy(float); Mstruct=.5+.5*infl; Mneg=.5+.5*rneg
q_full=wfac*Mstruct[src]*Mneg[src]
rng=np.random.default_rng(7)
# fair variants preserve mean q approximately/exactly
q_weightshuffle=rng.permutation(wfac)*Mstruct[src]*Mneg[src]; q_weightshuffle*=q_full.mean()/q_weightshuffle.mean()
q_influence_flat=wfac*np.mean(Mstruct[src])*Mneg[src]; q_influence_flat*=q_full.mean()/q_influence_flat.mean()
q_orientation_flat=wfac*Mstruct[src]*np.mean(Mneg[src]); q_orientation_flat*=q_full.mean()/q_orientation_flat.mean()
# label shuffle moves orientation across nodes while preserving distribution
rsh=rng.permutation(rneg); q_labelshuffle=wfac*Mstruct[src]*(.5+.5*rsh[src]); q_labelshuffle*=q_full.mean()/q_labelshuffle.mean()
variants={'Full model':q_full,'Weight shuffled':q_weightshuffle,'Influence flattened':q_influence_flat,'Orientation flattened':q_orientation_flat,'Orientation shuffled':q_labelshuffle}

def mkadj(q):
 a=[[] for _ in range(N)]
 for s,d,qq in zip(src,dst,q): a[s].append((d,float(qq)))
 return a
adjs={k:mkadj(v) for k,v in variants.items()}
gamma=.25; beta=1.5 # deliberately strong; p clipped
seed_count=round(.01*N)
def sim(adj,seed):
 r=np.random.default_rng(seed); st=np.zeros(N,np.int8); seeds=r.choice(N,seed_count,replace=False); st[seeds]=1; active=list(map(int,seeds)); ever=np.zeros(N,bool); ever[seeds]=1
 for t in range(100):
  if not active: break
  new=set()
  for u in active:
   for d,q in adj[u]:
    if st[d]==0 and r.random()<min(1,beta*q): new.add(d)
  nxt=[]
  for u in active:
   if r.random()<gamma: st[u]=2
   else: nxt.append(u)
  for d in new:
   if st[d]==0: st[d]=1; ever[d]=1; nxt.append(d)
  active=nxt
 return int(ever.sum())
rows=[]
for vi,(name,adj) in enumerate(adjs.items()):
 for run in range(40): rows.append({'variant':name,'run':run,'final_size':sim(adj,900000+run)}) # common random seeds
res=pd.DataFrame(rows); res.to_csv(TAB/'mean_preserving_ablation_runs.csv',index=False)
# paired delta vs full, bootstrap CI
full=res[res.variant=='Full model'].set_index('run').final_size
summ=[]
for name in variants:
 vals=res[res.variant==name].set_index('run').final_size
 d=(vals-full).to_numpy(); boot=[]; br=np.random.default_rng(1)
 for _ in range(5000): boot.append(br.choice(d,len(d),replace=True).mean())
 lo,hi=np.percentile(boot,[2.5,97.5]); summ.append({'variant':name,'mean_final_size':vals.mean(),'delta_vs_full':d.mean(),'ci95_lo':lo,'ci95_hi':hi})
s=pd.DataFrame(summ); s.to_csv(TAB/'mean_preserving_ablation_summary.csv',index=False)
plot=s[s.variant!='Full model'].iloc[::-1]; fig,ax=plt.subplots(figsize=(5.2,2.8)); y=np.arange(len(plot)); x=plot.delta_vs_full.to_numpy(); lo=x-plot.ci95_lo.to_numpy(); hi=plot.ci95_hi.to_numpy()-x; ax.errorbar(x,y,xerr=np.vstack([lo,hi]),fmt='o',color=BLUE,capsize=3); ax.axvline(0,color='k',lw=.8); ax.set_yticks(y); ax.set_yticklabels(plot.variant); ax.set_xlabel('Change in final reached users vs full model'); ax.set_title('Mean-preserving component controls (40 paired runs)'); fig.tight_layout(); fig.savefig(FIG/'fig_mean_preserving_ablation.pdf',bbox_inches='tight'); fig.savefig(FIG/'fig_mean_preserving_ablation.png',dpi=220,bbox_inches='tight'); plt.close(fig)
print(s.to_string(index=False))
