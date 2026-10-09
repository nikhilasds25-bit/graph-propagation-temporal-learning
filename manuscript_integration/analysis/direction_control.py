from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; TAB=ROOT/'source'/'results_tables_revised'
nodes=pd.read_csv(TAB/'node_features_revision.csv',index_col=0); nodes.index=nodes.index.astype(str); idx={n:i for i,n in enumerate(nodes.index)}; N=len(nodes)
ed=pd.read_csv(ROOT/'data'/'raw'/'network_edges.csv');ed.Source=ed.Source.astype(str);ed.Target=ed.Target.astype(str);ed=ed[ed.Source.isin(idx)&ed.Target.isin(idx)].copy(); src=ed.Source.map(idx).to_numpy(int);dst=ed.Target.map(idx).to_numpy(int);w=ed.Weight.to_numpy(float);m=src!=dst;src,dst,w=src[m],dst[m],w[m]
wfac=np.log1p(w)/np.log1p(w.max()); infl=nodes.structural_influence_score.fillna(.5).to_numpy(float); orient=nodes.target0_ratio.fillna(.5).to_numpy(float); q=wfac*(.5+.5*infl[src])*(.5+.5*orient[src])
def mk(s,d,q):
 a=[[] for _ in range(N)]
 for x,y,z in zip(s,d,q): a[x].append((y,float(z)))
 return a
fwd=mk(src,dst,q)
# reversed direction recalculates source multiplier on new source but preserves weight; use same structural score/orientation of reversed source
qrev=wfac*(.5+.5*infl[dst])*(.5+.5*orient[dst]);rev=mk(dst,src,qrev)
def sim(adj,seed,beta=1.5,gamma=.25):
 r=np.random.default_rng(seed);sc=round(.01*N);st=np.zeros(N,np.int8);seeds=r.choice(N,sc,replace=False);st[seeds]=1;act=list(map(int,seeds));ever=np.zeros(N,bool);ever[seeds]=1
 for _ in range(100):
  if not act:break
  new=set()
  for u in act:
   for v,qq in adj[u]:
    if st[v]==0 and r.random()<min(1,beta*qq):new.add(v)
  nxt=[]
  for u in act:
   if r.random()<gamma:st[u]=2
   else:nxt.append(u)
  for v in new:
   if st[v]==0:st[v]=1;ever[v]=1;nxt.append(v)
  act=nxt
 return ever.sum()
rows=[]
for run in range(40):
 seed=1200000+run
 rows += [{'direction':'Source-to-Target assumption','run':run,'final_size':sim(fwd,seed)},{'direction':'Reversed-edge control','run':run,'final_size':sim(rev,seed)}]
r=pd.DataFrame(rows);r.to_csv(TAB/'direction_reversal_control_runs.csv',index=False)
a=r.pivot(index='run',columns='direction',values='final_size');d=a['Reversed-edge control']-a['Source-to-Target assumption'];br=np.random.default_rng(0);boot=[br.choice(d,len(d),replace=True).mean() for _ in range(5000)];print(r.groupby('direction').final_size.agg(['mean','std']));print('delta reverse-forward',d.mean(),np.percentile(boot,[2.5,97.5]))
pd.DataFrame([{'forward_mean':a['Source-to-Target assumption'].mean(),'reverse_mean':a['Reversed-edge control'].mean(),'delta_reverse_minus_forward':d.mean(),'ci95_lo':np.percentile(boot,2.5),'ci95_hi':np.percentile(boot,97.5)}]).to_csv(TAB/'direction_reversal_control_summary.csv',index=False)
