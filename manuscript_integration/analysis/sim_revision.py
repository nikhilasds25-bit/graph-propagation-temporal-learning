from pathlib import Path
import json, math
import numpy as np, pandas as pd, networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT; TAB=OUT/'results_tables_revised'; FIG=OUT/'figures_revised'
BLUE='#0072B2'; VERM='#D55E00'; GREY='#7F7F7F'; GREEN='#009E73'; ORANGE='#E69F00'

nodes=pd.read_csv(TAB/'node_features_revision.csv',index_col=0)
# index is node name
nodes.index=nodes.index.astype(str)
# reconstruct final graph edges from raw and node index
edges=pd.read_csv(ROOT/'data'/'raw'/'network_edges.csv'); edges['Source']=edges.Source.astype(str); edges['Target']=edges.Target.astype(str)
idx={n:i for i,n in enumerate(nodes.index)}
edges=edges[edges.Source.isin(idx)&edges.Target.isin(idx)].copy()
# source/target arrays
src=edges.Source.map(idx).to_numpy(int); dst=edges.Target.map(idx).to_numpy(int); w=edges.Weight.to_numpy(float)
N=len(nodes); M=len(edges)
# log-normalized edge factor avoids max weight crushing 1-count edges too severely but stays [0,1]
wfac=np.log1p(w)/np.log1p(w.max())
# rank-normalized directed source influence: 0.6 out-strength + 0.4 reverse PR
influence=nodes['structural_influence_score'].fillna(.5).to_numpy(float)
# operational orientation ratios; inactive/missing default 0.5/0.5
rneg=nodes['target0_ratio'].fillna(.5).to_numpy(float); rpos=nodes['target1_ratio'].fillna(.5).to_numpy(float)
# source modifiers in [0.5,1]
Mstruct=.5+.5*influence
Mneg=.5+.5*rneg; Mpos=.5+.5*rpos
base_q=wfac*Mstruct[src]
qneg=np.clip(base_q*Mneg[src],0,1); qpos=np.clip(base_q*Mpos[src],0,1)

# sparse spectral radii of q matrices (transmission orientation source->target)
def rho_from_q(q):
    A=csr_matrix((q,(src,dst)),shape=(N,N))
    try: val=eigs(A,k=1,which='LM',return_eigenvectors=False,maxiter=5000,tol=1e-5)[0]
    except Exception: val=eigs(A.astype(float),k=1,which='LR',return_eigenvectors=False,maxiter=10000,tol=1e-4)[0]
    return float(abs(val))
rho_base=rho_from_q(base_q); rho_neg=rho_from_q(qneg); rho_pos=rho_from_q(qpos)
print('rhos',rho_base,rho_neg,rho_pos)
# simple linearized discrete threshold beta_c ~ gamma/rho(q) for infection attempts per step prior to recovery; cap expression discussed as approximation
gamma=.25
beta_c_neg=gamma/rho_neg; beta_c_pos=gamma/rho_pos; beta_c=gamma/rho_base
print('beta_c approx',beta_c,beta_c_neg,beta_c_pos)

# adjacency edge lists per source for efficient simulation
adj=[[] for _ in range(N)]
for s,d,qn,qp in zip(src,dst,qneg,qpos):
    if s==d: continue
    adj[s].append((d,qn,qp))

rng_global=np.random.default_rng(20261004)

def simulate_single(beta, qtype='neg', seed_count=25, max_steps=100, seed=None):
    rng=np.random.default_rng(seed)
    state=np.zeros(N,np.int8) # 0 S 1 I 2 R
    seeds=rng.choice(N,size=min(seed_count,N),replace=False); state[seeds]=1
    infected=list(map(int,seeds)); ever=np.zeros(N,bool); ever[seeds]=True
    traj=[len(infected)]
    secondary=0
    for t in range(max_steps):
        if not infected: break
        new=set()
        for u in infected:
            for d,qn,qp in adj[u]:
                if state[d]!=0: continue
                q=qn if qtype=='neg' else qp
                if rng.random()<min(1,beta*q): new.add(d)
        # synchronous: transmissions by currently infected, then recoveries, then activate new
        survivors=[]
        for u in infected:
            if rng.random()>=gamma: survivors.append(u)
            else: state[u]=2
        for d in new:
            if state[d]==0:
                state[d]=1; ever[d]=True; secondary+=1; survivors.append(d)
        infected=survivors; traj.append(len(infected))
    return {'final_size':int(ever.sum()),'secondary':secondary,'duration':len(traj)-1,'peak':max(traj),'traj':traj}

# Phase transition with 1% seeds, 40 runs each
ratios=np.array([0.2,0.35,0.5,0.7,0.85,1.0,1.2,1.5,2.0,3.0,5.0])
rows=[]
seed_count=max(1,round(.01*N))
for typ,bc in [('neg',beta_c_neg),('pos',beta_c_pos)]:
  for r in ratios:
    b=float(r*bc)
    # probabilities beta might exceed 1; this beta is a rate multiplier, p clipped in simulator
    for run in range(40):
      res=simulate_single(b,typ,seed_count=seed_count,max_steps=120,seed=100000+(0 if typ=='neg' else 50000)+int(r*1000)*50+run)
      rows.append({'type':typ,'beta_ratio':r,'beta':b,'run':run,**{k:v for k,v in res.items() if k!='traj'}})
phase=pd.DataFrame(rows); phase.to_csv(TAB/'phase_transition_runs.csv',index=False)
agg=phase.groupby(['type','beta_ratio']).final_size.agg(['mean','std','count']).reset_index(); agg['frac']=agg['mean']/N; agg['ci95']=1.96*agg['std']/np.sqrt(agg['count']); agg['ci95_frac']=agg.ci95/N; agg.to_csv(TAB/'phase_transition_summary.csv',index=False)
fig,ax=plt.subplots(figsize=(5.5,3.4))
for typ,c,label in [('neg',VERM,'Target-0-oriented contagion'),('pos',BLUE,'Target-1-oriented contagion')]:
    a=agg[agg.type==typ]; ax.plot(a.beta_ratio,a.frac,marker='o',ms=3,color=c,label=label); ax.fill_between(a.beta_ratio,a.frac-a.ci95_frac,a.frac+a.ci95_frac,color=c,alpha=.15)
ax.axvline(1,color='k',ls='--',lw=1,label='Approx. spectral threshold'); ax.set_xscale('log'); ax.set_xlabel(r'$\beta/\beta_c$'); ax.set_ylabel('Final reached fraction'); ax.set_title('Reach increases but remains constrained by directed topology'); ax.legend(frameon=False,fontsize=7)
fig.tight_layout(); fig.savefig(FIG/'fig8_phase_transition.pdf',bbox_inches='tight'); fig.savefig(FIG/'fig8_phase_transition.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Competition simulator with separate beta and seed shares, using total seed budget 2% N, grid 9x9, 20 runs
# states 0 S, 1 pos, 2 neg, 3 stopped; no reinfection. Current active states transmit then recover. competing successes tie by exposure probabilities.
def simulate_comp(beta_pos,beta_neg,neg_seed_share=.5,total_seed_count=None,max_steps=120,seed=None,track=False):
    rng=np.random.default_rng(seed); total_seed_count=total_seed_count or max(2,round(.02*N)); nneg=max(1,round(total_seed_count*neg_seed_share)); npos=max(1,total_seed_count-nneg)
    chosen=rng.choice(N,size=min(nneg+npos,N),replace=False); neg_seeds=chosen[:nneg]; pos_seeds=chosen[nneg:nneg+npos]
    state=np.zeros(N,np.int8); state[pos_seeds]=1; state[neg_seeds]=2
    active_pos=list(map(int,pos_seeds)); active_neg=list(map(int,neg_seeds)); ever_pos=np.zeros(N,bool); ever_neg=np.zeros(N,bool); ever_pos[pos_seeds]=True; ever_neg[neg_seeds]=True
    hist=[]
    for t in range(max_steps):
        if track: hist.append((int((state==0).sum()),len(active_pos),len(active_neg),int((state==3).sum())))
        if not active_pos and not active_neg: break
        # collect failure products for each target for each contagion
        fp={}; fn={}
        for u in active_pos:
            for d,qn,qp in adj[u]:
                if state[d]!=0: continue
                p=min(1,beta_pos*qp); fp[d]=fp.get(d,1.0)*(1-p)
        for u in active_neg:
            for d,qn,qp in adj[u]:
                if state[d]!=0: continue
                p=min(1,beta_neg*qn); fn[d]=fn.get(d,1.0)*(1-p)
        cand=set(fp)|set(fn); newp=[]; newn=[]
        for d in cand:
            ep=1-fp.get(d,1.0); en=1-fn.get(d,1.0)
            hp=rng.random()<ep if ep>0 else False; hn=rng.random()<en if en>0 else False
            if hp and not hn: newp.append(d)
            elif hn and not hp: newn.append(d)
            elif hp and hn:
                if rng.random()<en/(en+ep): newn.append(d)
                else: newp.append(d)
        nextp=[]; nextn=[]
        for u in active_pos:
            if rng.random()<gamma: state[u]=3
            else: nextp.append(u)
        for u in active_neg:
            if rng.random()<gamma: state[u]=3
            else: nextn.append(u)
        # ties between new lists cannot happen due above
        for d in newp:
            if state[d]==0: state[d]=1; ever_pos[d]=True; nextp.append(d)
        for d in newn:
            if state[d]==0: state[d]=2; ever_neg[d]=True; nextn.append(d)
        active_pos,active_neg=nextp,nextn
    if track: hist.append((int((state==0).sum()),len(active_pos),len(active_neg),int((state==3).sum())))
    cp=int(ever_pos.sum()); cn=int(ever_neg.sum()); stat=(cn-cp)/(cn+cp) if cn+cp else 0
    return {'pos_final':cp,'neg_final':cn,'outcome':stat,'duration':t+1,'hist':hist}

# choose supercritical reference 2* respective beta_c; outcome grid around relative ratios with absolute avg threshold multiplier 2
base=2*((beta_c_neg+beta_c_pos)/2)
lograt=np.linspace(-1,1,9); shares=np.linspace(.1,.9,9); orows=[]
for yi,share in enumerate(shares):
  for xi,lr in enumerate(lograt):
    ratio=10**lr; # maintain geometric mean base
    bp=base/np.sqrt(ratio); bn=base*np.sqrt(ratio)
    for run in range(20):
      z=simulate_comp(bp,bn,share,total_seed_count=max(2,round(.01*N)),seed=300000+yi*10000+xi*500+run)
      orows.append({'log10_beta_neg_over_pos':lr,'neg_seed_share':share,'beta_neg':bn,'beta_pos':bp,'run':run,**{k:v for k,v in z.items() if k!='hist'}})
out=pd.DataFrame(orows); out.to_csv(TAB/'competition_outcome_runs.csv',index=False); om=out.groupby(['neg_seed_share','log10_beta_neg_over_pos']).outcome.mean().unstack(); om.to_csv(TAB/'competition_outcome_map.csv')
fig,ax=plt.subplots(figsize=(5.4,3.8)); im=ax.imshow(om.values,origin='lower',aspect='auto',extent=[lograt.min(),lograt.max(),shares.min(),shares.max()],cmap='RdBu_r',vmin=-1,vmax=1); cs=ax.contour(lograt,shares,om.values,levels=[0],colors='k',linewidths=1); ax.clabel(cs,fmt={0:'parity'},fontsize=7); ax.set_xlabel(r'$\log_{10}(\beta_-/\beta_+)$'); ax.set_ylabel('Target-0 seed share'); ax.set_title('Competition outcome depends on transmission and seeding asymmetry'); cb=fig.colorbar(im,ax=ax); cb.set_label(r'$(C_- - C_+)/(C_-+C_+)$'); fig.tight_layout(); fig.savefig(FIG/'fig9_competition_outcome_map.pdf',bbox_inches='tight'); fig.savefig(FIG/'fig9_competition_outcome_map.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Representative time series: symmetric supercritical and asymmetric neg-favoured
series=[]
for label,bp,bn,share in [('Symmetric',base,base,.5),('Target-0 advantage',base/np.sqrt(2),base*np.sqrt(2),.5)]:
    # 30 runs; pad histories
    hs=[]
    for run in range(30): hs.append(simulate_comp(bp,bn,share,total_seed_count=max(2,round(.01*N)),seed=500000+(0 if label=='Symmetric' else 10000)+run,track=True)['hist'])
    L=max(map(len,hs)); arr=np.full((len(hs),L,4),np.nan)
    for i,h in enumerate(hs): arr[i,:len(h),:]=h
    mean=np.nanmean(arr,axis=0); lo=np.nanpercentile(arr,5,axis=0); hi=np.nanpercentile(arr,95,axis=0)
    for t in range(L): series.append({'scenario':label,'step':t,'S_mean':mean[t,0],'Ipos_mean':mean[t,1],'Ineg_mean':mean[t,2],'R_mean':mean[t,3],'Ipos_lo':lo[t,1],'Ipos_hi':hi[t,1],'Ineg_lo':lo[t,2],'Ineg_hi':hi[t,2]})
ser=pd.DataFrame(series); ser.to_csv(TAB/'competition_time_series.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(7.5,3.0),sharey=True)
for ax,(label,g) in zip(axs,ser.groupby('scenario',sort=False)):
    ax.plot(g.step,g.Ineg_mean,color=VERM,label='Target-0 active'); ax.fill_between(g.step,g.Ineg_lo,g.Ineg_hi,color=VERM,alpha=.15)
    ax.plot(g.step,g.Ipos_mean,color=BLUE,label='Target-1 active'); ax.fill_between(g.step,g.Ipos_lo,g.Ipos_hi,color=BLUE,alpha=.15)
    ax.set_title(label); ax.set_xlabel('Simulation step')
axs[0].set_ylabel('Active users'); axs[0].legend(frameon=False,fontsize=7); fig.suptitle('Competitive dynamics in non-degenerate regimes',fontsize=10); fig.tight_layout(); fig.savefig(FIG/'fig10_compartment_time_series.pdf',bbox_inches='tight'); fig.savefig(FIG/'fig10_compartment_time_series.png',dpi=220,bbox_inches='tight'); plt.close(fig)

summary={'gamma':gamma,'rho_base_q':rho_base,'rho_neg_q':rho_neg,'rho_pos_q':rho_pos,'beta_c_base_linearized':beta_c,'beta_c_neg_linearized':beta_c_neg,'beta_c_pos_linearized':beta_c_pos,'phase_runs_per_cell':40,'competition_runs_per_cell':20,'competition_total_seed_fraction':.01,'simulation_update':'synchronous transmissions from active nodes; recovery after exposure attempts; new infections activated for next step','seeds_count_in_final_size':True,'edge_factor':'log1p(weight)/max(log1p(weight))','structural_influence':'0.6 rank(out-strength)+0.4 rank(reverse PageRank)','orientation_modifier':'0.5+0.5 user Target-class ratio; missing=0.5'}
(TAB/'revised_simulation_settings.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
