from pathlib import Path
import json, math, hashlib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'raw'
OUT=ROOT
FIG=OUT/'figures_revised'; TAB=OUT/'results_tables_revised'
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)

# consistent Okabe-Ito-ish colors
BLUE='#0072B2'; VERM='#D55E00'; GREEN='#009E73'; ORANGE='#E69F00'; SKY='#56B4E9'; PURPLE='#CC79A7'; GREY='#7F7F7F'; BLACK='#000000'
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8})

tweets=pd.read_csv(DATA/'master_tweets.csv')
edges=pd.read_csv(DATA/'network_edges.csv')
tweets['User']=tweets['User'].astype(str); edges['Source']=edges['Source'].astype(str); edges['Target']=edges['Target'].astype(str)
tweets['Date']=pd.to_datetime(tweets['Date'])
for c in ['Weight']:
    edges[c]=pd.to_numeric(edges[c],errors='coerce')

# checksums
checks={}
for fn in ['master_tweets.csv','network_edges.csv']:
    h=hashlib.sha256((DATA/fn).read_bytes()).hexdigest(); checks[fn]=h

# full graph
G=nx.DiGraph()
for r in edges.itertuples(index=False):
    if pd.notna(r.Weight): G.add_edge(r.Source,r.Target,weight=float(r.Weight))

tweet_users=set(tweets.User.unique()); net_users=set(G.nodes()); overlap=tweet_users & net_users
ov_edges=edges[edges.Source.isin(overlap)&edges.Target.isin(overlap)].copy()
G_ov=nx.DiGraph()
G_ov.add_nodes_from(overlap)
for r in ov_edges.itertuples(index=False): G_ov.add_edge(r.Source,r.Target,weight=float(r.Weight))
components=sorted(nx.weakly_connected_components(G_ov), key=len, reverse=True)
wcc_nodes=set(components[0]); H=G_ov.subgraph(wcc_nodes).copy()
final_tweets=tweets[tweets.User.isin(wcc_nodes)].copy()
outside=overlap-wcc_nodes

# component sizes outside WCC
comp_sizes=np.array([len(c) for c in components[1:]],dtype=int)
comp_df=pd.DataFrame({'component_rank':np.arange(2,len(components)+1),'size':comp_sizes})
comp_df.to_csv(TAB/'component_sizes_outside_wcc.csv',index=False)

# shares
summary={
 'tweet_records':len(tweets), 'tweet_users':len(tweet_users),'network_nodes':G.number_of_nodes(),'network_edges':G.number_of_edges(),
 'overlap_users':len(overlap),'overlap_edges':G_ov.number_of_edges(),'final_users':H.number_of_nodes(),'final_edges':H.number_of_edges(),
 'final_tweets':len(final_tweets),'self_loops':nx.number_of_selfloops(H),'n_wcc_components_overlap':len(components),
 'outside_users':len(outside),'outside_edges':G_ov.number_of_edges()-H.number_of_edges(),
 'isolated_overlap':sum(1 for n in G_ov if G_ov.degree(n)==0),
 'wcc_share_overlap_users':len(wcc_nodes)/len(overlap),
 'wcc_share_tweet_users':len(wcc_nodes)/len(tweet_users),
 'wcc_share_final_tweets':len(final_tweets)/len(tweets),
 'wcc_share_overlap_edges':H.number_of_edges()/G_ov.number_of_edges(),
 'full_density':nx.density(G), 'final_density':nx.density(H), 'reciprocity':nx.reciprocity(H),
 'n_scc':nx.number_strongly_connected_components(H),
 'largest_scc':max(map(len,nx.strongly_connected_components(H))),
 'date_min':str(tweets.Date.min().date()), 'date_max':str(tweets.Date.max().date()),
 'weight_min':float(edges.Weight.min()),'weight_median':float(edges.Weight.median()),'weight_mean':float(edges.Weight.mean()),'weight_max':float(edges.Weight.max()),
 'checksums':checks,
}
(TAB/'audit_summary.json').write_text(json.dumps(summary,indent=2))
pd.DataFrame([summary|{'checksums':None}]).to_csv(TAB/'audit_summary.csv',index=False)
print(json.dumps(summary,indent=2))

# Fig 1: data funnel + component-size CCDF + weekly activity
fig,axs=plt.subplots(1,3,figsize=(10.8,3.25))
ax=axs[0]
stages=['Tweet users','Network users','Matched users','Largest WCC']
vals=[len(tweet_users),G.number_of_nodes(),len(overlap),H.number_of_nodes()]
ax.bar(np.arange(4),vals,color=[SKY,GREY,ORANGE,BLUE])
ax.set_xticks(np.arange(4)); ax.set_xticklabels(stages,rotation=25,ha='right'); ax.set_ylabel('Users'); ax.set_title('(a) Population funnel')
for i,v in enumerate(vals): ax.text(i,v+max(vals)*.02,f'{v:,}',ha='center',fontsize=8)
ax.text(2, vals[2]*.52, f'{G_ov.number_of_edges():,} induced edges', ha='center', fontsize=7)
ax.text(3, vals[3]*.52, f'{H.number_of_edges():,} retained edges', ha='center', fontsize=7,color='white')

ax=axs[1]
if len(comp_sizes):
    xs=np.sort(comp_sizes[comp_sizes>0]); ys=1-np.arange(len(xs))/len(xs)
    ax.loglog(xs,ys,marker='.',ls='none',ms=3,color=GREY)
ax.set_xlabel('Component size (users)'); ax.set_ylabel('P(Size ≥ x)'); ax.set_title('(b) Components outside largest WCC')
ax.text(.04,.08,f'Outside users: {len(outside):,}\nOutside induced edges: {summary["outside_edges"]:,}\nComponents outside: {len(components)-1:,}',transform=ax.transAxes,fontsize=7,bbox=dict(boxstyle='round',fc='white',ec='0.8'))

# weekly activity and target0 share (Wilson)
weekly=tweets.set_index('Date').groupby(pd.Grouper(freq='W-MON')).agg(tweets=('Tweet','size'),active_users=('User','nunique'),neg=('Target',lambda x:(x==0).sum()),n=('Target','size')).reset_index()
weekly['neg_share']=weekly.neg/weekly.n
z=1.959963984540054
p=weekly.neg_share; n=weekly.n
center=(p+z*z/(2*n))/(1+z*z/n); half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
weekly['neg_lo']=center-half; weekly['neg_hi']=center+half
weekly.to_csv(TAB/'weekly_activity_label_balance.csv',index=False)
ax=axs[2]
ax2=ax.twinx(); ax.bar(weekly.Date,weekly.tweets,width=5,color=SKY,alpha=.5,label='Tweets/week')
ax2.plot(weekly.Date,weekly.neg_share,color=VERM,lw=1.6,label='Target 0 share'); ax2.fill_between(weekly.Date,weekly.neg_lo,weekly.neg_hi,color=VERM,alpha=.15)
ax.set_ylabel('Tweets/week'); ax2.set_ylabel('Target 0 share'); ax2.set_ylim(0,1); ax.set_title('(c) Weekly activity and label balance')
ax.tick_params(axis='x',rotation=30)
fig.tight_layout(); fig.savefig(FIG/'fig1_study_design_data_funnel.pdf',bbox_inches='tight'); fig.savefig(FIG/'fig1_study_design_data_funnel.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Degrees, strengths, weight distribution
nodes=list(H.nodes())
in_deg=np.array([H.in_degree(n) for n in nodes]); out_deg=np.array([H.out_degree(n) for n in nodes])
in_str=np.array([H.in_degree(n,weight='weight') for n in nodes],float); out_str=np.array([H.out_degree(n,weight='weight') for n in nodes],float)
node_stats=pd.DataFrame({'node':nodes,'in_degree':in_deg,'out_degree':out_deg,'in_strength':in_str,'out_strength':out_str})
node_stats.to_csv(TAB/'network_node_structural_stats.csv',index=False)

def ccdf(x):
    x=np.asarray(x,float); x=x[x>0]; x=np.sort(x); y=1-np.arange(len(x))/len(x); return x,y
fig,axs=plt.subplots(1,2,figsize=(7.3,3.0))
for arr,label,c,ls in [(in_deg,'In-degree',BLUE,'-'),(out_deg,'Out-degree',VERM,'--'),(in_str,'In-strength',GREEN,'-'),(out_str,'Out-strength',ORANGE,'--')]:
    x,y=ccdf(arr); axs[0].loglog(x,y,label=label,color=c,ls=ls,lw=1.4)
axs[0].set_xlabel('Degree or strength'); axs[0].set_ylabel('CCDF  P(X ≥ x)'); axs[0].set_title('(a) Heavy-tailed connectivity'); axs[0].legend(frameon=False)
w=edges.loc[edges.Source.isin(wcc_nodes)&edges.Target.isin(wcc_nodes),'Weight'].values
bins=np.logspace(np.log10(max(w.min(),1e-6)),np.log10(w.max()),35)
axs[1].hist(w,bins=bins,color=GREY,alpha=.8); axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_xlabel('Recorded edge Weight'); axs[1].set_ylabel('Edge count'); axs[1].set_title('(b) Edge-weight distribution')
fig.tight_layout(); fig.savefig(FIG/'fig2_network_structure_ccdf.pdf',bbox_inches='tight'); fig.savefig(FIG/'fig2_network_structure_ccdf.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Bow-tie around largest SCC in H
sccs=sorted(nx.strongly_connected_components(H),key=len,reverse=True); core=set(sccs[0])
# nodes that can reach core: ancestors of any representative in condensation? Multi-source reverse BFS from core
rev=H.reverse(copy=False)
def multi_reach(graph,seeds):
    seen=set(seeds); stack=list(seeds)
    while stack:
        u=stack.pop()
        for v in graph.successors(u):
            if v not in seen: seen.add(v); stack.append(v)
    return seen
OUT=multi_reach(H,core)-core
IN=multi_reach(rev,core)-core
TEND=set(H.nodes())-core-IN-OUT
bow={'SCC':len(core),'IN':len(IN),'OUT':len(OUT),'Tendrils/other':len(TEND)}
pd.DataFrame(list(bow.items()),columns=['region','nodes']).to_csv(TAB/'bow_tie_counts.csv',index=False)
fig,ax=plt.subplots(figsize=(5.2,2.7))
labels=list(bow); vals=list(bow.values()); cols=[PURPLE,BLUE,ORANGE,GREY]
ax.bar(labels,vals,color=cols); ax.set_ylabel('Users'); ax.set_title('Directed bow-tie decomposition of the 17,598-user WCC')
for i,v in enumerate(vals): ax.text(i,v+max(vals)*.02,f'{v:,}',ha='center',fontsize=8)
fig.tight_layout(); fig.savefig(FIG/'fig3_bow_tie.pdf',bbox_inches='tight'); fig.savefig(FIG/'fig3_bow_tie.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# centrality: PageRank / reverse PageRank / rank-normalized out strength / total degree
pr=nx.pagerank(H,alpha=.85,weight='weight',tol=1e-8,max_iter=200)
rpr=nx.pagerank(H.reverse(copy=False),alpha=.85,weight='weight',tol=1e-8,max_iter=200)
cent=node_stats.set_index('node').copy(); cent['pagerank']=pd.Series(pr); cent['reverse_pagerank']=pd.Series(rpr); cent['total_degree']=cent.in_degree+cent.out_degree
for col in ['out_strength','pagerank','reverse_pagerank','total_degree']:
    cent[col+'_rank']=cent[col].rank(method='average',pct=True)
cent['structural_influence_score']=0.6*cent['out_strength_rank']+0.4*cent['reverse_pagerank_rank']
cent.to_csv(TAB/'centrality_and_structural_influence.csv')
cols=['out_degree','out_strength','pagerank','reverse_pagerank','total_degree','structural_influence_score']
rho=cent[cols].corr(method='spearman'); rho.to_csv(TAB/'centrality_spearman.csv')
fig,axs=plt.subplots(1,2,figsize=(7.4,3.2))
im=axs[0].imshow(rho.values,vmin=-1,vmax=1,cmap='coolwarm'); axs[0].set_xticks(range(len(cols))); axs[0].set_yticks(range(len(cols))); short=['out-deg','out-str','PR','rev-PR','degree','influence']; axs[0].set_xticklabels(short,rotation=45,ha='right'); axs[0].set_yticklabels(short); axs[0].set_title('(a) Spearman rank correlations')
for i in range(len(cols)):
    for j in range(len(cols)): axs[0].text(j,i,f'{rho.iloc[i,j]:.2f}',ha='center',va='center',fontsize=6)
fig.colorbar(im,ax=axs[0],fraction=.046,pad=.04)
axs[1].loglog(cent.pagerank.values,cent.reverse_pagerank.values,'.',ms=2,alpha=.35,color=BLUE); rr=spearmanr(cent.pagerank,cent.reverse_pagerank).statistic
axs[1].set_xlabel('Standard PageRank'); axs[1].set_ylabel('Reverse PageRank'); axs[1].set_title(f'(b) Direction matters (Spearman ρ={rr:.2f})')
fig.tight_layout(); fig.savefig(FIG/'fig4_centrality_direction.pdf',bbox_inches='tight'); fig.savefig(FIG/'fig4_centrality_direction.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# label ratio per user and edge mixing
agg=final_tweets.groupby('User')['Target'].agg(['count',lambda s:(s==0).mean(),lambda s:(s==1).mean()]); agg.columns=['tweet_count','target0_ratio','target1_ratio']
# three operational bins: majority target0, mixed, majority target1
agg['orientation']=np.where(agg.target0_ratio>=.6,'Target-0 majority',np.where(agg.target1_ratio>=.6,'Target-1 majority','Mixed'))
cent=cent.join(agg,how='left'); cent.to_csv(TAB/'node_features_revision.csv')
# edges with both source/target labels
edf=pd.DataFrame(H.edges(),columns=['source','target']); edf['source_orientation']=edf.source.map(agg.orientation); edf['target_orientation']=edf.target.map(agg.orientation); edf=edf.dropna()
order=['Target-0 majority','Mixed','Target-1 majority']; mix=pd.crosstab(edf.source_orientation,edf.target_orientation).reindex(index=order,columns=order,fill_value=0); mix_norm=mix.div(mix.sum(axis=1),axis=0)
mix.to_csv(TAB/'orientation_mixing_counts.csv'); mix_norm.to_csv(TAB/'orientation_mixing_row_normalized.csv')
# assortativity numeric on continuous target0 ratio across edges: Pearson-like network assortativity; use Spearman across edges for robustness
x=np.array([agg.loc[u,'target0_ratio'] for u,v in H.edges() if u in agg.index and v in agg.index]); y=np.array([agg.loc[v,'target0_ratio'] for u,v in H.edges() if u in agg.index and v in agg.index])
obs_r=spearmanr(x,y).statistic
# label shuffle null 1000
rng=np.random.default_rng(42); vals=agg.target0_ratio.reindex(nodes).fillna(.5).values; node_idx={n:i for i,n in enumerate(nodes)}; edge_idx=np.array([(node_idx[u],node_idx[v]) for u,v in H.edges()],int)
null=[]
for _ in range(500):
    vv=rng.permutation(vals); null.append(spearmanr(vv[edge_idx[:,0]],vv[edge_idx[:,1]]).statistic)
null=np.array(null); p_emp=(1+np.sum(np.abs(null)>=abs(obs_r)))/(len(null)+1)
pd.DataFrame({'null_spearman':null}).to_csv(TAB/'orientation_assortativity_null.csv',index=False)
(TAB/'orientation_assortativity_summary.json').write_text(json.dumps({'observed_spearman':float(obs_r),'empirical_two_sided_p':float(p_emp),'null_mean':float(null.mean()),'null_sd':float(null.std(ddof=1))},indent=2))
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0))
im=axs[0].imshow(mix_norm.values,vmin=0,vmax=max(.01,mix_norm.values.max()),cmap='Blues'); axs[0].set_xticks(range(3)); axs[0].set_yticks(range(3)); axs[0].set_xticklabels(['T0 maj.','Mixed','T1 maj.'],rotation=30,ha='right'); axs[0].set_yticklabels(['T0 maj.','Mixed','T1 maj.']); axs[0].set_xlabel('Target user'); axs[0].set_ylabel('Source user'); axs[0].set_title('(a) Row-normalized orientation mixing')
for i in range(3):
  for j in range(3): axs[0].text(j,i,f'{mix_norm.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
fig.colorbar(im,ax=axs[0],fraction=.046,pad=.04)
axs[1].hist(null,bins=30,color=GREY,alpha=.8); axs[1].axvline(obs_r,color=VERM,lw=2,label=f'Observed ρ={obs_r:.3f}'); axs[1].set_xlabel('Spearman ρ after label shuffle'); axs[1].set_ylabel('Null realizations'); axs[1].set_title(f'(b) Label-shuffle null (p={p_emp:.3g})'); axs[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG/'fig5_orientation_mixing_null.pdf',bbox_inches='tight'); fig.savefig(FIG/'fig5_orientation_mixing_null.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Top100 PageRank WCC figure, title corrected; communities greedy modularity
_top=sorted(pr,key=pr.get,reverse=True)[:100]; SG=H.subgraph(_top).copy(); SG.remove_edges_from(nx.selfloop_edges(SG)); U=SG.to_undirected(); comms=list(nx.community.greedy_modularity_communities(U,weight='weight')); cmap={n:i for i,c in enumerate(comms) for n in c}
pos=nx.spring_layout(SG,seed=42,k=1.5,iterations=300,weight='weight')
prv=np.array([pr[n] for n in SG.nodes()]); s=60+900*(prv-prv.min())/(prv.max()-prv.min()+1e-15)
fig,ax=plt.subplots(figsize=(8.2,5.4)); nx.draw_networkx_edges(SG,pos,ax=ax,edge_color='0.75',alpha=.45,width=.6,arrows=True,arrowsize=7); nx.draw_networkx_nodes(SG,pos,ax=ax,node_size=s,node_color=[cmap[n] for n in SG.nodes()],cmap=plt.cm.tab20,linewidths=.3,edgecolors='white'); lab={n:n for n in _top[:20] if n in SG}; nx.draw_networkx_labels(SG,pos,labels=lab,ax=ax,font_size=6,font_weight='bold'); ax.set_title('Top-100 weighted-PageRank accounts in the analysis graph'); ax.axis('off'); fig.tight_layout(); fig.savefig(FIG/'fig6_top100_pagerank_wcc.pdf',bbox_inches='tight'); fig.savefig(FIG/'fig6_top100_pagerank_wcc.png',dpi=240,bbox_inches='tight'); plt.close(fig)

# no-skill audit from existing confusion
models=[('Static GCN',202,2669,1100,13627,0.8371,0.0968,0.4877,0.4978,0.7858),('EvolveGCN-style',390,2481,1978,12749,0.8364,0.1489,0.5000,0.5008,0.7466)]
rows=[]
neg_prev=2871/17598; pos_prev=14727/17598
for name,TN,FP,FN,TP,prauc,negf1,macf1,bal,acc in models:
    neg_pred=TN+FN; neg_precision=TN/neg_pred; neg_recall=TN/(TN+FP); random_f1=2*neg_prev*(neg_pred/17598)/(neg_prev+neg_pred/17598)
    rows.append({'model':name,'negative_flag_share':neg_pred/17598,'negative_precision':neg_precision,'negative_recall':neg_recall,'negative_f1':negf1,'random_f1_same_flag_rate':random_f1,'macro_f1':macf1,'balanced_accuracy':bal,'accuracy':acc,'positive_class_pr_auc_reported':prauc,'positive_prevalence_no_skill':pos_prev})
ns=pd.DataFrame(rows); ns.to_csv(TAB/'prediction_no_skill_audit.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0))
q=np.linspace(.001,.5,500); f=2*neg_prev*q/(neg_prev+q); axs[0].plot(q,f,color=GREY,label='Random flagger');
for _,r in ns.iterrows(): axs[0].scatter(r.negative_flag_share,r.negative_f1,s=45,label=r.model)
axs[0].set_xlabel('Fraction flagged Target 0'); axs[0].set_ylabel('Target-0 F1'); axs[0].set_title('(a) F1 against same-rate random flagging'); axs[0].legend(frameon=False,fontsize=7)
axs[1].bar(ns.model,ns.negative_precision,color=[BLUE,ORANGE]); axs[1].axhline(neg_prev,color=GREY,ls='--',label=f'Prevalence={neg_prev:.3f}'); axs[1].set_ylim(0,.22); axs[1].set_ylabel('Target-0 precision'); axs[1].set_title('(b) Precision is at prevalence'); axs[1].tick_params(axis='x',rotation=15); axs[1].legend(frameon=False,fontsize=7)
fig.tight_layout(); fig.savefig(FIG/'fig7_prediction_no_skill_audit.pdf',bbox_inches='tight'); fig.savefig(FIG/'fig7_prediction_no_skill_audit.png',dpi=220,bbox_inches='tight'); plt.close(fig)

print('analysis outputs written',FIG,TAB)
