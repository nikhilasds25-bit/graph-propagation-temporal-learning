"""Reviewer evidence pass. Exclusive new outputs; no training or manuscript edits.

Run phases separately: python -X utf8 src/reviewer_correction_analysis.py PHASE
PHASE: reconcile, layers, mixing, temporal, simulations, spectral, forecasting,
figures, package. All random analyses use seed 42. See configuration/assumptions.
"""
from pathlib import Path
import sys, json, hashlib, time, heapq, bisect, itertools, platform, shutil
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT/'results/tables'
FIG = ROOT/'results/figures/reviewer_corrected'
PKG = ROOT/'reproducibility_package'
SEED = 42
COLORS = ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#777777']
LAYERS = ['retweet','reply_inferred','mention']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
                     'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
START = time.perf_counter()

def norm(v):
    s=str(v).strip()
    return (s[1:] if s.startswith('@') else s).casefold()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def csv(name, data, directory=TABLE):
    p=directory/name; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='') as f:
        pd.DataFrame(data).to_csv(f,index=False,na_rep='NA')
    return p

def txt(name, text, directory=TABLE):
    p=directory/name; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: f.write(str(text)+'\n')
    return p

_READ_VERIFIED=set()
_BASELINE_INDEX=None
def read(p, **kw):
    global _BASELINE_INDEX
    import os
    path=ROOT/p
    baseline=ROOT/'reviewer_protected_hashes_20261009.json'
    canonical=os.path.normcase(str(path))
    if baseline.exists() and canonical not in _READ_VERIFIED:
        if _BASELINE_INDEX is None:
            records=json.loads(baseline.read_text(encoding='utf-8-sig'))
            _BASELINE_INDEX={os.path.normcase(str(Path(r['path']))):r for r in records}
        match=_BASELINE_INDEX.get(canonical)
        if match and sha(path).lower()!=match['sha256'].lower(): raise RuntimeError('Protected input changed before read: '+str(path))
        _READ_VERIFIED.add(canonical)
    return pd.read_csv(path,**kw)

def protect():
    import os
    target=ROOT/'reviewer_protected_hashes_20261009.json'
    if target.exists(): raise FileExistsError('Existing protected-input baseline; refusing overwrite')
    records=[]
    for directory,dirs,files in os.walk(ROOT):
        dirs[:]=[x for x in dirs if x not in ['.venv','.git','__pycache__']]
        for name in files:
            p=Path(directory)/name; records.append(dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)))
    with target.open('x',encoding='utf-8') as f: json.dump(records,f,indent=2)
    print('Protected files hashed:',len(records),flush=True)

def plot(name, fig, data):
    csv(name+'_source.csv',data,FIG)
    for ext in ['svg','png']:
        p=FIG/(name+'.'+ext)
        with p.open('xb') as f: fig.savefig(f,format=ext,dpi=400,facecolor='white',bbox_inches='tight')
    plt.close(fig)

def graph(events, direction='recorded', nodes=None):
    g=nx.DiGraph()
    if nodes is not None: g.add_nodes_from(sorted(nodes))
    s=events.Source.map(norm).to_numpy(); t=events.Target.map(norm).to_numpy()
    rev=np.zeros(len(events),bool)
    if direction=='reversed': rev[:]=True
    if direction=='layer_aware': rev=events.Interaction_Type.eq('retweet').to_numpy()
    for a,b in zip(np.where(rev,t,s),np.where(rev,s,t)):
        if g.has_edge(a,b): g[a][b]['weight']+=1
        else: g.add_edge(a,b,weight=1)
    return g

def cc(g):
    w=max(nx.weakly_connected_components(g),key=lambda x:(len(x),sorted(x)),default=set())
    s=max(nx.strongly_connected_components(g),key=lambda x:(len(x),sorted(x)),default=set())
    return w,s

def bowtie(g):
    _,core=cc(g)
    if not core: return dict(SCC=0,IN=0,OUT=0,other=0)
    a=next(iter(core)); inc=nx.ancestors(g,a)-core; out=nx.descendants(g,a)-core
    return dict(SCC=len(core),IN=len(inc),OUT=len(out),other=len(g)-len(core)-len(inc)-len(out))

def gini(x):
    a=np.sort(np.asarray(x,float)); n=len(a)
    return (2*np.dot(np.arange(1,n+1),a)/(n*a.sum())-(n+1)/n) if n and a.sum() else np.nan

def corr(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else np.nan

def spearman(x,y): return corr(stats.rankdata(x),stats.rankdata(y))

def summary(x,rng,boot=2000):
    a=np.asarray(x,float); a=a[np.isfinite(a)]
    if not len(a): return dict(n=0,mean=np.nan,sd=np.nan,median=np.nan,q25=np.nan,q75=np.nan,iqr=np.nan,ci95_low=np.nan,ci95_high=np.nan,median_ci95_low=np.nan,median_ci95_high=np.nan)
    q=np.quantile(a,[.25,.5,.75]); means=[]; meds=[]
    for _ in range(boot):
        z=rng.choice(a,len(a),replace=True); means.append(z.mean()); meds.append(np.median(z))
    return dict(n=len(a),mean=a.mean(),sd=a.std(ddof=1) if len(a)>1 else np.nan,
                median=q[1],q25=q[0],q75=q[2],iqr=q[2]-q[0],
                ci95_low=np.quantile(means,.025),ci95_high=np.quantile(means,.975),
                median_ci95_low=np.quantile(meds,.025),median_ci95_high=np.quantile(meds,.975))

def verify(write=False):
    records=json.loads((ROOT/'reviewer_protected_hashes_20261009.json').read_text(encoding='utf-8-sig'))
    failures=[]
    for r in records:
        p=Path(r['path'])
        if not p.is_file() or sha(p).lower()!=r['sha256'].lower(): failures.append(r['path'])
    result={'protected_file_count':len(records),'all_byte_identical':not failures,'changed_or_missing':failures}
    if write: txt('protected_input_posthash_verification.json',json.dumps(result,indent=2))
    print('NO EXISTING FILES OVERWRITTEN:',str(not failures).upper(),flush=True)
    if failures: raise RuntimeError(json.dumps(result))
    return result

def reconcile():
    tweets=read('data/raw/master_tweets.csv',keep_default_na=False)
    net=read('data/raw/network_edges.csv',dtype=str,keep_default_na=False)
    ev=read('data/processed/reconstructed_temporal_edges.csv',keep_default_na=False)
    rows=[]; populations={}
    for rule in ['exact','normalized']:
        f=(lambda x:str(x)) if rule=='exact' else norm
        u=tweets.User.map(f); s=net.Source.map(f); t=net.Target.map(f)
        tu=set(u); nu=set(s)|set(t); matched=tu&nu
        pairs={(a,b) for a,b in zip(s,t) if a in matched and b in matched}
        g=nx.DiGraph(); g.add_nodes_from(sorted(matched)); g.add_edges_from(sorted(pairs)); w,c=cc(g)
        populations[rule]=w
        rows.append(dict(rule=rule,unique_tweet_users=len(tu),unique_network_users=len(nu),matched_users=len(matched),
                         induced_directed_edges=len(pairs),weak_components=nx.number_weakly_connected_components(g),
                         isolates=len(list(nx.isolates(g))),largest_WCC_nodes=len(w),largest_WCC_edges=g.subgraph(w).number_of_edges(),
                         largest_SCC_nodes=len(c),self_loops=nx.number_of_selfloops(g),tweets_in_largest_WCC=int(u.isin(w).sum())))
    csv('matching_sensitivity_comparison.csv',rows)
    spell=defaultdict(Counter)
    for v in pd.concat([net.Source,net.Target,tweets.User]): spell[norm(v)][str(v).strip().removeprefix('@')]+=1
    csv('username_normalization_display_map.csv',[dict(normalized_key=k,display_spelling=sorted(v,key=lambda s:(-v[s],s))[0],spellings=len(v)) for k,v in sorted(spell.items())])
    same={norm(x) for x in populations['exact']}==populations['normalized']
    txt('matching_sensitivity_report.txt',f'{pd.DataFrame(rows).to_string(index=False)}\nMain WCC membership identical after normalizing old membership: {same}.\nExact matching preserves literal strings. Normalization trims whitespace, removes ONE leading @, casefolds. All matched users are added before edges, preserving isolates. Directed edges count unique pairs, not rows; duplicate weights do not affect connectivity. Largest-component ties break lexically. No original WCC files replaced. Display uses most frequent spelling with lexical tie break.')
    ep=set(zip(net.Source,net.Target)); npairs=set(zip(net.Source.map(norm),net.Target.map(norm)))
    rp=set(zip(ev.Source.map(norm),ev.Target.map(norm))); re=set(zip(ev.Source,ev.Target))
    out=[]
    for definition,pairs,recon in [('literal_exact',ep,re),('normalized',npairs,rp)]:
        overlap=len(pairs&recon); rowhits=sum((a,b) in recon for a,b in zip(net.Source.map(norm) if definition=='normalized' else net.Source,net.Target.map(norm) if definition=='normalized' else net.Target))
        out.append(dict(definition=definition,raw_supplied_rows=len(net),unique_supplied_pairs=len(pairs),reconstructed_unique_pairs=len(recon),overlap=overlap,
                        precision_numerator=overlap,precision_denominator=len(recon),precision_overlap=overlap/len(recon),
                        coverage_numerator=overlap,coverage_denominator=len(pairs),supplied_pair_coverage=overlap/len(pairs),
                        covered_raw_rows=rowhits,raw_row_denominator=len(net),raw_row_coverage=rowhits/len(net),
                        raw_row_definition='Every supplied row tested for pair membership; duplicates count repeatedly'))
    csv('reconstruction_overlap_reconciliation.csv',out)
    sys.path.insert(0,str(PKG/'vendor'))
    from textblob import TextBlob
    polarity=np.array([TextBlob(str(x)).sentiment.polarity for x in tweets.Tweet])
    csv('textblob_polarity_recomputed.csv',dict(tweet_row=np.arange(len(tweets)),Target=tweets.Target,polarity=polarity))
    y=pd.to_numeric(tweets.Target).to_numpy().astype(int); agree=((polarity>0).astype(int)==y)
    counts=[]
    for label in [0,1]:
        m=y==label; counts.append(dict(Target=label,negative=int(((polarity<0)&m).sum()),neutral=int(((polarity==0)&m).sum()),positive=int(((polarity>0)&m).sum()),
                                     agreement_numerator=int(agree[m].sum()),agreement_denominator=int(m.sum()),agreement=agree[m].mean(),agreement_percentage=100*agree[m].mean()))
    counts.append(dict(Target='overall',negative=int((polarity<0).sum()),neutral=int((polarity==0).sum()),positive=int((polarity>0).sum()),agreement_numerator=int(agree.sum()),agreement_denominator=len(y),agreement=agree.mean(),agreement_percentage=100*agree.mean()))
    csv('target_textblob_agreement_recheck.csv',counts)
    old=read('results/tables/target_sentiment_summary.csv')
    discrepancies=[]
    for r in counts[:2]:
        o=old[old.Target==r['Target']].iloc[0]
        for sign in ['negative','neutral','positive']:
            discrepancies.append(dict(Target=r['Target'],sign=sign,saved_count_derived=round(o[sign+'_polarity_percentage']*o.tweet_count/100),recomputed_count=r[sign]))
    csv('textblob_saved_summary_consistency.csv',discrepancies)
    ov=out[1]; implied=109217/(99.8638/100)
    txt('numerical_reconciliation_report.txt',f'Normalized pair coverage = {ov["overlap"]}/{ov["coverage_denominator"]} = {ov["supplied_pair_coverage"]*100:.8f}%.\n99.8638% implies denominator approximately {implied:.6f}; compare with exact unique supplied-pair count above. Raw rows are not a pair denominator.\nLexical agreement: {agree.sum()}/{len(y)} = {agree.mean()*100:.8f}%. Exact definition TextBlob(str(Tweet)).sentiment.polarity > 0; zero maps to Target 0. Target preserved. Saved aggregate counts compared independently.\nTextBlob version is recorded in software_versions; original notebook package version not saved, so historical-version equivalence cannot be guaranteed independently of count agreement. Lexical agreement is not validation of political affiliation, truth or human labels.')
    print(pd.DataFrame(rows).to_string(index=False),flush=True); print(pd.DataFrame(out).to_string(index=False),flush=True); print(pd.DataFrame(counts).to_string(index=False),flush=True)

def layers():
    ev=read('data/processed/reconstructed_temporal_edges.csv'); results=[]; node_rows=[]; lorenz=[]; graphs={}
    for layer in LAYERS+['pooled']:
        e=ev if layer=='pooled' else ev[ev.Interaction_Type==layer]
        for direction in ['recorded','layer_aware' if layer=='pooled' else 'reversed']:
            g=graph(e,direction); graphs[(layer,direction)]=g; w,c=cc(g); b=bowtie(g)
            ins=dict(g.in_degree(weight='weight')); outs=dict(g.out_degree(weight='weight'))
            pr=nx.pagerank(g,alpha=.85,weight='weight',tol=1e-10,max_iter=1000)
            rpr=nx.pagerank(g.reverse(copy=False),alpha=.85,weight='weight',tol=1e-10,max_iter=1000)
            n=len(g); top=max(1,int(np.ceil(.01*n)))
            results.append(dict(layer=layer,direction=direction,nodes=n,edges=g.number_of_edges(),events=len(e),self_loops=nx.number_of_selfloops(g),density=nx.density(g),
                                reciprocity=nx.reciprocity(g),largest_WCC=len(w),largest_SCC=len(c),SCC_fraction=len(c)/n,**b,
                                mean_in_degree=np.mean([d for _,d in g.in_degree()]),mean_out_degree=np.mean([d for _,d in g.out_degree()]),
                                mean_in_strength=np.mean(list(ins.values())),mean_out_strength=np.mean(list(outs.values())),
                                maximum_in_degree=max(dict(g.in_degree()).values()),maximum_out_degree=max(dict(g.out_degree()).values()),maximum_in_strength=max(ins.values()),maximum_out_strength=max(outs.values()),
                                gini_in_strength=gini(list(ins.values())),gini_out_strength=gini(list(outs.values())),top_1pct_in_strength_share=sum(sorted(ins.values(),reverse=True)[:top])/sum(ins.values())))
            for u in sorted(g): node_rows.append(dict(layer=layer,direction=direction,node=u,in_degree=g.in_degree(u),out_degree=g.out_degree(u),in_strength=ins[u],out_strength=outs[u],PageRank=pr[u],reverse_PageRank=rpr[u]))
            for side,values in [('in',ins),('out',outs)]:
                a=np.sort(list(values.values())); y=np.r_[0,np.cumsum(a)/sum(a)]; x=np.linspace(0,1,n+1)
                for k in np.unique(np.linspace(0,n,501).astype(int)): lorenz.append(dict(layer=layer,direction=direction,side=side,node_fraction=x[k],strength_fraction=y[k]))
            print('Layer complete:',layer,direction,n,g.number_of_edges(),flush=True)
    csv('typed_layer_network_statistics.csv',results); csv('typed_layer_node_centralities.csv',node_rows)
    overlaps=[]
    for direction in ['recorded','reversed']:
        for a,b in itertools.combinations(LAYERS,2):
            g,h=graphs[(a,direction)],graphs[(b,direction)]; ge=set(g.edges()); he=set(h.edges()); shared=sorted(set(g)&set(h))
            overlaps.append(dict(layer_a=a,layer_b=b,direction=direction,edge_intersection=len(ge&he),edge_union=len(ge|he),edge_jaccard=len(ge&he)/len(ge|he),
                                 node_intersection=len(shared),node_union=len(set(g)|set(h)),node_jaccard=len(shared)/len(set(g)|set(h)),correlation_population='intersection of layer nodes',
                                 in_degree_spearman=spearman([g.in_degree(u) for u in shared],[h.in_degree(u) for u in shared]),out_degree_spearman=spearman([g.out_degree(u) for u in shared],[h.out_degree(u) for u in shared]),
                                 in_strength_spearman=spearman([g.in_degree(u,weight='weight') for u in shared],[h.in_degree(u,weight='weight') for u in shared]),out_strength_spearman=spearman([g.out_degree(u,weight='weight') for u in shared],[h.out_degree(u,weight='weight') for u in shared])))
    csv('typed_layer_overlap.csv',overlaps)
    txt('typed_direction_assumptions.txt','Nodes are normalized endpoint unions per layer, without adding tweet-only isolates. Edges are distinct directed pairs; weights count retained reconstructed events, not supplied Weight. Self-loops retained; density uses networkx m/[n(n-1)], including loops in m; reciprocity uses NetworkX convention. PageRank alpha=.85 is event-weighted; reverse PageRank runs on transpose. Bow-tie uses largest SCC on all layer nodes; other includes disconnected components and tendrils. In/out strength Gini includes zero-strength nodes; top 1% uses ceiling. Recorded direction author -> referenced account. Retweet plausible content flow reverses recorded edge. Mention and inferred reply reversals are uncertainty controls, not verified information flow. Pooled layer-aware PRIMARY convention reverses retweets only, retaining attention/action direction for mention and inferred reply. An all-reversed pooled control is separately analyzed in temporal/simulation phases. Reconstruction prioritizes retweet > inferred reply > mention per target per tweet. Native retweet/reply metadata absent. Direction controls do not establish causal transmission. Full typed universe differs from the original exact-match diffusion WCC.')
    d=pd.DataFrame(results); rec=d[(d.direction=='recorded')&(d.layer!='pooled')]
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,col in zip(axs,['nodes','edges','events']): ax.bar(rec.layer,rec[col],color=COLORS[:3]); ax.set(ylabel=col.capitalize()); ax.tick_params(axis='x',labelrotation=15)
    plot('typed_layer_sizes',fig,rec)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained'); base=np.zeros(len(rec))
    for k,col in enumerate(['SCC','IN','OUT','other']): ax.bar(rec.layer,rec[col]/rec.nodes,bottom=base,label=col,color=COLORS[k]); base+=rec[col].to_numpy()/rec.nodes.to_numpy()
    ax.set(ylabel='Fraction of layer nodes'); ax.legend(ncol=4); plot('typed_bowtie',fig,rec)
    ld=pd.DataFrame(lorenz); fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,layer in zip(axs,LAYERS):
        for k,side in enumerate(['in','out']):
            z=ld[(ld.layer==layer)&(ld.direction=='recorded')&(ld.side==side)]; ax.plot(z.node_fraction,z.strength_fraction,label=side,color=COLORS[k])
        ax.plot([0,1],[0,1],color='#999999',ls=':'); ax.set(title=layer,xlabel='Cumulative node fraction',ylabel='Cumulative strength fraction'); ax.legend()
    plot('typed_strength_lorenz',fig,ld)
    matrix=np.eye(3)
    for r in overlaps:
        if r['direction']=='recorded': i,j=LAYERS.index(r['layer_a']),LAYERS.index(r['layer_b']); matrix[i,j]=matrix[j,i]=r['edge_jaccard']
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained'); im=ax.imshow(matrix,vmin=0,vmax=1,cmap='cividis'); ax.set_xticks(range(3),LAYERS); ax.set_yticks(range(3),LAYERS)
    for i,j in itertools.product(range(3),repeat=2): ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',color='white' if matrix[i,j]<.5 else 'black')
    fig.colorbar(im,ax=ax,label='Directed edge Jaccard'); plot('typed_layer_overlap_heatmap',fig,overlaps)
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,col in zip(axs,['IN','OUT','gini_in_strength']):
        for k,direction in enumerate(['recorded','reversed']):
            z=d[(d.direction==direction)&(d.layer!='pooled')]; ax.bar(np.arange(3)+(k-.5)*.35,z[col],width=.35,label=direction,color=COLORS[k])
        ax.set_xticks(range(3),LAYERS,rotation=15); ax.set(ylabel=col); ax.legend(fontsize=9)
    plot('typed_direction_structure',fig,d)

def mixing():
    ev=read('data/processed/reconstructed_temporal_edges.csv')
    f=read('data/processed/diffusion_ready_node_features_full.csv'); f['node']=f.node_id.map(norm)
    hist=f.groupby('node')[['negative_tweet_count','positive_tweet_count']].sum(); ratios=hist.positive_tweet_count/(hist.positive_tweet_count+hist.negative_tweet_count)
    rows=[]; tests=[]; draws=[]; strata_out=[]; rng=np.random.default_rng(SEED)
    for layer in LAYERS:
        g=graph(ev[ev.Interaction_Type==layer]); users=sorted(set(g)&set(ratios.index)); ids={u:i for i,u in enumerate(users)}
        pairs=[(a,b) for a,b in g.edges() if a in ids and b in ids]; a=np.array([ids[x] for x,_ in pairs]); b=np.array([ids[y] for _,y in pairs]); r=ratios.reindex(users).to_numpy()
        # Fixed bivariate degree/strength quartiles with duplicate cutpoints removed.
        degree=np.array([g.degree(u) for u in users]); strength=np.array([g.degree(u,weight='weight') for u in users])
        db=np.digitize(degree,np.unique(np.quantile(degree,[.25,.5,.75])),right=True); sb=np.digitize(strength,np.unique(np.quantile(strength,[.25,.5,.75])),right=True)
        bins=db*4+sb; groups=[np.flatnonzero(bins==v) for v in np.unique(bins)]
        for u,dg,st,bi in zip(users,degree,strength,bins): strata_out.append(dict(layer=layer,node=u,total_degree=dg,total_strength=st,stratum=bi,orientation_ratio=ratios[u]))
        cat=np.where(r<.5,0,np.where(r>.5,1,2)); names=['Target 0 dominant','Target 1 dominant','tie']
        mat=np.zeros((3,3),int); np.add.at(mat,(cat[a],cat[b]),1)
        sr=spearman(r[a],r[b]); pearson=corr(r[a],r[b]); nominal=nx.attribute_assortativity_coefficient(nx.set_node_attributes(g,{u:names[cat[i]] for i,u in enumerate(users)},'orientation') or g.subgraph(users),'orientation') if False else np.nan
        # Directed nominal assortativity from the included 3x3 edge mixing matrix.
        e=mat/mat.sum(); expected=np.dot(e.sum(1),e.sum(0)); nominal=(np.trace(e)-expected)/(1-expected) if expected<1 else np.nan
        for i,j in itertools.product(range(3),repeat=2): rows.append(dict(layer=layer,direction='recorded',source_category=names[i],target_category=names[j],edge_count=mat[i,j],row_normalized=mat[i,j]/mat[i].sum() if mat[i].sum() else np.nan,
                                                                       source_target_spearman=sr,numeric_assortativity_Pearson=pearson,nominal_assortativity=nominal,eligible_nodes=len(users),eligible_edges=len(pairs),excluded_missing_orientation_edges=g.number_of_edges()-len(pairs)))
        for null in ['degree_strength_stratified','global_shuffle']:
            vals=np.empty(2000)
            for p in range(2000):
                rr=r.copy()
                if null=='global_shuffle': rr=rng.permutation(rr)
                else:
                    for idx in groups: rr[idx]=rng.permutation(rr[idx])
                vals[p]=spearman(rr[a],rr[b])
            valid=vals[np.isfinite(vals)]; mean=valid.mean(); sd=valid.std(ddof=1); undefined=int((~np.isfinite(vals)).sum())
            # An undefined draw is not an extreme statistic. Do not manufacture a small p-value.
            pv=(1+np.sum(np.abs(valid-mean)>=abs(sr-mean)))/2001 if not undefined and np.isfinite(sr) else np.nan
            tests.append(dict(layer=layer,null=null,statistic='Spearman orientation ratio across unique directed edges',observed=sr,null_mean=mean,null_SD=sd,p_two_sided=pv,z_score=(sr-mean)/sd if sd and not undefined else np.nan,null_95_low=np.quantile(valid,.025),null_95_high=np.quantile(valid,.975),permutations=2000,undefined_null_draws=undefined,null_interval_definition='finite null draws only; inferential p/z NA if any null undefined',seed=SEED,strata=len(groups),singleton_strata=sum(len(x)==1 for x in groups)))
            draws.extend(dict(layer=layer,null=null,permutation=k+1,statistic=v) for k,v in enumerate(vals))
            print('Permutation complete:',layer,null,sr,pv,flush=True)
    t=pd.DataFrame(tests)
    for null in t.null.unique():
        idx=t.index[t.null==null].tolist(); ordered=sorted(idx,key=lambda i:t.loc[i,'p_two_sided'] if np.isfinite(t.loc[i,'p_two_sided']) else np.inf); last=0
        for k,i in enumerate(ordered):
            if not np.isfinite(t.loc[i,'p_two_sided']): t.loc[i,'Holm_p']=np.nan; continue
            last=max(last,min(1,t.loc[i,'p_two_sided']*(len(ordered)-k))); t.loc[i,'Holm_p']=last
    csv('layer_orientation_mixing.csv',rows); csv('layer_orientation_permutation_tests.csv',t); csv('layer_orientation_permutation_draws.csv',draws); csv('layer_orientation_degree_strength_strata.csv',strata_out)
    txt('layer_orientation_assumptions.txt','Orientation = saved full-period positive_tweet_count / (negative_tweet_count + positive_tweet_count) from diffusion_ready_node_features_full.csv, summed if normalized keys merge. This is descriptive all-history orientation, not a causal pre-interaction label. Missing endpoint labels are excluded, not imputed. Ratio <0.5 is Target-0 dominant; >0.5 Target-1; exactly .5 retained as a separate tie category. Statistics use one observation per distinct directed pair, with self-loops retained. Numeric assortativity is endpoint Pearson correlation of ratio; nominal assortativity uses three categories. Primary predeclared tests are three recorded-layer Spearman correlations; Holm within these three. Global-shuffle sensitivities form a separate family of three. Stratified null shuffles entire ratio values within total-degree quartile x total-event-strength quartile strata defined on eligible layer nodes (tied boundaries retained together, duplicate boundaries removed). This approximately controls total degree/strength, not exact directed degree sequence or temporal exposure. Empirical two-sided deviations are centered on the empirical null mean with +1 correction. 2000 permutations per layer per null, seed 42. Unequal/isolated strata limit exchangeability; association does not establish homophily or political identity.')
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    d=pd.DataFrame(rows)
    for ax,layer in zip(axs,LAYERS):
        z=d[d.layer==layer]; m=z.row_normalized.to_numpy().reshape(3,3); ax.imshow(m,vmin=0,vmax=1,cmap='cividis'); ax.set(title=layer); ax.set_xticks(range(3),['0 dominant','1 dominant','tie']); ax.set_yticks(range(3),['0 dominant','1 dominant','tie']); ax.set(xlabel='Referenced account',ylabel='Author')
        for i,j in itertools.product(range(3),repeat=2): ax.text(j,i,f'{m[i,j]:.2f}\nn={mat[i,j] if False else z.edge_count.to_numpy().reshape(3,3)[i,j]}',ha='center',va='center',fontsize=9,color='white' if m[i,j]<.5 else 'black')
    plot('layer_orientation_matrices',fig,d)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained'); z=t[t.null=='degree_strength_stratified']
    for k,(_,r) in enumerate(z.iterrows()): ax.plot([r.null_95_low,r.null_95_high],[k,k],color=COLORS[6],lw=6); ax.scatter(r.observed,k,color=COLORS[0],zorder=3); ax.scatter(r.null_mean,k,color=COLORS[1],marker='|',zorder=4)
    ax.set_yticks(range(3),z.layer); ax.set(xlabel='Spearman correlation (point: observed; bar: 95% stratified null)'); plot('layer_orientation_null_forest',fig,t)

def temporal_events(ev,direction):
    e=ev.copy(); e['time']=pd.to_datetime(e.Timestamp,utc=True,errors='coerce'); invalid=int(e.time.isna().sum()); e=e[e.time.notna()]
    s=e.Source.map(norm).to_numpy(); t=e.Target.map(norm).to_numpy(); rev=np.ones(len(e),bool) if direction=='reversed' else e.Interaction_Type.eq('retweet').to_numpy() if direction=='layer_aware' else np.zeros(len(e),bool)
    e['s']=np.where(rev,t,s); e['t']=np.where(rev,s,t); e['ts']=e.time.astype('int64')//10**9
    return e,invalid

def arrival(seed,adj,t0):
    reached={seed:(t0,0)}; queue=[(t0,0,seed)]
    while queue:
        tm,h,u=heapq.heappop(queue)
        if reached[u]!=(tm,h): continue
        times,targets=adj.get(u,([],[])); start=bisect.bisect_left(times,tm)
        for j in range(start,len(times)):
            v=targets[j]; candidate=(times[j],h+1)
            if v not in reached or candidate<reached[v]: reached[v]=candidate; heapq.heappush(queue,(*candidate,v))
    reached.pop(seed,None); return reached

def temporal():
    ev=read('data/processed/reconstructed_temporal_edges.csv'); rng=np.random.default_rng(SEED); rows=[]; samples=[]
    cases=[('pooled',x) for x in ['recorded','layer_aware','reversed']]+[(l,d) for l in LAYERS for d in ['recorded','reversed']]
    for layer,direction in cases:
        raw=ev if layer=='pooled' else ev[ev.Interaction_Type==layer]; e,invalid=temporal_events(raw,direction); g=graph(raw,direction)
        users=sorted(g); adj={}
        for u,z in e.sort_values(['ts','t'],kind='stable').groupby('s',sort=False): adj[u]=(z.ts.to_list(),z.t.to_list())
        pr=nx.pagerank(g,weight='weight',alpha=.85,tol=1e-10,max_iter=1000); rpr=nx.pagerank(g.reverse(copy=False),weight='weight',alpha=.85,tol=1e-10,max_iter=1000); n=min(500,len(users)); cache={}
        strata={'random':sorted(rng.choice(users,n,replace=False)), 'high_out_strength':sorted(users,key=lambda u:(-g.out_degree(u,weight='weight'),u))[:n],
                'high_reverse_PageRank':sorted(users,key=lambda u:(-rpr[u],u))[:n], 'high_PageRank':sorted(users,key=lambda u:(-pr[u],u))[:n]}
        t0=int(e.ts.min())
        for stratum,seeds in strata.items():
            for seed in seeds:
                if seed not in cache:
                    static=len(nx.descendants(g,seed)); reached=arrival(seed,adj,t0); tm=np.array([v[0]-t0 for v in reached.values()],float)/86400
                    cache[seed]=dict(static_out_component=static,temporal_reachable=len(reached),temporal_static_ratio=len(reached)/static if static else np.nan,
                                     fraction_lost_time_order=(static-len(reached))/static if static else np.nan,earliest_arrival_min_days=tm.min() if len(tm) else np.nan,
                                     earliest_arrival_median_days=np.median(tm) if len(tm) else np.nan,earliest_arrival_max_days=tm.max() if len(tm) else np.nan,
                                     max_earliest_arrival_witness_hops=max((v[1] for v in reached.values()),default=0))
                    assert len(reached)<=static
                    # Earliest-arrival examples: first 20 reached nodes per unique seed.
                    samples.extend(dict(layer=layer,direction=direction,seed=seed,target=v,arrival_UTC=pd.to_datetime(tm2,unit='s',utc=True).isoformat(),witness_hops=h) for v,(tm2,h) in sorted(reached.items(),key=lambda x:(x[1],x[0]))[:20])
                rows.append(dict(layer=layer,direction=direction,stratum=stratum,seed=seed,graph_nodes=len(g),graph_edges=g.number_of_edges(),valid_events=len(e),invalid_timestamp_events=invalid,start_UTC=pd.to_datetime(t0,unit='s',utc=True).isoformat(),**cache[seed]))
            print('Temporal complete:',layer,direction,stratum,flush=True)
    d=pd.DataFrame(rows); csv('time_respecting_reachability_by_seed.csv',d); csv('time_respecting_earliest_arrival_examples.csv',samples)
    out=[]
    for key,z in d.groupby(['layer','direction','stratum'],sort=False):
        for metric in ['static_out_component','temporal_reachable','temporal_static_ratio','fraction_lost_time_order','max_earliest_arrival_witness_hops']:
            out.append(dict(layer=key[0],direction=key[1],stratum=key[2],metric=metric,zero_static_seeds=int(z.static_out_component.eq(0).sum()),**summary(z[metric],rng)))
    csv('time_respecting_reachability_summary.csv',out)
    txt('time_respecting_reachability_report.txt','Static and temporal counts exclude the seed. Seeds are present from the first valid timestamp of each layer, not from their first tweet. Non-decreasing timestamps allow arbitrarily ordered same-time edges: heap-based earliest-arrival relaxation reaches the fixed point within equal timestamps, avoiding row-order artifacts. No waiting deadline; observation horizon is the last event. Max hops is the maximum shortest witness length among earliest-arrival paths, NOT the longest temporal walk (cycles can make that undefined). Invalid timestamp events are excluded temporally but included statically, with counts saved. 500 random seeds plus top 500 each of out-strength, reverse PageRank and standard PageRank per graph, stable lexical tie break; strata can overlap and centrality strata are deterministic targeted samples. Ratios for zero static descendants are NA. Bootstrap uses 2000 seed resamples within each stratum, reporting mean and median CI; this describes sampled seeds, not independent network realizations, and is descriptive for deterministic top-centrality samples. Layer-aware pooled direction reverses retweets only. Each layer also has recorded and reversed sensitivity. Earliest arrival examples are capped at 20 nodes per unique seed; per-seed arrival range/median saved. Full temporal endpoint universe is not original diffusion WCC. No causal claims or missing event imputation.\n'+pd.DataFrame(out).query("metric == 'temporal_static_ratio'").to_string(index=False))
    z=d[d.layer=='pooled']; fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,direction in zip(axs,['recorded','layer_aware','reversed']):
        q=z[z.direction==direction]; ax.scatter(q.static_out_component,q.temporal_reachable,s=8,alpha=.3,color=COLORS[0]); lim=max(q.static_out_component.max(),1); ax.plot([0,lim],[0,lim],ls=':',color=COLORS[6]); ax.set(title=direction,xlabel='Static descendants',ylabel='Temporal descendants'); ax.set_xscale('symlog',linthresh=1); ax.set_yscale('symlog',linthresh=1)
    plot('static_vs_temporal_reach',fig,z)
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,direction in zip(axs,['recorded','layer_aware','reversed']):
        for k,stratum in enumerate(z.stratum.unique()):
            a=z[(z.direction==direction)&(z.stratum==stratum)].temporal_static_ratio.dropna(); ax.hist(a,bins=np.linspace(0,1,21),histtype='step',label=stratum,color=COLORS[k])
        ax.set(title=direction,xlabel='Temporal / static reach',ylabel='Seed count')
    axs[-1].legend(fontsize=8); plot('temporal_static_ratio_distribution',fig,z)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained'); sums=pd.DataFrame(out); q=sums[(sums.layer=='pooled')&(sums.direction=='layer_aware')&(sums.metric=='temporal_reachable')]
    ax.errorbar(q['mean'],np.arange(len(q)),xerr=[q['mean']-q.ci95_low,q.ci95_high-q['mean']],fmt='o',color=COLORS[0]); ax.set_yticks(range(len(q)),q.stratum); ax.set(xlabel='Mean temporal descendants (seed-bootstrap 95% CI)'); plot('temporal_reach_by_seed_stratum',fig,q)

def simulation_graphs():
    nodes=read('data/processed/diffusion_model_nodes.csv'); nodes=nodes.assign(node=nodes.node_id.map(norm)).sort_values('node').reset_index(drop=True)
    assert nodes.node.is_unique
    edges=read('data/processed/diffusion_model_edges.csv'); edges.Source=edges.Source.map(norm); edges.Target=edges.Target.map(norm)
    events=read('data/processed/reconstructed_temporal_edges.csv'); events.Source=events.Source.map(norm); events.Target=events.Target.map(norm)
    counts=events.groupby(['Source','Target','Interaction_Type']).size().unstack(fill_value=0)
    graphs={x:nx.DiGraph() for x in ['recorded','reversed','layer_aware']}
    for g in graphs.values(): g.add_nodes_from(nodes.node)
    missing=0
    for s,t,w in edges[['Source','Target','Weight']].itertuples(index=False,name=None):
        frac=0.
        if (s,t) in counts.index: c=counts.loc[(s,t)]; frac=float(c.get('retweet',0)/c.sum())
        else: missing+=1
        for direction,parts in [('recorded',[(s,t,float(w))]),('reversed',[(t,s,float(w))]),('layer_aware',[(s,t,float(w)*(1-frac)),(t,s,float(w)*frac)])]:
            g=graphs[direction]
            for a,b,weight in parts:
                if weight<=0: continue
                if g.has_edge(a,b): g[a][b]['weight']+=weight
                else: g.add_edge(a,b,weight=weight)
    return nodes,graphs,float(edges.Weight.max()),missing

def keyed_uniform(ids,run,step,channel):
    # Common random numbers indexed by node and timestep; no direction-dependent draw consumption.
    a=np.asarray(ids,dtype=np.uint64)
    with np.errstate(over='ignore'):
        key=np.uint64(SEED) ^ (np.uint64(run+1)*np.uint64(0x9e3779b97f4a7c15)) ^ (np.uint64(step+1)*np.uint64(0xd2b74407b1ce6e93)) ^ (np.uint64(channel+1)*np.uint64(0xca5a826395121157))
        z=(a*np.uint64(0x8cb92baa3f3d8dd7)) ^ key
        z=(z^(z>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        z=(z^(z>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        z=z^(z>>np.uint64(31))
    return (z>>np.uint64(11)).astype(float)*(1.0/9007199254740992.0)

def simulator(adj,n,neg,pos,beta0,beta1,run,features,full):
    state=np.zeros(n,np.int8); state[neg]=2; state[pos]=1
    seeds=set(neg)|set(pos); ever0=set(neg); ever1=set(pos); exposed0=set(); exposed1=set(); resolutions=0; peaks=[0,0]
    active=np.array(sorted(seeds),int); p0,p1,trust,pagerank,polarity=features
    for step in range(25):
        if len(active)==0: break
        failures=[{},{}]
        for u in active:
            c=0 if state[u]==2 else 1; beta=beta0 if c==0 else beta1; orient=p0 if c==0 else p1
            for v,w in adj[u]:
                if state[v]!=0: continue
                p=beta*w
                if full:
                    src=.5*orient[u]+.5*max(0.,-polarity[u] if c==0 else polarity[u])
                    p*=(.5+.5*trust[u])*(.5+.5*pagerank[u])*(.5+.5*src)*(.5+.5*orient[v])
                p=min(max(p,0.),1.)
                if p>0: failures[c][v]=failures[c].get(v,1.)*(1.-p)
        exposed0.update(failures[0]); exposed1.update(failures[1]); targets=np.array(sorted(set(failures[0])|set(failures[1])),int)
        new=[]
        if len(targets):
            e0=np.array([1-failures[0].get(v,1.) for v in targets]); e1=np.array([1-failures[1].get(v,1.) for v in targets]); accepted=keyed_uniform(targets,run,step,0)<1-(1-e0)*(1-e1)
            winners=np.where(keyed_uniform(targets,run,step,1)<e0/np.maximum(e0+e1,1e-300),2,1)
            resolutions+=int((accepted&(e0>0)&(e1>0)).sum()); new=targets[accepted].tolist(); state[targets[accepted]]=winners[accepted]
            ever0.update(targets[accepted&(winners==2)].tolist()); ever1.update(targets[accepted&(winners==1)].tolist())
        recover=keyed_uniform(active,run,step,2)<.25; state[active[recover]]=3
        active=np.array(sorted(active[~recover].tolist()+new),int)
        for c in [0,1]: peaks[c]=max(peaks[c],sum(state[v]==(2 if c==0 else 1) and v not in seeds for v in active))
    sec0=len(ever0)-len(neg); sec1=len(ever1)-len(pos); both=exposed0&exposed1; total=sec0+sec1; total_seeds=len(seeds)
    def outcome(a,b):
        if a+b==0: return 'NA'
        return 'mixed_close' if abs(a-b)/(a+b)<=.05 else 'Target_0_win' if a>b else 'Target_1_win'
    return dict(seed_0=len(neg),seed_1=len(pos),final_reached_0=len(ever0),final_reached_1=len(ever1),secondary_0=sec0,secondary_1=sec1,secondary_total=total,
                amplification_0=sec0/len(neg) if len(neg) else np.nan,amplification_1=sec1/len(pos) if len(pos) else np.nan,amplification_total=total/total_seeds,
                reach_fraction_excluding_seeds=total/(n-total_seeds),peak_active_secondary_0=peaks[0],peak_active_secondary_1=peaks[1],
                exposed_0_only=len(exposed0-exposed1),exposed_1_only=len(exposed1-exposed0),exposed_both=len(both),exposed_union=len(exposed0|exposed1),
                contested_node_fraction=len(both)/(n-total_seeds),contested_fraction_of_exposed=len(both)/len(exposed0|exposed1) if exposed0|exposed1 else np.nan,
                competing_exposure_secondary_adopter_fraction=len((ever0|ever1)&both)/total if total else np.nan,
                competing_exposure_all_adopter_fraction=len((ever0|ever1)&both)/(total+total_seeds),competition_resolution_events=resolutions,
                C0_minus_C1_total=(len(ever0)-len(ever1))/(len(ever0)+len(ever1)),C0_minus_C1_secondary=(sec0-sec1)/total if total else np.nan,
                outcome_total=outcome(len(ever0),len(ever1)),outcome_secondary=outcome(sec0,sec1),active_at_horizon=len(active),steps=step+1)

def mc_summary(a):
    x=np.asarray(a,float); x=x[np.isfinite(x)]
    if not len(x): return dict(n=0,mean=np.nan,sd=np.nan,median=np.nan,q25=np.nan,q75=np.nan,iqr=np.nan,ci95_low=np.nan,ci95_high=np.nan)
    sd=x.std(ddof=1) if len(x)>1 else np.nan; delta=stats.t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)) if len(x)>1 else np.nan; q=np.quantile(x,[.25,.5,.75])
    return dict(n=len(x),mean=x.mean(),sd=sd,median=q[1],q25=q[0],q75=q[2],iqr=q[2]-q[0],ci95_low=x.mean()-delta,ci95_high=x.mean()+delta)

def simulations():
    nodes,graphs,max_weight,missing=simulation_graphs(); users=nodes.node.tolist(); ids={u:i for i,u in enumerate(users)}; n=len(users)
    monthly=read('data/processed/diffusion_ready_monthly_node_features_full.csv'); monthly['node']=monthly.node_id.map(norm); monthly=monthly[monthly.month.astype(str)<='2022-11'].copy()
    monthly['polarity_sum']=monthly.avg_polarity*monthly.tweet_count
    h=monthly.groupby('node')[['tweet_count','negative_count','positive_count','polarity_sum']].sum().reindex(users).fillna(0)
    p0=(h.negative_count/h.tweet_count.replace(0,np.nan)).fillna(0).to_numpy(); p1=(h.positive_count/h.tweet_count.replace(0,np.nan)).fillna(0).to_numpy()
    polarity=(h.polarity_sum/h.tweet_count.replace(0,np.nan)).fillna(0).clip(-1,1).to_numpy(); trust=nodes.trust_score.clip(0,1).to_numpy(); pr=nodes.pagerank.to_numpy(); pr=(pr-pr.min())/(pr.max()-pr.min())
    features=(p0,p1,trust,pr,polarity); elig0=np.flatnonzero(p0>0); elig1=np.flatnonzero(p1>0)
    settings=[('baseline_beta_005',.05,.05,25,25,False),('symmetric_baseline',.14,.14,25,25,False),('baseline_beta_030',.30,.30,25,25,False),
              ('full_balanced',.14,.14,25,25,True),('Target_0_advantage',.28,.14,25,25,True),('Target_1_advantage',.14,.28,25,25,True),
              ('asymmetric_40_10',.14,.14,40,10,True),('asymmetric_10_40',.14,.14,10,40,True)]
    csv('reviewer_simulation_configuration.csv',[dict(configuration=c,beta_0=b0,beta_1=b1,seeds_0=s0,seeds_1=s1,full_modifiers=full,runs=2000,recovery=.25,max_steps=25,seed=SEED) for c,b0,b1,s0,s1,full in settings])
    arrays={}
    for direction,g in graphs.items():
        adj=[[] for _ in users]
        for u,v,d in g.edges(data=True): adj[ids[u]].append((ids[v],min(d['weight']/max_weight,1.)))
        for a in adj: a.sort()
        arrays[direction]=adj
    rows=[]
    for config,b0,b1,s0,s1,full in settings:
        for run in range(2000):
            rng=np.random.default_rng(SEED+run); neg=sorted(rng.choice(elig0,s0,replace=False).tolist()); pos=sorted(rng.choice(np.setdiff1d(elig1,neg),s1,replace=False).tolist())
            for direction in graphs:
                r=simulator(arrays[direction],n,neg,pos,b0,b1,run,features,full); rows.append(dict(configuration=config,direction=direction,run=run+1,RNG_seed=SEED+run,beta_0=b0,beta_1=b1,full_modifiers=full,graph_nodes=n,**r))
        print('Simulation complete:',config,flush=True)
    d=pd.DataFrame(rows); csv('secondary_reach_simulation_runs.csv',d)
    numeric=['secondary_0','secondary_1','secondary_total','amplification_0','amplification_1','amplification_total','reach_fraction_excluding_seeds','peak_active_secondary_0','peak_active_secondary_1','final_reached_0','final_reached_1','C0_minus_C1_total','C0_minus_C1_secondary','active_at_horizon']
    out=[]; exposure=[]
    expmetrics=['exposed_0_only','exposed_1_only','exposed_both','exposed_union','contested_node_fraction','contested_fraction_of_exposed','competing_exposure_secondary_adopter_fraction','competing_exposure_all_adopter_fraction','competition_resolution_events']
    for key,z in d.groupby(['configuration','direction'],sort=False):
        for col in numeric: out.append(dict(configuration=key[0],direction=key[1],metric=col,undefined_runs=int(z[col].isna().sum()),**mc_summary(z[col])))
        for col in expmetrics: exposure.append(dict(configuration=key[0],direction=key[1],metric=col,undefined_runs=int(z[col].isna().sum()),**mc_summary(z[col])))
    csv('secondary_reach_simulation_results.csv',out); csv('competition_exposure_overlap.csv',exposure)
    compare=[]
    for config,z in d.groupby('configuration',sort=False):
        for direction in ['reversed','layer_aware']:
            for metric in ['secondary_total','amplification_total','C0_minus_C1_total','C0_minus_C1_secondary','contested_node_fraction']:
                base=z[z.direction=='recorded'].set_index('run')[metric]; control=z[z.direction==direction].set_index('run')[metric]; diff=(control-base).dropna(); s=mc_summary(diff)
                compare.append(dict(configuration=config,comparison=direction+' minus recorded',metric=metric,paired_runs=len(diff),paired_standardized_effect_dz=s['mean']/s['sd'] if s['sd'] else np.nan,**s))
    csv('direction_simulation_comparison.csv',compare)
    outcomes=[]
    for key,z in d.groupby(['configuration','direction'],sort=False):
        for definition in ['total','secondary']:
            for label in ['Target_0_win','Target_1_win','mixed_close','NA']: outcomes.append(dict(configuration=key[0],direction=key[1],definition=definition,outcome=label,runs=int(z['outcome_'+definition].eq(label).sum()),fraction=z['outcome_'+definition].eq(label).mean()))
    csv('competition_outcome_definition_comparison.csv',outcomes)
    txt('reviewer_simulation_assumptions.txt',f'Original model population ({n} normalized original diffusion_model_nodes) and supplied model-edge Weight preserved. Recorded and all-reversed use same weights. Layer-aware allocates EACH supplied pair Weight according to reconstructed type event proportions, reversing its retweet share and retaining mention/reply share; aggregated directed weights then divided by the original recorded maximum {max_weight}, clipped to 1. Missing type evidence for {missing} original pairs: these remain recorded; no fabricated type. This is a controlled direction reallocation on the original population, distinct from the full event-count network used for layer/temporal analysis. Node normalization did not merge original model IDs (asserted).\nOriginal synchronous competitive SIR: states susceptible/Target1/Target0/recovered, no reinfection, conditional contagion proportional to aggregate exposure, max 25 steps, recovery .25. Base .14, seeds 25 each; full modifiers source trust, normalized original PageRank, historical ratio and TextBlob aggregate polarity match original notebook formulas; feature cutoff through 2022-11 (not necessarily before all network events). Baseline uses edge weights only. Historical eligible seed selection (ratio>0), disjoint random seeds, no future simulated information. Random base 42 rather than original notebook 20260905 per user instruction. Keyed pseudorandom uniforms by run/step/node/channel provide COMMON random numbers across directions and cells, retaining original transition probabilities while avoiding direction-dependent RNG consumption. Simulation is a static hypothetical process, not timestamp-constrained or calibrated to observed adoption. Finite-horizon active count and uncertainty reported, not extrapolated to eventual absorption.\nSecondary reach excludes all seeds; reach fraction denominator n minus seeds. Peak excludes the identities of initial seeds, not an arithmetic subtraction from an old peak. Exposure means positive-probability transmission opportunity to a still-susceptible node; it is not a verified observed contact or successful infection. Ever-both counts either simultaneous or different-step opportunities before adoption; actual resolution counts successful adoption with both types simultaneously positive. Contested-node fraction denominator n minus seeds; exposed-union and adopter fractions also separate. C statistic signed (C0-C1)/(C0+C1), plus original 5% mixed-close classification, calculated on totals and secondary counts. Zero secondary denominator is NA. 2000 paired runs per cell, mean t CI reflects Monte Carlo variation conditional on graph and settings; SD, median, IQR and paired standardized dz saved. No equivalence margin was justified or equivalence test predeclared; intervals overlapping zero are uncertainty, not evidence of no effect. Beta advantages and asymmetric allocations are sensitivity settings, not estimated effects.')
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained'); z=d[d.configuration=='symmetric_baseline']
    for ax,direction in zip(axs,graphs): ax.hist(z[z.direction==direction].amplification_total,bins=30,color=COLORS[0]); ax.set(title=direction,xlabel='Secondary adoptions / initial seeds',ylabel='Run count')
    plot('simulation_amplification_distribution',fig,z)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained'); sums=pd.DataFrame(out); pdata=[]
    for k,direction in enumerate(graphs):
        z=d[(d.direction==direction)&~d.full_modifiers]; q=z.groupby('beta_0').secondary_total.agg(['mean','std','count']).reset_index(); q['ci']=1.96*q['std']/np.sqrt(q['count']); q['direction']=direction; pdata.extend(q.to_dict('records')); ax.errorbar(q.beta_0,q['mean'],yerr=q.ci,label=direction,color=COLORS[k],marker='o')
    ax.set(xlabel='Symmetric transmission beta_0 = beta_1',ylabel='Mean final secondary reach'); ax.legend(); plot('simulation_secondary_reach_vs_beta',fig,pdata)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained'); q=pd.DataFrame(exposure); q=q[(q.metric=='contested_node_fraction')&(q.direction=='recorded')]; ax.errorbar(q['mean'],range(len(q)),xerr=[q['mean']-q.ci95_low,q.ci95_high-q['mean']],fmt='o',color=COLORS[0]); ax.set_yticks(range(len(q)),q.configuration); ax.set(xlabel='Ever-both exposure / non-seed population'); plot('simulation_contested_fraction',fig,q)
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained'); q=pd.DataFrame(compare); q=q[q.metric=='secondary_total']; ax.errorbar(q['mean'],range(len(q)),xerr=[q['mean']-q.ci95_low,q.ci95_high-q['mean']],fmt='o',color=COLORS[0]); ax.axvline(0,color=COLORS[6],ls=':'); ax.set_yticks(range(len(q)),q.configuration+' / '+q.comparison,fontsize=9); ax.set(xlabel='Paired difference in secondary adoptions (95% Monte Carlo CI)'); plot('simulation_direction_forest',fig,q)
    fig,axs=plt.subplots(1,2,figsize=(11,5),layout='constrained'); q=d[(d.configuration=='full_balanced')&(d.direction=='recorded')]
    for ax,col,title in zip(axs,['C0_minus_C1_secondary','C0_minus_C1_total'],['Primary: secondary C_0 / C_1','Sensitivity: total C_0 / C_1']): ax.hist(q[col].dropna(),bins=np.linspace(-1,1,21),color=COLORS[0]); ax.set(title=title,xlabel='(C_0 - C_1) / (C_0 + C_1)',ylabel='Run count'); ax.text(.03,.95,f'NA runs: {q[col].isna().sum()}',transform=ax.transAxes,va='top')
    plot('competition_secondary_outcome',fig,q)

def spectral():
    nodes,model,maxweight,missing=simulation_graphs(); ev=read('data/processed/reconstructed_temporal_edges.csv'); rows=[]; vectors=[]
    for population,graphs in [('original_model_population',model),('full_temporal_endpoint_population',{d:graph(ev,d) for d in ['recorded','reversed','layer_aware']})]:
        for direction,base in graphs.items():
            for loops in [True,False]:
                g=base.copy()
                if not loops: g.remove_edges_from(list(nx.selfloop_edges(g)))
                comps=list(nx.strongly_connected_components(g)); largest=max(map(len,comps)); best=(0.,[],np.array([]),np.array([[]]))
                # Exact SCC-block eigenvalues: triangular SCC decomposition has union spectrum.
                for component in comps:
                    if len(component)==1 and not g.has_edge(next(iter(component)),next(iter(component))): continue
                    us=sorted(component); a=nx.to_numpy_array(g,nodelist=us,weight='weight'); vals,vecs=np.linalg.eig(a); k=int(np.argmax(np.abs(vals))); rho=float(abs(vals[k]))
                    if rho>best[0]: best=(rho,us,np.abs(vecs[:,k]),a)
                rho,us,vec,a=best; vec=vec/vec.sum() if len(vec) and vec.sum() else vec
                denom=maxweight if population=='original_model_population' else max((d['weight'] for _,_,d in base.edges(data=True)),default=1)
                # Audit original max-normalized operator, with clipping where reallocation aggregates.
                gn=g.copy()
                for _,_,d in gn.edges(data=True): d['weight']=min(d['weight']/denom,1.)
                normalized_rho=0.
                for component in nx.strongly_connected_components(gn):
                    if len(component)==1 and not gn.has_edge(next(iter(component)),next(iter(component))): continue
                    vals=np.linalg.eigvals(nx.to_numpy_array(gn,nodelist=sorted(component),weight='weight')); normalized_rho=max(normalized_rho,float(np.abs(vals).max()))
                order=np.argsort(-vec)[:20] if len(vec) else []
                rows.append(dict(population=population,direction=direction,self_loops_retained=loops,nodes=len(g),edges=g.number_of_edges(),self_loops=nx.number_of_selfloops(g),largest_SCC=largest,
                                 spectral_radius_weighted=rho,dominant_SCC_nodes=len(us),dominant_SCC_edges=g.subgraph(us).number_of_edges(),dominant_nodes=';'.join(us[i] for i in order),
                                 max_weight_denominator=denom,normalized_spectral_radius=normalized_rho,recovery=.25,beta_c_spectral_diagnostic=.25/normalized_rho if normalized_rho else np.nan,
                                 top1_eigenvector_mass=float(vec.max()) if len(vec) else np.nan,operator='weighted adjacency Source row Target column; right Perron SCC block vector'))
                for i,u in enumerate(us): vectors.append(dict(population=population,direction=direction,self_loops_retained=loops,node=u,dominant_SCC_right_eigenvector_mass=vec[i]))
                if len(us):
                    edgecon=[]
                    for i,u in enumerate(us):
                        for j,v in enumerate(us):
                            if a[i,j]>0: edgecon.append(dict(population=population,direction=direction,self_loops_retained=loops,source=u,target=v,weight=a[i,j],right_eigenvector_contribution=a[i,j]*vec[j]))
                    csv(f'spectral_dominant_edges_{population}_{direction}_{"loops" if loops else "no_loops"}.csv',edgecon)
                print('Spectral complete:',population,direction,loops,rho,largest,flush=True)
    csv('spectral_threshold_audit.csv',rows); csv('spectral_dominant_block_eigenvectors.csv',vectors)
    txt('spectral_threshold_audit_report.txt','The saved diffusion notebook does not define a spectral operator or beta_c. We therefore audit explicitly specified weighted adjacency and max-normalized adjacency, with recovery .25 from the saved simulator. beta_c=.25/rho(normalized A) is a linear spectral diagnostic, NOT a validated competitive finite-horizon SIR epidemic threshold. No undocumented original threshold definition is inferred.\nSpectrum is evaluated by exact dense eigendecomposition of each nontrivial SCC or singleton self-loop: block-triangular SCC decomposition makes the maximum block radius the full adjacency spectral radius. Largest SCC and dominant spectral SCC need not coincide. A transposed operator has exactly the same spectrum, so recorded and all-reversed should agree; this is algebraic and does not imply equal directed reach. Layer-aware transformation changes cycles and may change the spectrum. Eigenvector support is reported for the dominant SCC block (not a unique full-graph eigenvector, whose mass can extend downstream and depend on multiplicity); edge contributions are A_ij*v_j, local diagnostic not causal attribution. Self-loop removal evaluated separately. For a DAG rho=0 and a finite reciprocal threshold is NA. Full temporal graphs use reconstructed event weights; original-model graph uses supplied Weight allocated by type as documented in simulator assumptions. Tiny recurrent SCCs can dominate despite most nodes lying outside them. No quasi-stationary SIS assumptions, homogeneous transmission, infinite horizon, calibrated recovery, or population-scale recurrence were validated.\n'+pd.DataFrame(rows).to_string(index=False))

def forecasting():
    dataset=read('data/processed/weekly_leakage_controlled_prediction_dataset.csv'); test=dataset[dataset.Week_Start>='2022-12-19']; weeks=sorted(test.Week_Start.unique()); cohort_count=len(test); users=test.User.nunique()
    base=read('results/tables/graph_models_summary.csv'); hist=read('results/tables/history_graph_summary.csv'); nod=read('results/tables/node_only_neural_summary.csv')
    metrics=['macro_F1','balanced_accuracy','MCC','class_0_PR_AUC','ROC_AUC','Brier_score','no_skill_class_0_PR_AUC']; rows=[]
    specs=[('Majority','Majority'),('Persistence','Persistence'),('Logistic Regression','Logistic Regression'),('MLP','MLP'),('Original GRU','GRU (legacy lagged)'),('Static GCN','Static GCN'),('GraphSAGE','GraphSAGE'),('GAT forward','GAT'),('GAT reversed','GAT (reversed)'),('Aligned GRU','GRU (aligned)'),('Temporal GCN / GRU-over-GCN','Temporal GCN')]
    structural='prior-week in/out degree, in/out event strength, PageRank, reverse PageRank'
    history='Previous_Label, Previous_Label_Available, Past_Class0_Rate, Past_Class1_Rate, Past_Label_Count, Weeks_Since_Last_Label'
    for display,model in specs:
        z=base[base.Model==model]; r=dict(model=display,saved_model=model,cohort='All eligible: active, non-tie weekly label, historical graph features available',test_user_weeks=cohort_count,test_unique_users=users,test_weeks=';'.join(weeks),seed_values='42' if model in ['Majority','Persistence','Logistic Regression'] else ';'.join(map(str,range(42,52))),
                                      feature_set='last available prior orientation label; train-prior fallback' if model=='Persistence' else 'training class prevalence' if model=='Majority' else structural,
                                      graph_direction='reversed' if 'reversed' in model else 'none (features from recorded prior graphs)' if model in ['Majority','Persistence','Logistic Regression','MLP','GRU (aligned)','GRU (legacy lagged)'] else 'recorded incoming message aggregation',
                                      source='results/tables/graph_models_summary.csv',CI_definition='saved seed/week bootstrap 95%, not recomputed or pooled with other cohorts')
        for metric in metrics:
            q=z[z.Metric==metric]
            r[metric]=q.Mean.iloc[0] if len(q) else np.nan; r[metric+'_ci95_low']=q.Bootstrap_95_Lower.iloc[0] if len(q) else np.nan; r[metric+'_ci95_high']=q.Bootstrap_95_Upper.iloc[0] if len(q) else np.nan
        rows.append(r)
    for cohort in hist.Cohort.unique():
        for model in hist.Model.unique():
            z=hist[(hist.Cohort==cohort)&(hist.Model==model)]
            if not len(z): continue
            weekly=read('results/tables/history_graph_metrics_by_seed_week.csv'); q=weekly[(weekly.Cohort==cohort)&(weekly.Model==model)]
            if len(q): count=int(q.groupby('Week_Start').Test_Users.first().sum())
            else: count=cohort_count if cohort=='All eligible' else np.nan
            r=dict(model=model,saved_model=model,cohort=cohort,test_user_weeks=count,test_unique_users=users if cohort=='All eligible' else np.nan,test_weeks=';'.join(weeks),seed_values='42' if model in ['History Logistic','Persistence'] else ';'.join(map(str,range(42,52))),
                   feature_set=history if model=='History Logistic' else 'last available prior orientation label; train-prior fallback' if model=='Persistence' else structural if model=='Reversed GAT (no history)' else structural+'; '+history,
                   graph_direction='reversed' if 'Reversed' in model else 'recorded' if 'Forward' in model else 'none (prior structural features)' if 'Structural' in model else 'none',source='results/tables/history_graph_summary.csv',CI_definition='saved ordinary bootstrap over week means; seeds averaged within week')
            for metric in metrics:
                q=z[z.Metric==metric]; r[metric]=q.Mean.iloc[0] if len(q) else np.nan; r[metric+'_ci95_low']=q.Ordinary_95_Lower.iloc[0] if len(q) else np.nan; r[metric+'_ci95_high']=q.Ordinary_95_Upper.iloc[0] if len(q) else np.nan
            rows.append(r)
    # Old monthly outputs are explicitly a different task/cohort, never blended.
    old=read('results/tables/stage8_final_paper_results.csv')
    for _,r in old.iterrows(): rows.append(dict(model=r['model'],saved_model=r['model'],cohort='Original monthly node task: '+r.protocol,test_user_weeks=np.nan,test_node_count=int(r.tn+r.fp+r.fn+r.tp),test_weeks='NA (monthly task; consult saved methodology)',seed_values='NA (not in this summary)',feature_set='See stage8_final_paper_methodology.md',graph_direction='See saved monthly methodology',macro_F1=r.macro_f1,balanced_accuracy=r.balanced_accuracy,MCC=np.nan,class_0_PR_AUC=np.nan,ROC_AUC=np.nan,Brier_score=np.nan,source='results/tables/stage8_final_paper_results.csv',CI_definition='NA (no interval in saved summary)'))
    csv('all_forecasting_models_reconciled.csv',rows)
    bounds=[]
    for r in rows: bounds.append({k:r.get(k,np.nan) for k in ['model','cohort','test_user_weeks','test_unique_users','test_node_count','test_weeks','seed_values','feature_set','graph_direction','source','CI_definition']})
    bounds.extend([dict(model='Original static diffusion population',cohort='exact matching largest weak component',test_node_count=17598,feature_set='full-period labels descriptive; simulator orientation cutoff through Nov 2022',graph_direction='supplied Source -> Target'),dict(model='Full temporal network',cohort='normalized reconstructed endpoint union',test_node_count=60219,feature_set='no restriction to tweet authors or original WCC',graph_direction='author -> text referenced account')])
    csv('forecasting_population_and_information_boundary.csv',bounds)
    # Independent read-only mean check; no fitting, no rerun of forecasts.
    weekly=read('results/tables/graph_forecasting_all_model_weekly_metrics.csv'); checks=[]
    for _,r in base.iterrows():
        if r.Metric in weekly.columns:
            z=weekly[weekly.Model==r.Model]; v=z[r.Metric].mean()
            checks.append(dict(model=r.Model,metric=r.Metric,saved_mean=r.Mean,weekly_mean=v,difference=v-r.Mean,within_1e_10=bool(abs(v-r.Mean)<1e-10)))
    csv('forecasting_saved_mean_consistency.csv',checks)
    aligned='Aligned GRU is the saved ProjectedGRU in weekly_graph_forecasting.py: Linear(6,hidden) -> ReLU -> GRU(hidden,hidden) -> dropout -> linear binary head, using raw structural feature sequences under the same weekly cache/cutoffs, scaler, validation grid, early stopping and seeds as Temporal GCN, with graph aggregation replaced by raw sequences (feature_only). Temporal GCN has fixed GCN weights and weekly GCN -> projection -> GRU node states; it is not parameter-evolving EvolveGCN. Original GRU uses run_node_only_neural_baselines.py GRU(6,hidden) directly on strictly earlier historical feature rows, with its separate six-candidate grid and sequence construction; the graph summary labels it GRU (legacy lagged). These are separate architectures and temporal alignment controls, not interchangeable results.'
    txt('forecasting_consistency_report.txt',f'No models trained; saved summaries and weekly means audited. Weekly primary cohort: {cohort_count} user-weeks, {users} unique users, {len(weeks)} test weeks ({weeks[0]} through {weeks[-1]}). Active weekly labels are majority Target, ties excluded; test structural inputs strictly precede prediction week, training precedes validation precedes test. Different static/monthly cohorts remain separate.\n{aligned}\nChance references: balanced accuracy=.5, AUROC=.5, MCC=0 for independent no-information predictions when defined. Class-0 average-precision no-skill baseline is test class-0 prevalence, varying by week. Brier no-information baseline at prevalence p is p(1-p) for calibrated constant score, and .25 for p=.5; no universal .5 macro-F1 baseline is assumed. Persistence is a meaningful task baseline distinct from chance.\nSaved graph-only balanced accuracies are about .5006-.5091, with original GRU .5055 and structural logistic .5121. These are small departures from chance; significance and material practical value are distinct and no materiality margin was predeclared. Persistence .5403 and history-only/structural-history models about .5535 give modest predictive signal; saying ALL forecasting is chance would contradict these saved outputs. Original monthly models with balanced accuracy .5 can achieve .8369 accuracy by the majority class and should not be characterized as strong forecasting. Saved history supplement reports no corrected macro-F1 superiority over Persistence; class-0 AP improves for history logistic and history+structural MLP, and balanced accuracy/MCC improve for history+structural MLP. History+reversed GAT fails to add incremental value over matched history+structural MLP. Read saved paired/Holm findings for inference, not just individual CIs.\nAll missing metadata/metrics marked NA; no invented seeds, unavailable CIs or AUROC. Subject population-specific claims require saved protocol citations.\n'+pd.DataFrame(rows)[['model','cohort','macro_F1','balanced_accuracy','MCC','class_0_PR_AUC']].to_string(index=False))
    txt('aligned_GRU_definition.txt',aligned)
    csv('forecasting_chance_references.csv',[dict(metric='balanced_accuracy',reference=.5,assumption='independent no-information predictions'),dict(metric='ROC_AUC',reference=.5,assumption='independent scores with both classes'),dict(metric='MCC',reference=0.,assumption='independent nondegenerate predictions; constants convention zero'),dict(metric='class_0_PR_AUC',reference=weekly[weekly.Model=='Persistence'].no_skill_class_0_PR_AUC.mean(),assumption='mean test class-0 prevalence across weeks'),dict(metric='Brier_score',reference=.25,assumption='constant .5 forecast'),dict(metric='macro_F1',reference=np.nan,assumption='depends on predicted class allocation and prevalence; no universal .5 reference')])
    d=pd.DataFrame(rows); z=d[d.cohort.str.startswith('All eligible') & ~d.model.eq('Reversed GAT (no history)')].drop_duplicates('model')
    fig,axs=plt.subplots(1,2,figsize=(13,8),layout='constrained')
    for ax,metric in zip(axs,['balanced_accuracy','class_0_PR_AUC']):
        for k,(_,r) in enumerate(z.iterrows()):
            ax.plot([r[metric+'_ci95_low'],r[metric+'_ci95_high']],[k,k],color=COLORS[0]); ax.scatter(r[metric],k,color=COLORS[0])
        ax.set_yticks(range(len(z)),z.model,fontsize=9); ax.set(xlabel=metric+' (saved 95% CI)'); ref=.5 if metric=='balanced_accuracy' else weekly[weekly.Model=='Persistence'].no_skill_class_0_PR_AUC.mean(); ax.axvline(ref,color=COLORS[6],ls=':',label='Chance / no-skill'); pers=base[(base.Model=='Persistence')&(base.Metric==metric)].Mean.iloc[0]; ax.axvline(pers,color=COLORS[1],ls='--',label='Persistence mean'); ax.legend(fontsize=9,loc='lower right')
    plot('forecasting_reconciled_uncertainty',fig,z)

def figures():
    match=read('results/tables/matching_sensitivity_comparison.csv'); ev=read('data/processed/reconstructed_temporal_edges.csv'); tweets=read('data/raw/master_tweets.csv')
    # Separate dimensions: user counts and edge counts have different panels.
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained'); source=[]
    for ax,cols,title in zip(axs,[['unique_tweet_users','matched_users','largest_WCC_nodes'],['induced_directed_edges','largest_WCC_edges']],['User population','Directed edge population']):
        for k,(_,r) in enumerate(match.iterrows()):
            values=[r[c] for c in cols]; ax.bar(np.arange(len(cols))+(k-.5)*.35,values,width=.35,label=r.rule,color=COLORS[k]); source.extend(dict(panel=title,rule=r.rule,stage=c,count=r[c]) for c in cols)
        ax.set_xticks(range(len(cols)),[c.replace('_','\n') for c in cols],fontsize=10); ax.set(title=title,ylabel='Users' if title=='User population' else 'Unique edges'); ax.legend()
    plot('population_funnel_separate_units',fig,source)
    dates=pd.to_datetime(tweets.Date,utc=True); week=dates.dt.tz_localize(None).dt.normalize()-pd.to_timedelta(dates.dt.dayofweek,unit='d'); tweets['week']=week
    q=tweets.groupby('week').agg(volume=('Target','size'),target_0=('Target',lambda v:(v==0).sum()),target_1=('Target',lambda v:(v==1).sum())).reset_index(); q['target_0_share']=q.target_0/q.volume; q['partial_boundary_week']=q.week.isin([q.week.min(),q.week.max()])
    fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained'); axs[0].plot(q.week,q.volume,color=COLORS[0]); axs[1].plot(q.week,q.target_0_share,color=COLORS[1]); axs[0].set(ylabel='Tweet volume'); axs[1].set(ylabel='Target 0 share',ylim=(0,1),xlabel='Week beginning Monday (UTC)')
    for ax in axs:
        for w in q[q.partial_boundary_week].week: ax.axvspan(w,w+pd.Timedelta(days=7),color=COLORS[6],alpha=.15)
    plot('weekly_activity_split_panels',fig,q)
    p=read('results/tables/textblob_polarity_recomputed.csv'); agreement=read('results/tables/target_textblob_agreement_recheck.csv'); source=[]
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for k,label in enumerate([0,1]):
        z=p[p.Target==label]; counts=[int((z.polarity<0).sum()),int((z.polarity==0).sum()),int((z.polarity>0).sum())]; axs[0].bar(np.arange(3)+(k-.5)*.35,np.array(counts)/len(z),width=.35,color=COLORS[k],label=f'Target {label}')
        for sign,count in zip(['negative','exact zero','positive'],counts): source.append(dict(panel='sign proportions',Target=label,sign=sign,count=count,denominator=len(z),proportion=count/len(z)))
        a=z.polarity[z.polarity!=0]; counts2,bins=np.histogram(a,bins=np.linspace(-1,1,41)); axs[1].stairs(counts2,bins,label=f'Target {label}; zero excluded',color=COLORS[k])
        source.extend(dict(panel='nonzero distribution',Target=label,bin_left=bins[i],bin_right=bins[i+1],count=counts2[i],zero_excluded_count=int((z.polarity==0).sum())) for i in range(40))
    axs[0].set_xticks(range(3),['negative','exact zero','positive']); axs[0].set(ylabel='Class proportion'); axs[1].set(xlabel='TextBlob polarity (exact zero shown in left panel)',ylabel='Nonzero tweet count'); axs[0].legend(); axs[1].legend(fontsize=9); plot('textblob_zero_spike_visible',fig,source)
    nodes=read('data/processed/diffusion_model_nodes.csv'); data=[]; fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,kind,cols in zip(axs,['Degree','Strength'],[['in_degree','out_degree'],['weighted_in_degree','weighted_out_degree']]):
        for k,col in enumerate(cols):
            a=nodes[col].to_numpy(); positive=np.unique(a[a>0]); ccdf=np.array([(a>=v).mean() for v in positive]); ax.loglog(positive,ccdf,color=COLORS[k],label=col); data.extend(dict(population='original model WCC',measure=col,value=v,ccdf=c,zero_nodes=int((a==0).sum()),denominator=len(a)) for v,c in zip(positive,ccdf))
        ax.set(xlabel=kind,ylabel='P(X >= x)'); ax.legend(fontsize=9); ax.grid(alpha=.2)
    plot('degree_strength_loglog_CCDF',fig,data)
    g=nx.DiGraph(); g.add_nodes_from(nodes.node_id); e=read('data/processed/diffusion_model_edges.csv'); g.add_edges_from(zip(e.Source,e.Target)); b=bowtie(g); fig,ax=plt.subplots(figsize=(8,4),layout='constrained'); labels=list(b); vals=list(b.values()); ax.hlines(range(4),1,vals,color=COLORS[0]); ax.scatter(vals,range(4),color=COLORS[0]); ax.set_xscale('log'); ax.set_yticks(range(4),labels); ax.set(xlabel='Original WCC node count (log scale)')
    for i,v in enumerate(vals): ax.annotate(f'{v:,}',(v,i),xytext=(5,5),textcoords='offset points')
    plot('original_WCC_bowtie_readable',fig,[dict(region=k,nodes=v,population_nodes=len(g)) for k,v in b.items()])
    # Re-express the original sensitivity grid using its explicitly saved seed_count.
    old=read('results/tables/diffusion_parameter_sensitivity_complete_corrected.csv'); old['secondary_total']=old.final_negative_cascade_size+old.final_positive_cascade_size-2*old.seed_count; assert old.secondary_total.min()>=0
    old['amplification']=old.secondary_total/(2*old.seed_count); old['seed_fraction_total']=2*old.seed_count/17598
    q=old[(old.normalization=='max')&(old.competition_rule=='proportional')].groupby(['seed_count','seed_fraction_total','base_transmission','recovery']).agg(amplification=('amplification','mean'),secondary_total=('secondary_total','mean'),runs=('run','size')).reset_index(); seeds=sorted(q.seed_count.unique()); fig,axs=plt.subplots(1,len(seeds),figsize=(14,4),layout='constrained'); ranges=[]
    for ax,s in zip(axs,seeds):
        z=q[q.seed_count==s]; m=z.pivot(index='recovery',columns='base_transmission',values='amplification'); lo,hi=float(m.min().min()),float(m.max().max()); im=ax.imshow(m,cmap='cividis',vmin=lo,vmax=hi if hi>lo else lo+1e-9); ax.set_xticks(range(len(m.columns)),[f'{v:g}' for v in m.columns]); ax.set_yticks(range(len(m.index)),[f'{v:g}' for v in m.index]); ax.set(title=f'{s} seeds per type\nrange {lo:.3g} to {hi:.3g}',xlabel='beta_0 = beta_1',ylabel='Recovery'); fig.colorbar(im,ax=ax,label='Mean amplification',shrink=.8); ranges.append(dict(seed_count=s,vmin=lo,vmax=hi,scale='independent per panel'))
    plot('sensitivity_amplification_independent_scales',fig,q); csv('sensitivity_panel_scale_audit.csv',ranges)
    txt('figure_correction_assumptions.txt','All corrected plots exported SVG + PNG at 400 DPI with editable SVG fonts, DejaVu Sans >=9pt in legends/ticks, Okabe-Ito colors/cividis, no dual axes/3D. Every plot has a source CSV. Population plots separate users and unique directed pairs. Boundary-week shading denotes partial observed weeks, not known collection completeness. TextBlob zero has its own sign category; right histogram explicitly excludes exact zero and saves bin edges/counts. Degree/strength CCDF denominators include zero nodes although x=0 is omitted on log scale; no power-law fitting or label. Original WCC bow-tie kept separate from full endpoint layer figures. Competition primary uses secondary adoptions and NA for no secondary adoption. Sensitivity plots derive secondary counts from saved seed_count, use per-panel numeric scales/ranges, preserve original settings and do not rerun old simulations. Forecast CI type and chance references are stated in source/report. Full per-seed temporal source retains stratum overlaps; plots describe this design. User-supplied manuscript was not present; no LaTeX modified.')

def supplements():
    d=read('results/tables/typed_layer_network_statistics.csv'); q=d[(d.direction=='recorded')&(d.layer!='pooled')]
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,col in zip(axs,['nodes','edges','events']):
        ax.bar(q.layer,q[col],color=COLORS[:3]); ax.set_yscale('log'); ax.set(ylabel=col.capitalize()+' (log scale)',ylim=(1,q[col].max()*4)); ax.tick_params(axis='x',labelrotation=15)
        for i,v in enumerate(q[col]): ax.annotate(f'{v:,}',(i,v),xytext=(0,5),textcoords='offset points',ha='center')
    plot('typed_layer_sizes_readable_log',fig,q)
    d=read('results/tables/time_respecting_reachability_by_seed.csv').drop_duplicates(['layer','direction','seed']); rng=np.random.default_rng(SEED); out=[]
    for key,z in d.groupby(['layer','direction'],sort=False):
        for metric in ['static_out_component','temporal_reachable','temporal_static_ratio','fraction_lost_time_order']:
            out.append(dict(layer=key[0],direction=key[1],stratum='unique union of four sampled strata',unique_sampled_seeds=len(z),zero_static_seeds=int(z.static_out_component.eq(0).sum()),metric=metric,**summary(z[metric],rng)))
    csv('time_respecting_reachability_unique_seed_summary.csv',out)
    txt('reviewer_analysis_validation_notes.txt','Superseded draft outputs are retained in results/reviewer_interim_20261009/ and must not be interpreted; archive_manifest.json records their original generated locations. They are not original project files. Regeneration corrected undefined global-shuffle correlations (p/z NA if any null statistic undefined), and domain-separated common-random-number keys across run/step/node/channel. Primary publication files live in results/tables and results/figures/reviewer_corrected. Use typed_layer_sizes_readable_log in preference to the earlier linear scale figure so the 16 retweet events are visible. All-seed summaries deduplicate the union of strata; the union is a targeted sample, not a representative random sample of the network. Temporal path hops are algorithm witness lengths, not longest temporal walks or guaranteed globally fewest-hop foremost paths; reach and arrival times are exact under the specified timestamp semantics. No homophily claim follows from the sparse retweet sample or the tested layer correlations. This file plus temporal_hop_interpretation_clarification.txt are authoritative clarifications of initial report wording.')

def master_report():
    m=read('results/tables/matching_sensitivity_comparison.csv').set_index('rule'); overlap=read('results/tables/reconstruction_overlap_reconciliation.csv'); o=overlap[overlap.definition=='normalized'].iloc[0]
    lexical=read('results/tables/target_textblob_agreement_recheck.csv'); lex=lexical[lexical.Target.astype(str)=='overall'].iloc[0]
    layers=read('results/tables/typed_layer_network_statistics.csv'); sim=read('results/tables/secondary_reach_simulation_results.csv'); exp=read('results/tables/competition_exposure_overlap.csv'); temp=read('results/tables/time_respecting_reachability_summary.csv'); spec=read('results/tables/spectral_threshold_audit.csv'); mix=read('results/tables/layer_orientation_permutation_tests.csv'); fc=read('results/tables/all_forecasting_models_reconciled.csv')
    def sval(config,metric,direction='recorded'):
        return sim[(sim.configuration==config)&(sim.metric==metric)&(sim.direction==direction)].iloc[0]
    sec=sval('full_balanced','secondary_total'); amp=sval('full_balanced','amplification_total'); baseline=sval('symmetric_baseline','secondary_total')
    ce=exp[(exp.configuration=='full_balanced')&(exp.direction=='recorded')&(exp.metric=='contested_node_fraction')].iloc[0]
    cr=exp[(exp.configuration=='full_balanced')&(exp.direction=='recorded')&(exp.metric=='competition_resolution_events')].iloc[0]
    tp=temp[(temp.layer=='pooled')&(temp.direction=='recorded')&(temp.stratum=='high_out_strength')&(temp.metric=='temporal_static_ratio')].iloc[0]
    tf=temp[(temp.layer=='pooled')&(temp.direction=='layer_aware')&(temp.stratum=='high_out_strength')&(temp.metric=='temporal_static_ratio')].iloc[0]
    sp=spec[(spec.population=='original_model_population')&(spec.direction=='recorded')]; loop=sp[sp.self_loops_retained].iloc[0]; noloop=sp[~sp.self_loops_retained].iloc[0]
    issues=[]
    def add(issue,concern,checked,result,status,sentence,file): issues.append(dict(issue=issue,concern=concern,checked=checked,result=result,status=status,sentence=sentence,file=file))
    add('Exact vs normalized username matching','Inconsistent literal and case-insensitive populations','Raw tweet/network endpoint sets, induced graph with all matched nodes including isolates',f'Exact WCC {int(m.loc["exact","largest_WCC_nodes"])} users/{int(m.loc["exact","largest_WCC_edges"])} edges; normalized {int(m.loc["normalized","largest_WCC_nodes"])} users/{int(m.loc["normalized","largest_WCC_edges"])} edges. WCC changes; existing model cohort remains preserved.','needs methodological correction','Literal matching selected 17,598 users and 29,715 directed edges in the largest weak component; trimming whitespace, removing one leading @ and casefolding selected 17,636 users and 29,825 edges, so the previously fitted models retain the original cohort while the matching sensitivity is reported separately.','matching_sensitivity_comparison.csv')
    add('99.8638% coverage denominator','Raw rows and unique pairs mixed','Literal and normalized unique pairs, reconstructed overlap, per-row membership',f'{int(o.overlap)}/{int(o.coverage_denominator)} = {100*o.supplied_pair_coverage:.8f}%; precision {int(o.overlap)}/{int(o.precision_denominator)} = {100*o.precision_overlap:.8f}%.','correct','Reconstructed interactions overlap 109,217 of 109,366 normalized unique supplied directed pairs (99.8638% pair coverage), and 109,217 of 109,219 reconstructed pairs (99.9982% precision-style overlap); the supplied file contains 109,443 rows.','reconstruction_overlap_reconciliation.csv')
    add('94.67% vs 94.61% lexical agreement','Disputed percentage and zero-polarity definition','TextBlob recomputed on every raw Tweet using exact saved formula, saved counts checked',f'{int(lex.agreement_numerator)}/{int(lex.agreement_denominator)} = {lex.agreement_percentage:.8f}%; rounds 94.67%, not 94.61%.','correct','Agreement between the binary Target label and indicator(TextBlob polarity > 0), with zero polarity assigned to the nonpositive class, is 80,937/85,495 (94.67%); Target-0 and Target-1 row agreements are 97.62% and 91.68%, respectively.','target_textblob_agreement_recheck.csv')
    add('Edge-direction semantics','Recorded action edges mistaken for verified content flow','Reconstruction author/reference semantics, layer controls and controlled weight reallocation','Retweet reversal is plausible content flow; mention and inferred reply have no verified content direction. Structural adjacency/spectral transpose invariance does not imply reach invariance.','needs wording correction','Recorded edges point from tweet author to a text-referenced account; retweet edges are reversed for a plausible content-flow sensitivity, whereas mention and inferred-reply direction remain uncertain and are tested in both orientations rather than treated as verified information transmission.','typed_direction_assumptions.txt')
    add('Typed retweet/reply/mention layers','Pooling hides interaction semantics','Separate weighted graphs and node-universe/layer overlaps','16 retweet events (14 pairs/20 endpoints), 41,024 inferred replies and 88,564 mentions. Native metadata absent; inferred reply is a leading-handle heuristic.','needs methodological correction','The 129,604 reconstructed events comprise 16 text-supported retweets, 41,024 inferred replies and 88,564 mentions; the sparse retweet layer and absent native reply/retweet metadata preclude broad conclusions about retweet propagation or verified reply behavior.','typed_layer_network_statistics.csv')
    add('Layer orientation mixing','Pooled association or global shuffle may reflect degree','2000 degree/strength-stratified and 2000 global permutations per layer; separate Holm families',f'Primary Holm p: '+', '.join(f'{r.layer}={r.Holm_p:.4g}' for _,r in mix[mix.null=='degree_strength_stratified'].iterrows())+'. Retweet global null has undefined draws and p=NA.','needs wording correction','None of the three layer-wise orientation correlations is supported after Holm correction under the degree/strength-stratified permutation null; these descriptive full-history associations do not establish political homophily.','layer_orientation_permutation_tests.csv')
    add('Static vs time-respecting reach','Static paths may violate event chronology','500 seeds per stratum where nodes available; non-decreasing timestamp earliest arrival',f'High-out-strength mean temporal/static reach {tp["mean"]:.6f} recorded ({tp.ci95_low:.6f}, {tp.ci95_high:.6f}); {tf["mean"]:.6f} layer-aware ({tf.ci95_low:.6f}, {tf.ci95_high:.6f}). Stratum-dependent reductions; many static sinks ratio=NA.','needs methodological correction',f'For the 500 highest-out-strength seeds, time-respecting reach retains on average {100*tp["mean"]:.2f}% of static descendants in recorded direction and {100*tf["mean"]:.2f}% under the retweet-reversed layer-aware convention, with substantial variation by seed stratum and no reach ratio assigned to static sinks.','time_respecting_reachability_summary.csv')
    add('Net-of-seed propagation reach','Final totals dominated by initial seeds','2000 paired Monte Carlo runs per configuration/direction; identity-based seed exclusion',f'Full balanced recorded mean secondary {sec["mean"]:.6f} (95% MC CI {sec.ci95_low:.6f}, {sec.ci95_high:.6f}); amplification {amp["mean"]:.6f}. Weighted baseline secondary {baseline["mean"]:.6f}.','needs wording correction',f'With 25 initial seeds per type, beta_0=beta_1=0.14 and recovery 0.25, the full recorded-direction model generates a mean of {sec["mean"]:.4f} secondary adoptions (95% Monte Carlo CI {sec.ci95_low:.4f}–{sec.ci95_high:.4f}) within 25 steps, corresponding to {amp["mean"]:.5f} secondary adoptions per initial seed; seed-inclusive cascade totals should not be interpreted as amplification.','secondary_reach_simulation_results.csv')
    add('Amount of actual competition','Exclusive adoption does not imply many contested nodes','Positive-probability susceptible opportunities vs successful simultaneous resolution',f'Full balanced recorded mean ever-both fraction {ce["mean"]:.8f}; actual resolution events/run {cr["mean"]:.6f}. Exposure denominator excludes 50 seeds.','needs wording correction',f'In the balanced full-model recorded-direction simulation, the mean fraction of non-seed nodes ever offered both contagions before adoption is {100*ce["mean"]:.4f}%, and the mean number of actual simultaneous competition-resolution events is {cr["mean"]:.4f} per run; these are model exposure opportunities, not observed Twitter contacts.','competition_exposure_overlap.csv')
    add('Seed-inclusive vs secondary competition outcome','Seed allocation can determine outcome statistic','Signed (C0-C1)/(C0+C1) and original 5% mixed-close rule on totals and secondary counts','Both definitions saved; zero secondary-adoption denominator is NA, not zero or tie. No equivalence margin defined.','needs methodological correction','Competitive outcomes are reported primarily using non-seed Target-0 and Target-1 adoptions; runs with no secondary adoption have undefined outcomes, and seed-inclusive outcomes are retained only as a sensitivity because initial allocation can dominate them.','competition_outcome_definition_comparison.csv')
    add('Spectral-threshold validity','Tiny recurrent core and self-loops may dominate','Exact SCC-block spectra, loop removal, three directions, explicit normalization',f'Largest SCC 17; radius {loop.spectral_radius_weighted:.6f} -> {noloop.spectral_radius_weighted:.6f}; loop-dominated singleton ictcimpact. Diagnostic beta {loop.beta_c_spectral_diagnostic:.6f} -> {noloop.beta_c_spectral_diagnostic:.6f}; all-reversed invariant.','needs wording correction',f'The adjacency spectral radius is dominated by a self-loop on a singleton SCC and decreases from {loop.spectral_radius_weighted:.4f} to {noloop.spectral_radius_weighted:.4f} after loop removal; because the largest SCC has only 17 nodes and epidemic-model assumptions were not validated, the associated critical-beta calculation is a loop-sensitive spectral diagnostic rather than an established epidemic threshold.','spectral_threshold_audit.csv')
    add('Forecasting near-chance interpretation','Accuracy may reflect imbalance and incomparable cohorts','Saved forecasts only, 156 summary means independently reproduced from saved weekly means','Graph-only balanced accuracy .5006–.5091; Persistence .5403; history logistic .5535 and history+structural MLP .5536. No corrected macro-F1 superiority over Persistence established.','needs wording correction','Graph-only weekly models achieve balanced accuracy close to 0.50, whereas Persistence and history-based models show modest signal; historical label information, rather than demonstrated incremental graph-message value, accounts for the strongest saved results, and the original monthly majority-like accuracy must not be presented as strong balanced forecasting.','all_forecasting_models_reconciled.csv')
    add('Aligned GRU definition','Different GRUs conflated','Saved architecture/configuration and cutoff construction compared','Original GRU direct six-feature GRU differs from graph-aligned projected raw-feature GRU. Temporal GCN fixed graph weights, not parameter-evolving EvolveGCN.','needs wording correction','The aligned GRU replaces the Temporal GCN graph aggregation with raw weekly structural feature sequences while retaining its projection, GRU head and evaluation pipeline; it is distinct from the original lagged node-only GRU and is reported separately.','aligned_GRU_definition.txt')
    add('Static vs temporal node universes','Population denominators differ across analyses','Exact/normalized static cohorts, reconstructed endpoint union, weekly prediction eligibility','Original WCC 17,598; normalized WCC 17,636; full event endpoint universe 60,219; weekly prediction uses eligible user-weeks, not every static node.','needs wording correction','The original static diffusion cohort contains 17,598 users, the normalized-matching sensitivity contains 17,636 largest-WCC users, and the full reconstructed temporal network contains 60,219 endpoint accounts; weekly forecasting evaluates an eligible user-week cohort and these denominators are not interchangeable.','forecasting_population_and_information_boundary.csv')
    add('Publication figure corrections','Mixed units, hidden zeros/core, seed-inclusive outcome and missing uncertainty','New figures only; source CSVs, SVG/400-DPI PNG, readable unit/scale/CI labels','Separated population units and weekly panels, visible exact polarity zero, CCDFs without power-law claims, log lollipop bow-tie, net-of-seed competition, panel-specific heatmap scales and forecast references.','needs methodological correction','Corrected figures separate population units and activity/class-share panels, display exact zero polarity, report degree/strength CCDFs without a power-law claim, and use secondary-adoption outcomes and explicitly labelled uncertainty and chance references.','../figures/reviewer_corrected/')
    add('Reproducibility and governance gaps','Environment, assumptions, provenance and hashes incomplete','Protected-file baseline/postcheck, scripts/configuration and output hashes packaged','Raw tweet text not included; original package versions/dataset provenance, validated native reply direction, causal adoption and public licence/DOI unavailable.','needs methodological correction','The correction analyses are accompanied by scripts, fixed random seeds, configuration, software versions, protected-input SHA-256 verification, figure source data and output manifests; the source tweet dataset is excluded and no public repository, DOI, redistribution licence or verified native interaction metadata is claimed.','reproducibility_manifest.csv')
    text=['# Reviewer correction evidence report','', 'Analysis only. Original files/model results preserved; no LaTeX edited. Classifications refer to claims described in the request because the current manuscript itself was not supplied in this project. Numerical statements that depend on matching rule or node universe must retain that qualifier. Canonical outputs below supersede generated interim validation drafts.','']
    for k,r in enumerate(issues,1):
        text.extend([f'## {k}. {r["issue"]}','',f'**Issue:** {r["issue"]}',f'**Reviewer concern:** {r["concern"]}',f'**What was checked:** {r["checked"]}',f'**Result:** {r["result"]}',f'**Current statement classification:** {r["status"]}',f'**Exact recommended replacement:** “{r["sentence"]}”',f'**Supporting output:** `results/tables/{r["file"]}`',''])
    text.extend(['## Remaining limits','', 'Native reply/retweet metadata, true information exposure/adoption, causal edge direction and completeness of the text-supported interaction network cannot be recovered from these inputs. Retweet layer inference rests on 16 events, with only nine orientation-labelled unique pairs; some global-shuffle correlations are undefined. The original manuscript spectral operator/version and dataset acquisition provenance are unavailable. The original TextBlob environment version was not saved; current recomputation exactly matches saved sign counts. No scientifically justified equivalence margin was supplied or invented. Finite 25-step simulations are conditional sensitivity analyses. Full-history orientation is descriptive, not a prior-time causal label. Temporal seed-bootstrap intervals describe seeds in this observed network, not network-population uncertainty. No new forecasting, model training or manuscript editing occurred.'])
    txt('reviewer_correction_master_report.md','\n'.join(text)); csv('reviewer_correction_issue_index.csv',issues)
    terminal=[f'Normalized matching changed main WCC: TRUE (17,598 -> 17,636 users; 29,715 -> 29,825 edges).',f'Pair coverage: {int(o.overlap)}/{int(o.coverage_denominator)} = {o.supplied_pair_coverage*100:.8f}%; reconstructed-pair precision {o.precision_overlap*100:.8f}%.',f'Lexical agreement: {int(lex.agreement_numerator)}/{int(lex.agreement_denominator)} = {lex.agreement_percentage:.8f}% (94.67% rounded).','Typed events: retweet 16; inferred reply 41,024; mention 88,564.','Direction: author -> referenced account is supported by reconstruction; reverse RT is plausible, mention/reply information flow unverified.',f'High-out-strength mean temporal/static ratio: recorded {tp["mean"]:.6f}; layer-aware {tf["mean"]:.6f}; stratum-specific, static sinks excluded.',f'Balanced full recorded: secondary {sec["mean"]:.6f}; amplification {amp["mean"]:.6f}; baseline secondary {baseline["mean"]:.6f}.',f'Contested non-seed fraction: {ce["mean"]:.8f} ({100*ce["mean"]:.6f}%); actual competition resolutions/run {cr["mean"]:.6f}.',f'Spectral radius loop-sensitive {loop.spectral_radius_weighted:.6f} -> {noloop.spectral_radius_weighted:.6f}; reversed invariant; beta diagnostic {loop.beta_c_spectral_diagnostic:.6f} -> {noloop.beta_c_spectral_diagnostic:.6f}.','Forecasting: graph-only near chance; Persistence/history models modestly above chance; practical materiality not established by a predeclared margin.','Claims MUST change: matching invariance, verified information direction, broad retweet/homophily inference, seed-inclusive amplification, strong competition, validated epidemic threshold, conflated GRUs/cohorts, broad graph-forecasting superiority.','Unavailable: native metadata, actual contagion/adoption, original spectral definition/manuscript, acquisition provenance, licence/DOI.','Exact paths for all new results: see results/tables/reproducibility_manifest.csv and reproducibility_package/table_manifest.csv / figure_manifest.csv.']
    txt('reviewer_final_terminal_summary.txt','\n'.join(terminal)); print('\n'.join(terminal),flush=True)

def package():
    import importlib.metadata as md, os
    PKG.mkdir(exist_ok=True)
    scripts=[ROOT/'src/reviewer_correction_analysis.py',ROOT/'src/reviewer_validate_evidence.py']
    dest=PKG/'scripts'; dest.mkdir(exist_ok=True)
    for p in scripts:
        with (dest/p.name).open('xb') as out, p.open('rb') as source: shutil.copyfileobj(source,out)
    for p in sorted((ROOT/'src').glob('reviewer_reference_*.txt')):
        with (dest/p.name).open('xb') as out, p.open('rb') as source: shutil.copyfileobj(source,out)
    # Copied modules continue to refer to project root when run at package/scripts depth.
    txt('README.md','# Reviewer correction reproducibility package\n\nRun from the existing project root with its Python environment. Protected data remain in data/raw and data/processed; data are not distributed here. Canonical analysis outputs are in results/tables and results/figures/reviewer_corrected. Superseded generated validation drafts in results/reviewer_interim_20261009 are retained only for audit.\n\nCommands (each output is exclusive; existing paths fail, so do not rerun in place):\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 src/reviewer_correction_analysis.py reconcile\n.\\.venv\\Scripts\\python.exe -X utf8 src/reviewer_correction_analysis.py layers\n.\\.venv\\Scripts\\python.exe -X utf8 src/reviewer_correction_analysis.py mixing\n.\\.venv\\Scripts\\python.exe -X utf8 src/reviewer_correction_analysis.py temporal\n.\\.venv\\Scripts\\python.exe -X utf8 src/reviewer_correction_analysis.py simulations\n.\\.venv\\Scripts\\python.exe -X utf8 src/reviewer_correction_analysis.py spectral\n.\\.venv\\Scripts\\python.exe -X utf8 src/reviewer_correction_analysis.py forecasting\n.\\.venv\\Scripts\\python.exe -X utf8 src/reviewer_correction_analysis.py figures\n.\\.venv\\Scripts\\python.exe -X utf8 src/reviewer_correction_analysis.py supplements\n.\\.venv\\Scripts\\python.exe -X utf8 src/reviewer_validate_evidence.py\n.\\.venv\\Scripts\\python.exe -X utf8 src/reviewer_correction_analysis.py report\n.\\.venv\\Scripts\\python.exe -X utf8 src/reviewer_correction_analysis.py package\n```\n\nFor fresh reproduction, use a new copy of the input project containing the listed protected inputs and existing forecasting/configuration artifacts, but excluding reviewer outputs, then hash that copy before analysis. Script copies in this package are archival; execute the original new src scripts. Package bootstrap excludes its own manifest/hash ledgers from their self-referential hash contents. Hash verification is independently repeated at package creation. No model is fitted or forecast recomputed. See assumptions_and_limitations.md and individual analysis reports. No raw tweet text is copied. Username-derived metadata require access control; these files are not automatically approved for public distribution.',PKG)
    sys.path.insert(0,str(PKG/'vendor')); versions={name:md.version(name) for name in ['numpy','pandas','scipy','networkx','matplotlib','pillow','textblob']}; versions['python']=platform.python_version()
    txt('software_versions.txt','\n'.join(f'{k}=={v}' for k,v in versions.items()),PKG)
    txt('requirements.txt','\n'.join(f'{k}=={v}' for k,v in versions.items() if k!='python'),PKG)
    txt('REPRODUCTION_BOOTSTRAP.md','# Fresh-copy bootstrap\n\nDo not rerun over these outputs. In a new copy of the original input project, first execute `python -X utf8 src/reviewer_correction_analysis.py protect` before installing dependencies or running analyses. It hashes every original project file except .venv/.git/__pycache__ and fails if a baseline already exists. Input readers verify each protected file against that baseline before use. Run the phase commands in README in order; package creation independently verifies all protected files. Keep existing forecasting artifacts as read-only inputs; do not train anything. Archived copies in package/scripts are retained for review; execute src copies because their root resolves to the existing project. If moving files to another computer, generate a fresh absolute-path baseline there and compare original input byte hashes against the archived input_file_hashes.txt. The seed/permutation/simulation configuration is saved; installed original model package versions are not fabricated. Third-party vendored libraries are optional local runtime support, not project raw data.',PKG)
    txt('hardware_runtime_notes.txt',f'Platform: {platform.platform()}\nProcessor: {platform.processor()}\nLogical CPUs: {os.cpu_count()}\nPython: {sys.version}\nAnalyses ran locally on CPU in the original project .venv. TextBlob dependencies installed only in new reproducibility_package/vendor; original environment files unchanged. Exact wall runtimes for initial API-invoked phases were not captured and are not invented. Future CLI invocations save exclusive phase runtime JSONs. Full-hash verification includes 4607 original project files excluding .venv/.git/__pycache__. Temporal graph case count 9, four seed strata, up to 500 seeds each; repeated seeds cached per graph. Simulations 8 configurations x 3 directions x 2000 paired runs = 48000 runs, finite horizon 25 steps. Permutations 3 layers x 2 nulls x 2000 = 12000. Bootstrap 2000 seed resamples per temporal summary. No GPU/model fitting. Source file paths/CPU metadata may disclose local details and should be reviewed before publication.',PKG)
    txt('random_seeds.txt','New analyses: base RNG seed 42. Permutations/bootstrap use numpy.default_rng(42). Simulated seed sets use default_rng(42+zero_based_run). Common direction/control transition uniforms use domain-separated uint64 keyed SplitMix-style hashing over base42, run, timestep, node and channel. Saved forecasting seeds 42..51 are reported without retraining; original simulator notebook seed20260905 is reference only. Stable lexical ties for graph ranks/components.',PKG)
    config=dict(seed=42,permutations_per_layer_null=2000,temporal_seeds_per_stratum=500,bootstrap_draws=2000,simulation_runs_per_cell=2000,simulation_max_steps=25,recovery=.25,primary_layer_aware_direction='reverse retweet; retain mention and inferred reply',username_normalization='strip whitespace; remove ONE leading @; casefold',writes='exclusive new outputs; no manuscript/model/raw edits',equivalence_margin=None)
    txt('reviewer_configuration.json',json.dumps(config,indent=2),PKG)
    for source,name in [('typed_direction_assumptions.txt','reconstruction_assumptions.txt'),('reviewer_simulation_assumptions.txt','simulator_settings.txt'),('layer_orientation_assumptions.txt','orientation_assumptions.txt'),('forecasting_consistency_report.txt','model_information_boundary.txt'),('reviewer_analysis_validation_notes.txt','validation_notes.txt')]: txt(name,(TABLE/source).read_text(encoding='utf-8'),PKG)
    txt('username_normalization_rule.txt','Trim leading/trailing whitespace; remove exactly one leading @ if present; casefold. Do not strip arbitrary multiple @ characters. Exact matching retains literal IDs. Deterministic display uses most frequent trimmed spelling across raw endpoints and authors, lexical tie break. Saved reconstruction had lstrip(@); its source/limitations are retained, and this new pass applies the explicitly requested one-at rule. Key collisions aggregate ratios by saved class counts, not unweighted mean of ratios.',PKG)
    configs={}
    for p in ['results/models/node_only_neural_runs/manifest.json','results/models/weekly_graph_forecasting_runs/manifest.json','results/models/history_graph_runs/config.json']:
        d=json.loads((ROOT/p).read_text(encoding='utf-8')); configs[p]=d.get('config',d)
    txt('saved_model_configuration_summary.json',json.dumps(configs,indent=2),PKG)
    hp=[]
    for p in ['node_only_training_hyperparameters.csv','graph_model_hyperparameters.csv','graph_direction_ablation_hyperparameters.csv','history_graph_hyperparameters.csv']:
        d=read('results/tables/'+p); cols=[c for c in ['Model','Direction','Control','hidden','dropout','lr','weight_decay','heads','Threshold','Selected_C','Selected_Grid_ID'] if c in d.columns]
        for r in d[cols].drop_duplicates().to_dict('records'): hp.append(dict(source=p,**r))
    csv('model_hyperparameter_summary.csv',hp,PKG)
    txt('DATA_NOT_INCLUDED.md','# Source data not included\n\nNo raw tweets or tweet text are included in this package. Reproduction requires the project owner’s authorized copies of data/raw/master_tweets.csv and network_edges.csv, plus the protected processed/model-output inputs listed in input_file_hashes.txt. The project inputs do not establish a public download URL, dataset DOI or collection licence; request provenance and lawful access instructions from the project owner, rather than assuming availability or using an invented source. Native tweet IDs/reply metadata are not recoverable from the supplied schema.',PKG)
    txt('ethics_and_data_governance.md','# Ethics and governance\n\nThis pass analyzes existing political Twitter records; Target labels are historical dataset categories, not verified political affiliation, mental state or ground truth. TextBlob is an English lexicon method with limitations for multilingual, ironic and quoted political text. Handles and per-user derived data remain potentially identifying; analysis does not anonymize them. Raw text is excluded from this package. Keep access restricted, follow source permissions/platform terms and institutional review requirements applicable to the actual collection, and review any release for identification risk. No consent, ethics approval, collection licence, causal transmission or current platform status is asserted. Simulated contagion is a hypothetical network process; reported exposure is an instrumented model opportunity. No messages were sent and no account actions were performed.',PKG)
    txt('LICENSE_PLACEHOLDER.md','# Licence not selected\n\nNo project redistribution licence was found or chosen in this pass. The owner must establish rights for code, derived identifying data and third-party source inputs before public release. Third-party libraries retain their own bundled licences; this placeholder does not grant rights over tweets or usernames.',PKG)
    txt('assumptions_and_limitations.md','# Analysis assumptions and limits\n\nRead reconstruction_assumptions.txt, username_normalization_rule.txt, simulator_settings.txt, orientation_assumptions.txt, model_information_boundary.txt and validation_notes.txt together with the canonical per-analysis reports. The reviewer master report gives replacement sentences with exact supporting outputs. No uncertainty interval upgrades an unverified causal interpretation. No equivalence margin was invented. Historical orientation spans the saved full period and is descriptive. Temporal reach starts seeds at first layer timestamp; equal-time closure is permitted. Hop depths are foremost-algorithm witnesses, not longest paths. Top-centrality/bootstrap summaries are descriptive of the selected seeds. The 16-event RT layer is too sparse for broad claims, and nulls with undefined statistics are not assigned small p-values. The original spectral formula and historical environment/dataset provenance are unavailable. No models were fitted.',PKG)
    baseline=json.loads((ROOT/'reviewer_protected_hashes_20261009.json').read_text(encoding='utf-8-sig'))
    txt('input_file_hashes.txt','\n'.join(f'{r["sha256"].lower()}  {Path(r["path"]).relative_to(ROOT).as_posix()}' for r in baseline),PKG)
    figure_rows=[]
    for p in sorted(FIG.glob('*.svg')): figure_rows.append(dict(figure=p.relative_to(ROOT).as_posix(),PNG=p.with_suffix('.png').relative_to(ROOT).as_posix(),source_csv=p.with_name(p.stem+'_source.csv').relative_to(ROOT).as_posix(),producing_script='src/reviewer_correction_analysis.py',primary=p.stem!='typed_layer_sizes',DPI=400))
    csv('figure_manifest.csv',figure_rows,PKG)
    protected={os.path.normcase(str(Path(r['path']))):r for r in baseline}
    table_rows=[dict(file=p.relative_to(ROOT).as_posix(),purpose=p.stem.replace('_',' ')) for p in sorted(TABLE.glob('*')) if p.is_file() and os.path.normcase(str(p)) not in protected]
    csv('table_manifest.csv',table_rows,PKG)
    verify(write=True)
    # Inventory all new project files, including dependency support and retained audit drafts.
    outputs=[]
    for directory,dirs,files in os.walk(ROOT):
        dirs[:]=[x for x in dirs if x not in ['.venv','.git','__pycache__']]
        for name in files:
            p=Path(directory)/name
            if os.path.normcase(str(p)) in protected: continue
            relative=p.relative_to(ROOT).as_posix()
            if relative in ['results/tables/reproducibility_manifest.csv','reproducibility_package/derived_output_hashes.txt']: continue
            purpose='superseded generated draft; audit only' if '/reviewer_interim_' in relative else 'third-party dependency support' if '/vendor/' in relative else p.stem.replace('_',' ')
            producer='pip install --target reproducibility_package/vendor textblob' if '/vendor/' in relative else 'src/reviewer_validate_evidence.py' if p.name in ['reviewer_evidence_validation.csv','temporal_hop_interpretation_clarification.txt'] else 'src/reviewer_correction_analysis.py' if not p.name.startswith('reviewer_reference_') else 'notebook code extraction (read-only reference)'
            outputs.append(dict(file_path=relative,producing_script=producer,protected_inputs='see reproducibility_package/input_file_hashes.txt (conservative all-input protection)' if '/vendor/' not in relative else 'none',SHA256=sha(p),purpose=purpose))
    manifest=csv('reproducibility_manifest.csv',outputs)
    # A file cannot contain its own digest. Hash ledger includes manifest, excludes itself.
    txt('derived_output_hashes.txt','# SHA-256 ledger excludes this ledger itself to avoid self-reference.\n'+'\n'.join(f'{r["SHA256"]}  {r["file_path"]}' for r in outputs)+f'\n{sha(manifest)}  {manifest.relative_to(ROOT).as_posix()}',PKG)
    print('Exact canonical result paths:',flush=True)
    for r in outputs:
        if r['file_path'].startswith(('results/tables/','results/figures/reviewer_corrected/')): print(str(ROOT/r['file_path']),flush=True)
    print(f'New output inventory: {len(outputs)} files; protected original files: {len(baseline)}. NO EXISTING FILES OVERWRITTEN: TRUE',flush=True)

def main():
    phases=dict(protect=protect,reconcile=reconcile,layers=layers,mixing=mixing,temporal=temporal,simulations=simulations,spectral=spectral,forecasting=forecasting,figures=figures,supplements=supplements,report=master_report,package=package)
    if len(sys.argv)!=2 or sys.argv[1] not in phases: raise SystemExit('Specify one phase: '+', '.join(phases))
    phase=sys.argv[1]
    if phase!='protect' and not (ROOT/'reviewer_protected_hashes_20261009.json').exists(): raise RuntimeError('Run protect first, before any analysis')
    phases[phase]()
    if phase!='package': txt('reviewer_phase_runtime_'+phase+'.json',json.dumps(dict(phase=phase,elapsed_seconds=time.perf_counter()-START,platform=platform.platform()),indent=2))

if __name__=='__main__': main()
