"""New weekly causal-cutoff graph experiments; never overwrite prior outputs.

Exact one-hop weighted GCN and mean GraphSAGE, sampled weighted GAT, and
weekly GCN -> projected GRU node states (not parameter-evolving EvolveGCN).
Independent CPU processes execute frozen experiment jobs, not agent tasks.
"""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import random
import time
import os

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from build_orientation_baselines import FEATURES, METRICS, score, sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/models/weekly_graph_forecasting_runs'
SEEDS = list(range(42,52))
PRIMARY = ['Static GCN','GraphSAGE','GAT','Temporal GCN']
FANOUT = 16
BATCH = 1024
EPOCHS = 30
PATIENCE = 4
GRID = [
    dict(hidden=16,dropout=0.,lr=.001,weight_decay=0.,heads=2),
    dict(hidden=32,dropout=.2,lr=.001,weight_decay=.0001,heads=4),
    dict(hidden=64,dropout=.5,lr=.0003,weight_decay=.0001,heads=4),
]
CONFIG = dict(seeds=SEEDS,grid=GRID,fanout=FANOUT,batch=BATCH,max_epochs=EPOCHS,patience=PATIENCE,
              threshold=.5,graph_layers=1,temporal='weekly GCN -> projection ReLU -> GRU; fixed GCN weights',
              features=FEATURES,torch=torch.__version__,workers=2)


class Dense(nn.Module):
    def __init__(self,cfg,width=6):
        super().__init__()
        h,d=cfg['hidden'],cfg['dropout']
        self.net=nn.Sequential(nn.Linear(width,h),nn.ReLU(),nn.Dropout(d),nn.Linear(h,h),nn.ReLU(),nn.Dropout(d),nn.Linear(h,1))
    def forward(self,x,lengths):
        return self.net(x).squeeze(-1)


class WeightedGAT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.h,self.heads=cfg['hidden'],cfg['heads']
        self.project=nn.Linear(6,self.h*self.heads,bias=False)
        self.a_source=nn.Parameter(torch.empty(self.heads,self.h))
        self.a_target=nn.Parameter(torch.empty(self.heads,self.h))
        nn.init.xavier_uniform_(self.a_source)
        nn.init.xavier_uniform_(self.a_target)
        self.drop=nn.Dropout(cfg['dropout'])
        self.net=nn.Sequential(nn.Linear(self.h*self.heads,self.h),nn.ReLU(),nn.Dropout(cfg['dropout']),nn.Linear(self.h,1))
    def forward(self,x,lengths):
        # Slot zero is the receiving node; others are incoming neighbors.
        projected=self.project(x[:,:,:6]).reshape(len(x),x.shape[1],self.heads,self.h)
        source=(projected*self.a_source).sum(-1)
        target=(projected[:,0]*self.a_target).sum(-1).unsqueeze(1)
        logits=nn.functional.leaky_relu(source+target,.2)+x[:,:,6].unsqueeze(-1)
        valid=torch.arange(x.shape[1])[None,:]<lengths[:,None]
        logits=logits.masked_fill(~valid.unsqueeze(-1),-torch.inf)
        attention=torch.softmax(logits,dim=1)
        aggregated=(self.drop(attention).unsqueeze(-1)*projected).sum(1)
        return self.net(nn.functional.elu(aggregated.reshape(len(x),-1))).squeeze(-1)


class ProjectedGRU(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        h=cfg['hidden']
        self.project=nn.Sequential(nn.Linear(6,h),nn.ReLU())
        self.gru=nn.GRU(h,h,batch_first=True)
        self.drop=nn.Dropout(cfg['dropout'])
        self.out=nn.Linear(h,1)
    def forward(self,x,lengths):
        # Trim padding to this batch's real maximum, never using future rows.
        steps=max(1,int(lengths.max()))
        z=self.project(x[:,:steps])
        output,_=self.gru(z)
        h=output[torch.arange(len(x)),(lengths-1).clamp(min=0)]*lengths.gt(0).unsqueeze(1)
        return self.out(self.drop(h)).squeeze(-1)


def make_model(name,cfg):
    if name=='GraphSAGE':
        return Dense(cfg,12)
    if name=='GAT':
        return WeightedGAT(cfg)
    if name in ['Temporal GCN','GRU (aligned)']:
        return ProjectedGRU(cfg)
    return Dense(cfg,6)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def p1_scores(model,x,lengths):
    model.eval()
    values=[]
    with torch.no_grad():
        for start in range(0,len(x),BATCH):
            values.append(torch.sigmoid(model(x[start:start+BATCH],lengths[start:start+BATCH])).numpy())
    return np.concatenate(values)


def fit_model(name,cfg,seed,train,validation,weights):
    seed_all(seed)
    net=make_model(name,cfg)
    opt=torch.optim.Adam(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    xt,lt,yt=train; xv,lv,yv=validation
    best,stalled,best_epoch,state=-1.,0,0,None
    started=time.perf_counter()
    for epoch in range(1,EPOCHS+1):
        net.train()
        for start in range(0,len(xt),BATCH):
            xb,lb,yb=xt[start:start+BATCH],lt[start:start+BATCH],yt[start:start+BATCH]
            opt.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(net(xb,lb),yb,reduction='none')
            loss=(loss*torch.where(yb.eq(0),weights[0],weights[1])).mean()
            assert torch.isfinite(loss)
            loss.backward(); opt.step()
        p=p1_scores(net,xv,lv)
        value=f1_score(yv.numpy(),p>=.5,labels=[0,1],average='macro',zero_division=0)
        if value>best+1e-12:
            best,stalled,best_epoch,state=float(value),0,epoch,copy.deepcopy(net.state_dict())
        else:
            stalled+=1
        if stalled>=PATIENCE:
            break
    net.load_state_dict(state)
    return net,dict(state=state,Validation_Macro_F1=best,Best_Epoch=best_epoch,Epochs_Run=epoch,Seconds=time.perf_counter()-started)


def save_pt(path,value):
    with path.open('xb') as f:
        torch.save(value,f)


def load_pt(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def normalized_gcn(a,x):
    # A rows receive messages from columns. Symmetrization gives reversal invariance.
    sym=(a+a.T)*.5+sparse.eye(a.shape[0],format='csr')
    degree=np.asarray(sym.sum(axis=1)).ravel()
    d=sparse.diags(1/np.sqrt(degree))
    s=d@sym@d
    return np.asarray(s@x),np.asarray(s.sum(axis=1)).ravel()


def mean_messages(a,x):
    total=np.asarray(a.sum(axis=1)).ravel()
    inv=np.divide(1.,total,out=np.zeros_like(total),where=total>0)
    mean=sparse.diags(inv)@a
    return np.asarray(mean@x),(total>0).astype(float)


def rewired(a,seed):
    coo=a.tocoo()
    repeats=coo.data.astype(int)
    assert np.array_equal(repeats,coo.data)
    src=np.repeat(coo.col,repeats); dst=np.repeat(coo.row,repeats)
    rng=np.random.default_rng(seed)
    dst=rng.permutation(dst)
    out=sparse.coo_matrix((np.ones(len(src)),(dst,src)),shape=a.shape).tocsr()
    assert np.array_equal(np.asarray(a.sum(0)),np.asarray(out.sum(0)))
    assert np.array_equal(np.asarray(a.sum(1)),np.asarray(out.sum(1)))
    return out


def gat_rows(a,x,node_indices,week,tag):
    inputs=np.zeros((len(node_indices),FANOUT+1,7),np.float32)
    lengths=np.zeros(len(node_indices),np.int64)
    for j,node in enumerate(node_indices):
        start,end=a.indptr[node:node+2]
        ids=a.indices[start:end]; weights=a.data[start:end]
        self_weight=1+float(weights[ids==node].sum())
        nonself=ids!=node; ids,weights=ids[nonself],weights[nonself]
        if len(ids)>FANOUT:
            digest=hashlib.sha256(f'42|{week}|{tag}|{node}'.encode()).digest()
            rng=np.random.default_rng(int.from_bytes(digest[:8],'little'))
            choice=np.sort(rng.choice(len(ids),FANOUT,replace=False))
            ids,weights=ids[choice],weights[choice]
        ids=np.r_[node,ids]; weights=np.r_[self_weight,weights]
        count=len(ids); lengths[j]=count
        inputs[j,:count,:6]=x[ids]
        inputs[j,:count,6]=np.log(weights)
    return inputs,lengths


def prepare():
    cache=RUN/'inputs.npz'
    if cache.exists():
        return
    data=pd.read_csv(ROOT/'data/processed/weekly_leakage_controlled_prediction_dataset.csv',dtype={'User':str}).sort_values(['Week_Start','User']).reset_index(drop=True)
    f=pd.read_csv(ROOT/'data/processed/weekly_temporal_node_features.csv',dtype={'User':str})
    f['User']=f.User.str.casefold()
    f=f.sort_values(['Week_Start','User']).reset_index(drop=True)
    assert not f.duplicated(['Week_Start','User']).any()
    assert not data.duplicated(['Week_Start','User']).any()
    assert pd.to_datetime(f.Feature_Window_End_Exclusive,utc=True).equals(pd.to_datetime(f.Week_Start,utc=True))
    assert pd.to_datetime(data.Feature_Last_Interaction_UTC,utc=True).lt(pd.to_datetime(data.Week_Start,utc=True)).all()
    n=len(data); nf=len(f)
    raw=f[FEATURES].to_numpy(float)
    static=np.zeros((nf,6)); mass=np.zeros(nf)
    weekly=np.zeros((nf,6)); weekly_mass=np.zeros(nf)
    null_static=np.zeros((nf,6)); null_mass=np.zeros(nf)
    null_weekly=np.zeros((nf,6)); null_weekly_mass=np.zeros(nf)
    sage={k:np.zeros((n,6)) for k in ['forward','reverse','rewired']}
    sage_mass={k:np.zeros(n) for k in sage}
    gat={k:np.zeros((n,FANOUT+1,7),np.float32) for k in sage}
    gat_lengths={k:np.zeros(n,np.int64) for k in sage}
    cumulative=defaultdict(int)
    mapping_rows=[]; audits=[]
    groups=dict(tuple(data.groupby('Week_Start',sort=False)))
    snapshot_files=sorted((ROOT/'data/processed/temporal_graphs').glob('edges_*.csv'))
    assert len(snapshot_files)==25
    for number,(cutoff,frame) in enumerate(f.groupby('Week_Start',sort=True)):
        cutoff_date=pd.Timestamp(cutoff,tz='UTC')
        completed_week=(cutoff_date-pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        assert pd.Timestamp(completed_week,tz='UTC')+pd.Timedelta(days=7)<=cutoff_date
        e=pd.read_csv(ROOT/f'data/processed/temporal_graphs/edges_{completed_week}.csv',dtype={'Source':str,'Target':str})
        assert list(e)==['Source','Target','Weight']
        assert (e.Weight>0).all() and np.all(e.Weight==e.Weight.astype(int))
        e['Source']=e.Source.str.casefold(); e['Target']=e.Target.str.casefold()
        for s,t,w in e.itertuples(index=False,name=None):
            cumulative[(s,t)]+=int(w)
        users=frame.User.tolist(); index={u:i for i,u in enumerate(users)}
        known={u for pair in cumulative for u in pair}
        assert set(users)==known, 'Future or missing user in historical node mapping'
        assert set(e.Source)|set(e.Target)<=known
        local=raw[frame.index]
        def adjacency(pairs):
            src=[index[s] for s,t,w in pairs]; dst=[index[t] for s,t,w in pairs]; weights=[w for s,t,w in pairs]
            return sparse.coo_matrix((weights,(dst,src)),shape=(len(users),len(users)),dtype=float).tocsr()
        a=adjacency([(s,t,w) for (s,t),w in cumulative.items()])
        aw=adjacency(list(e.itertuples(index=False,name=None)))
        ar=rewired(a,42000+number); awr=rewired(aw,43000+number)
        static[frame.index],mass[frame.index]=normalized_gcn(a,local)
        weekly[frame.index],weekly_mass[frame.index]=normalized_gcn(aw,local)
        null_static[frame.index],null_mass[frame.index]=normalized_gcn(ar,local)
        null_weekly[frame.index],null_weekly_mass[frame.index]=normalized_gcn(awr,local)
        reverse_values,reverse_mass=normalized_gcn(a.T,local)
        assert np.allclose(reverse_values,static[frame.index]) and np.allclose(reverse_mass,mass[frame.index])
        target=groups.get(cutoff,data.iloc[:0])
        node_ids=np.array([index[u] for u in target.User],dtype=int)
        for direction,adj in [('forward',a),('reverse',a.T.tocsr()),('rewired',ar)]:
            means,m=mean_messages(adj,local)
            sage[direction][target.index]=means[node_ids]
            sage_mass[direction][target.index]=m[node_ids]
            g,gl=gat_rows(adj,local,node_ids,cutoff,direction)
            gat[direction][target.index]=g
            gat_lengths[direction][target.index]=gl
        mapping_rows.extend((cutoff,u,i) for i,u in enumerate(users))
        audits.append(dict(Week_Start=cutoff,Latest_Completed_Graph_Week=completed_week,Known_Nodes=len(users),
                           Historical_Edges=a.nnz,Weekly_Edges=aw.nnz,Historical_Events=int(a.sum()),
                           Evaluated_Users=len(target),Symmetric_GCN_Reversal_Identical=True,
                           Rewired_In_Out_Strength_Preserved=True))
        print(f'Prepared {cutoff}: known nodes={len(users):,}, history events={int(a.sum()):,}',flush=True)
    lookup=pd.Series(f.index.to_numpy(),index=pd.MultiIndex.from_frame(f[['Week_Start','User']]))
    data_index=lookup.loc[pd.MultiIndex.from_frame(data[['Week_Start','User']])].to_numpy()
    max_length=25
    sequence=np.full((n,max_length),-1,np.int64); lengths=np.zeros(n,np.int64)
    histories={u:(g.Week_Start.to_numpy(),g.index.to_numpy()) for u,g in f.groupby('User',sort=False)}
    for i,row in enumerate(data.itertuples(index=False)):
        dates,ids=histories[row.User]
        count=np.searchsorted(dates,row.Week_Start,side='right')
        sequence[i,:count]=ids[:count]; lengths[i]=count
        assert count>0 and (dates[:count]<=row.Week_Start).all()
        assert dates[count-1]==row.Week_Start
    values={'week':data.Week_Start.to_numpy(dtype='U10'),'user':data.User.to_numpy(dtype='U15'),
            'y':data.Orientation_Label.to_numpy(np.int64),'x':data[FEATURES].to_numpy(float),
            'static':static[data_index],'static_mass':mass[data_index],
            'null_static':null_static[data_index],'null_static_mass':null_mass[data_index],
            'raw_seq':raw[np.maximum(sequence,0)],'weekly_seq':weekly[np.maximum(sequence,0)],
            'weekly_mass':weekly_mass[np.maximum(sequence,0)],
            'null_weekly_seq':null_weekly[np.maximum(sequence,0)],'null_weekly_mass':null_weekly_mass[np.maximum(sequence,0)],
            'sequence_mask':sequence>=0,'lengths':lengths}
    for direction in sage:
        values['sage_'+direction]=sage[direction]; values['sage_mass_'+direction]=sage_mass[direction]
        values['gat_'+direction]=gat[direction]; values['gat_lengths_'+direction]=gat_lengths[direction]
    with cache.open('xb') as stream:
        np.savez(stream,**values)
    for name,frame in [('graph_forecasting_cutoff_audit.csv',pd.DataFrame(audits)),
                      ('weekly_graph_learning_node_indices.csv',pd.DataFrame(mapping_rows,columns=['Week_Start','User','Node_Index']))]:
        directory='results/tables' if 'audit' in name else 'data/processed'
        with (ROOT/directory/name).open('x',encoding='utf-8',newline='') as stream:
            frame.to_csv(stream,index=False)


def design(cache,name,direction,control,mean,scale):
    x=(cache['x']-mean)/scale
    lengths=np.ones(len(x),np.int64)
    if name=='Static GCN':
        if control=='feature_only':
            out=x
        else:
            key='null_static' if control=='rewired' else 'static'
            out=(cache[key]-cache[key+'_mass'][:,None]*mean)/scale
    elif name=='GraphSAGE':
        if control=='feature_only':
            neighbor=x
        else:
            tag='rewired' if control=='rewired' else direction
            neighbor=(cache['sage_'+tag]-cache['sage_mass_'+tag][:,None]*mean)/scale
        out=np.concatenate([x,neighbor],axis=1)
    elif name=='GAT':
        if control=='feature_only':
            out=np.zeros((len(x),FANOUT+1,7),np.float32)
            out[:,0,:6]=x
        else:
            tag='rewired' if control=='rewired' else direction
            out=cache['gat_'+tag].copy()
            lengths=cache['gat_lengths_'+tag]
            out[:,:,:6]=(out[:,:,:6]-mean)/scale
            mask=np.arange(FANOUT+1)[None,:]<lengths[:,None]
            out[~mask]=0
    else:
        lengths=cache['lengths']
        if control=='feature_only' or name=='GRU (aligned)':
            out=(cache['raw_seq']-mean)/scale
        else:
            key='null_weekly' if control=='rewired' else 'weekly'
            out=(cache[key+'_seq']-cache[key+'_mass'][:,:,None]*mean)/scale
        out[~cache['sequence_mask']]=0
    assert np.isfinite(out).all()
    return out.astype(np.float32),lengths


def execute_job(origin,name,direction,extra_direction_controls=False):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cache=np.load(RUN/'inputs.npz',allow_pickle=False)
    weeks=cache['week']; y=cache['y']
    ids=[np.flatnonzero((weeks>=origin['Train_First_Week'])&(weeks<=origin['Train_Last_Week'])),
         np.flatnonzero(weeks==origin['Validation_Week']),np.flatnonzero(weeks==origin['Test_Week'])]
    train_ids,val_ids,test_ids=ids
    assert len(train_ids)==origin['Train_Users'] and len(val_ids)==origin['Validation_Users'] and len(test_ids)==origin['Test_Users']
    assert max(weeks[train_ids])<origin['Validation_Week']<origin['Test_Week']
    assert len(np.unique(weeks[train_ids]))==origin['Training_Weeks']
    scaler=StandardScaler().fit(cache['x'][train_ids])
    assert scaler.n_samples_seen_==len(train_ids)
    assert np.allclose(scaler.mean_,cache['x'][train_ids].mean(0))
    counts=np.bincount(y[train_ids],minlength=2)
    weights=len(train_ids)/(2*counts)
    assert (counts>0).all()
    prefix=f'{origin["Test_Week"]}_{name.replace(" ","_")}_{direction}'
    if extra_direction_controls:
        # Separate control jobs may run concurrently with the primary queue.
        # They ONLY replay finished validation choices; never race to tune.
        required=[RUN/f'{prefix}_tune_{i}.pt' for i in range(len(GRID))]
        required += [RUN/f'{prefix}_primary_seed_{seed}.pt' for seed in SEEDS]
        deadline=time.monotonic()+7200
        while not all(p.exists() for p in required):
            if time.monotonic()>deadline:
                raise TimeoutError('Primary reverse-direction trials did not finish')
            time.sleep(1)
        time.sleep(.5)
    primary='feature_only' if name=='GRU (aligned)' else 'primary'
    def split(out,l,idx):
        return torch.from_numpy(out[idx]),torch.from_numpy(l[idx]),torch.from_numpy(y[idx].astype(np.float32))
    out,l=design(cache,name,direction,primary,scaler.mean_,scaler.scale_)
    tr,va=split(out,l,train_ids),split(out,l,val_ids)
    candidates=[]; tune_rows=[]
    for grid_id,cfg in enumerate(GRID):
        path=RUN/f'{prefix}_tune_{grid_id}.pt'
        if path.exists():
            fitted=load_pt(path)
        else:
            assert not extra_direction_controls, 'Control attempted fresh hyperparameter tuning'
            _,fitted=fit_model(name,cfg,42,tr,va,weights)
            save_pt(path,fitted)
        candidates.append((fitted['Validation_Macro_F1'],grid_id))
        tune_rows.append(dict(Week_Start=origin['Test_Week'],Model=name,Direction=direction,Grid_ID=grid_id,**cfg,
                              **{k:v for k,v in fitted.items() if k!='state'}))
    _,selected=sorted(candidates,key=lambda v:(-v[0],v[1]))[0]
    cfg=GRID[selected]
    metric_rows=[]; parameter_rows=[]
    controls=['feature_only','rewired'] if extra_direction_controls else [primary]
    if name in PRIMARY and direction=='forward' and not extra_direction_controls:
        # Run both controls for ALL primary models before selecting a best model.
        controls+=['feature_only','rewired']
    for control in controls:
        out,l=design(cache,name,direction,control,scaler.mean_,scaler.scale_)
        tr,va,te=[split(out,l,i) for i in ids]
        for seed in SEEDS:
            path=RUN/f'{prefix}_{control}_seed_{seed}.pt'
            if path.exists():
                result=load_pt(path)
                assert result['selected_grid_id']==selected and result['seed']==seed
                fitted,result_metrics=result['fit'],result['metrics']
            else:
                net,fitted=fit_model(name,cfg,seed,tr,va,weights)
                p1=p1_scores(net,te[0],te[1])
                result_metrics=score(y[test_ids],(p1>=.5).astype(int),p1)
                save_pt(path,dict(fit=fitted,metrics=result_metrics,p1=p1,test_users=cache['user'][test_ids],
                                  seed=seed,selected_grid_id=selected))
            common=dict(Week_Start=origin['Test_Week'],Model=name,Direction=direction,Control=control,Seed=seed,Test_Users=len(test_ids))
            metric_rows.append(dict(**common,**result_metrics,Test_Class_0_Prevalence=float(np.mean(y[test_ids]==0))))
            parameter_rows.append(dict(**common,Train_First_Week=origin['Train_First_Week'],Train_Last_Week=origin['Train_Last_Week'],
                                       Validation_Week=origin['Validation_Week'],Train_Users=len(train_ids),Validation_Users=len(val_ids),
                                       Selected_Grid_ID=selected,**cfg,Threshold=.5,Class_0_Weight=weights[0],Class_1_Weight=weights[1],
                                       Scaler_Fit_Rows=int(scaler.n_samples_seen_),Scaler_Mean=json.dumps(scaler.mean_.tolist()),Scaler_Scale=json.dumps(scaler.scale_.tolist()),
                                       Latest_Test_Graph_Week=(pd.Timestamp(origin['Test_Week'])-pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
                                       **{k:v for k,v in fitted.items() if k!='state'}))
        print(f'{origin["Test_Week"]} {name} {direction} {control}: all ten seeds completed',flush=True)
    cache.close()
    return metric_rows,parameter_rows,tune_rows


def main():
    final=ROOT/'results/tables/graph_models_metrics_by_seed_week.csv'
    if final.exists():
        raise FileExistsError('Existing final graph outputs; no writes')
    expected=['graph_models_summary.csv','graph_model_comparisons.csv','graph_model_hyperparameters.csv',
              'graph_direction_control.csv','graph_ablation_results.csv','graph_models_report.txt']
    assert not any((ROOT/'results/tables'/name).exists() for name in expected)
    manifest_path=RUN/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        assert manifest['config']==CONFIG
        assert all(sha256(ROOT/p)==h for p,h in manifest['protected_hashes'].items())
    else:
        protected=[p for directory in ['data/raw','data/processed','results'] for p in (ROOT/directory).rglob('*') if p.is_file() and RUN not in p.parents]
        hashes={str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(protected))}
        RUN.mkdir(parents=True,exist_ok=True)
        manifest=dict(config=CONFIG,protected_hashes=hashes)
        with manifest_path.open('x',encoding='utf-8') as stream:
            json.dump(manifest,stream,indent=2)
    prepare()
    protocol=pd.read_csv(ROOT/'results/tables/rolling_origin_baseline_protocol.csv')
    assert len(protocol)==15 and protocol.Test_Week.is_monotonic_increasing
    jobs=[]
    for origin in protocol.to_dict('records'):
        for name in PRIMARY:
            jobs.append((origin,name,'forward'))
        for name in ['GraphSAGE','GAT']:
            jobs.append((origin,name,'reverse'))
        jobs.append((origin,'GRU (aligned)','forward'))
    rows,params,tuning=[],[],[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(execute_job,*job):job for job in jobs}
        for count,future in enumerate(as_completed(pending),1):
            m,h,g=future.result()
            rows+=m; params+=h; tuning+=g
            print(f'COMPLETED JOBS {count}/{len(jobs)}',flush=True)
    records=pd.DataFrame(rows).sort_values(['Week_Start','Model','Direction','Control','Seed'])
    assert len(records)==2250
    assert not records.duplicated(['Week_Start','Model','Direction','Control','Seed']).any()
    original=records[records.Model.isin(PRIMARY)&records.Direction.eq('forward')&records.Control.eq('primary')]
    assert len(original)==600
    assert all(sha256(ROOT/p)==h for p,h in manifest['protected_hashes'].items())
    for name,frame in [('graph_models_metrics_by_seed_week.csv',records),('graph_model_hyperparameters.csv',pd.DataFrame(params)),('graph_models_validation_grid.csv',pd.DataFrame(tuning))]:
        with (ROOT/'results/tables'/name).open('x',encoding='utf-8',newline='') as stream:
            frame.to_csv(stream,index=False,na_rep='NA')
    print('All 600 primary graph fits, 300 reversed fits, 150 aligned GRU fits and 1200 matched/null control fits complete.',flush=True)


if __name__=='__main__':
    main()
