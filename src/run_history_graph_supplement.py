"""New causal-history supplement with exact matched self-only/GAT backbone."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy,hashlib,json,time,random
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import torch
from torch import nn
from build_orientation_baselines import FEATURES,METRICS,score,sha256
from weekly_graph_forecasting import GRID,SEEDS,seed_all,p1_scores,save_pt,load_pt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/models/history_graph_runs'
HISTORY=['Previous_Label','Previous_Label_Available','Past_Class0_Rate','Past_Class1_Rate','Past_Label_Count','Weeks_Since_Last_Label']
INPUTS=FEATURES+HISTORY
MODELS=['History Logistic','History + Structural MLP','History + Reversed GAT','History + Forward GAT']
NEURAL=MODELS[1:]
BATCH=1024
MAX_EPOCHS=30
PATIENCE=4
C_GRID=[.01,.1,1.,10.,100.]
CONFIG=dict(features=INPUTS,seeds=SEEDS,grid=GRID,selection='common MLP/reverse-GAT mean validation macro-F1, seed42',
            lr_grid=C_GRID,threshold=.5,max_epochs=MAX_EPOCHS,patience=PATIENCE,fanout=16,workers=3,
            missing_rate='training target-label prevalence',recency_cap='max training observed label recency + 1',
            sensitivity='held-out evaluation subset only; no refit')


class HistoryGAT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.h,self.heads=cfg['hidden'],cfg['heads']
        self.project=nn.Linear(12,self.h*self.heads,bias=False)
        self.a_source=nn.Parameter(torch.empty(self.heads,self.h))
        self.a_target=nn.Parameter(torch.empty(self.heads,self.h))
        nn.init.xavier_uniform_(self.a_source);nn.init.xavier_uniform_(self.a_target)
        self.drop=nn.Dropout(cfg['dropout'])
        self.net=nn.Sequential(nn.Linear(self.h*self.heads,self.h),nn.ReLU(),nn.Dropout(cfg['dropout']),nn.Linear(self.h,1))
    def forward(self,x,lengths):
        z=self.project(x[:,:,:12]).reshape(len(x),x.shape[1],self.heads,self.h)
        logits=nn.functional.leaky_relu((z*self.a_source).sum(-1)+(z[:,0]*self.a_target).sum(-1).unsqueeze(1),.2)+x[:,:,12,None]
        mask=torch.arange(x.shape[1])[None,:]<lengths[:,None]
        a=torch.softmax(logits.masked_fill(~mask[:,:,None],-torch.inf),dim=1)
        h=(self.drop(a)[:,:,:,None]*z).sum(1).reshape(len(x),-1)
        return self.net(nn.functional.elu(h)).squeeze(-1)


def fit(cfg,seed,tr,va,weights):
    seed_all(seed);net=HistoryGAT(cfg)
    opt=torch.optim.Adam(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    x,l,y=tr;vx,vl,vy=va
    best=-1;stalled=0;start=time.perf_counter()
    for epoch in range(1,MAX_EPOCHS+1):
        net.train()
        for k in range(0,len(x),BATCH):
            opt.zero_grad(set_to_none=True)
            yy=y[k:k+BATCH]
            loss=nn.functional.binary_cross_entropy_with_logits(net(x[k:k+BATCH],l[k:k+BATCH]),yy,reduction='none')
            loss=(loss*torch.where(yy.eq(0),weights[0],weights[1])).mean()
            assert torch.isfinite(loss)
            loss.backward();opt.step()
        p=p1_scores(net,vx,vl)
        value=f1_score(vy.numpy(),p>=.5,labels=[0,1],average='macro',zero_division=0)
        if value>best+1e-12:
            best=float(value);best_epoch=epoch;state=copy.deepcopy(net.state_dict());stalled=0
        else:stalled+=1
        if stalled>=PATIENCE:break
    net.load_state_dict(state)
    return net,dict(state=state,Validation_Macro_F1=best,Best_Epoch=best_epoch,Epochs_Run=epoch,Seconds=time.perf_counter()-start)


def history_rows(frame,labels):
    """No imputation here: every observed value uses label weeks strictly < row week."""
    valid=labels[labels.Orientation_Label.notna()].copy()
    valid['day']=pd.to_datetime(valid.Week_Start).to_numpy(dtype='datetime64[D]').astype(np.int64)
    histories={u:g.sort_values('day') for u,g in valid.groupby('User',sort=False)}
    result=np.zeros((len(frame),6),float);result[:,[2,3,5]]=np.nan
    last=np.full(len(frame),'',dtype='U10')
    days=pd.to_datetime(frame.Week_Start).to_numpy(dtype='datetime64[D]').astype(np.int64)
    for user,g in frame.groupby('User',sort=False):
        h=histories.get(user)
        if h is None:continue
        hd=h.day.to_numpy();hy=h.Orientation_Label.to_numpy(int);hw=h.Week_Start.to_numpy()
        count=np.searchsorted(hd,days[g.index],side='left')
        present=count>0;idx=g.index.to_numpy()[present];c=count[present]
        if not len(idx):continue
        zeros=np.r_[0,np.cumsum(hy==0)]
        result[idx,0]=hy[c-1];result[idx,1]=1
        result[idx,2]=zeros[c]/c;result[idx,3]=1-result[idx,2];result[idx,4]=c
        result[idx,5]=(days[idx]-hd[c-1])/7
        last[idx]=hw[c-1]
        assert (hd[c-1]<days[idx]).all() and (result[idx,5]>=1).all()
    return result,last


def neighbor_indices(a,node_ids,cutoff,direction,global_ids):
    out=np.zeros((len(node_ids),17),np.int64);logs=np.zeros((len(node_ids),17),float);lengths=np.zeros(len(node_ids),np.int64)
    for j,node in enumerate(node_ids):
        ids=a.indices[a.indptr[node]:a.indptr[node+1]];weights=a.data[a.indptr[node]:a.indptr[node+1]]
        own=1+weights[ids==node].sum();other=ids!=node;ids,weights=ids[other],weights[other]
        if len(ids)>16:
            digest=hashlib.sha256(f'42|{cutoff}|{direction}|{node}'.encode()).digest()
            choice=np.sort(np.random.default_rng(int.from_bytes(digest[:8],'little')).choice(len(ids),16,replace=False))
            ids,weights=ids[choice],weights[choice]
        ids=np.r_[node,ids];weights=np.r_[own,weights];n=len(ids)
        out[j,:n]=global_ids[ids];logs[j,:n]=np.log(weights);lengths[j]=n
    return out,logs,lengths


def prepare():
    if (RUN/'inputs.npz').exists():return
    data=pd.read_csv(ROOT/'data/processed/weekly_leakage_controlled_prediction_dataset.csv',dtype={'User':str}).sort_values(['Week_Start','User']).reset_index(drop=True)
    f=pd.read_csv(ROOT/'data/processed/weekly_temporal_node_features.csv',dtype={'User':str}).sort_values(['Week_Start','User']).reset_index(drop=True)
    labels=pd.read_csv(ROOT/'data/processed/weekly_user_orientation_labels.csv',dtype={'User':str})
    labels['User']=labels.User.str.casefold();f['User']=f.User.str.casefold()
    assert not labels.duplicated(['Week_Start','User']).any()
    h,last=history_rows(f,labels)
    raw=np.c_[f[FEATURES].to_numpy(),h]
    lookup=pd.Series(f.index,index=pd.MultiIndex.from_frame(f[['Week_Start','User']]))
    target_ids=lookup.loc[pd.MultiIndex.from_frame(data[['Week_Start','User']])].to_numpy()
    assert (pd.to_datetime(data.Feature_Last_Interaction_UTC,utc=True)<pd.to_datetime(data.Week_Start,utc=True)).all()
    assert (pd.to_datetime(f.Feature_Window_End_Exclusive,utc=True)==pd.to_datetime(f.Week_Start,utc=True)).all()
    n=len(data);arrays={};audits=[];cumulative=defaultdict(int)
    for direction in ['forward','reverse']:
        arrays[direction+'_ids']=np.zeros((n,17),np.int64)
        arrays[direction+'_logs']=np.zeros((n,17),np.float32)
        arrays[direction+'_lengths']=np.zeros(n,np.int64)
    groups=dict(tuple(data.groupby('Week_Start')))
    events=pd.read_csv(ROOT/'data/processed/reconstructed_temporal_edges.csv')
    events['Timestamp']=pd.to_datetime(events.Timestamp,format='mixed',utc=True,errors='coerce')
    assert events.Timestamp.notna().all()
    events.Source=events.Source.str.casefold();events.Target=events.Target.str.casefold()
    for cutoff,g in f.groupby('Week_Start',sort=True):
        completed=(pd.Timestamp(cutoff)-pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        e=pd.read_csv(ROOT/f'data/processed/temporal_graphs/edges_{completed}.csv')
        e.Source=e.Source.str.casefold();e.Target=e.Target.str.casefold()
        for s,t,w in e.itertuples(index=False,name=None):cumulative[s,t]+=int(w)
        users=g.User.tolist();index={u:i for i,u in enumerate(users)}
        prior=events[events.Timestamp<pd.Timestamp(cutoff,tz='UTC')]
        assert set(users)==set(prior.Source)|set(prior.Target)
        assert sum(cumulative.values())==len(prior)
        assert pd.Timestamp(completed)+pd.Timedelta(days=7)<=pd.Timestamp(cutoff)
        source=[index[s] for s,t in cumulative];target=[index[t] for s,t in cumulative]
        a=sparse.coo_matrix((list(cumulative.values()),(target,source)),shape=(len(users),len(users))).tocsr()
        samples=groups.get(cutoff,data.iloc[:0]);node_ids=np.array([index[u] for u in samples.User],int)
        for direction,adj in [('forward',a),('reverse',a.T.tocsr())]:
            ids,logs,lengths=neighbor_indices(adj,node_ids,cutoff,direction,g.index.to_numpy())
            arrays[direction+'_ids'][samples.index]=ids
            arrays[direction+'_logs'][samples.index]=logs
            arrays[direction+'_lengths'][samples.index]=lengths
            mask=np.arange(17)[None,:]<lengths[:,None]
            assert (f.Week_Start.to_numpy()[ids][mask]==cutoff).all()
            known_history=last[ids][mask];has=known_history!=''
            assert (known_history[has]<cutoff).all()
        audits.append(dict(Week_Start=cutoff,Latest_Graph_Week=completed,Graph_Events=len(prior),Known_Nodes=len(users),Prediction_Users=len(samples),No_History_Users=int((h[target_ids[samples.index],1]==0).sum())))
    values=dict(week=data.Week_Start.to_numpy(dtype='U10'),user=data.User.to_numpy(dtype='U15'),y=data.Orientation_Label.to_numpy(int),
                x=raw[target_ids],raw=raw,target_ids=target_ids,last_label=last[target_ids],**arrays)
    with (RUN/'inputs.npz').open('xb') as stream:np.savez(stream,**values)
    output=data[['Week_Start','User','Feature_Last_Interaction_UTC']].copy()
    output[HISTORY]=h[target_ids];output['Latest_Previous_Label_Week']=last[target_ids]
    with (ROOT/'data/processed/weekly_historical_orientation_features.csv').open('x',encoding='utf-8',newline='') as stream:output.to_csv(stream,index=False,na_rep='NA')
    with (ROOT/'results/tables/history_graph_cutoff_audit.csv').open('x',encoding='utf-8',newline='') as stream:pd.DataFrame(audits).to_csv(stream,index=False)
    print('Prepared historical features and both graph directions for',n,'eligible rows.',flush=True)


def impute(x,prevalence,cap):
    out=x.copy();missing=out[...,7]==0
    out[...,8][missing]=prevalence;out[...,9][missing]=1-prevalence
    out[...,11][missing]=cap;out[...,11]=np.minimum(out[...,11],cap)
    assert np.isfinite(out).all()
    return out


def origin_arrays(cache,origin):
    week=cache['week'];ids=[np.flatnonzero((week>=origin['Train_First_Week'])&(week<=origin['Train_Last_Week'])),
                          np.flatnonzero(week==origin['Validation_Week']),np.flatnonzero(week==origin['Test_Week'])]
    tr,va,te=ids;assert len(tr)==origin['Train_Users'] and len(te)==origin['Test_Users'] and len(va)==origin['Validation_Users']
    assert max(week[tr])<origin['Validation_Week']<origin['Test_Week']
    p0=float(np.mean(cache['y'][tr]==0));recency=cache['x'][tr,11];cap=int(np.nanmax(recency)+1) if np.isfinite(recency).any() else 1
    own=impute(cache['x'],p0,cap)
    scaler=StandardScaler().fit(own[tr]);history_scaler=StandardScaler().fit(own[tr,6:])
    weights=len(tr)/(2*np.bincount(cache['y'][tr],minlength=2))
    arrays={}
    for name,direction in [(NEURAL[0],None),(NEURAL[1],'reverse'),(NEURAL[2],'forward')]:
        x=np.zeros((len(own),17,13),np.float32)
        if direction is None:
            x[:,0,:12]=scaler.transform(own);lengths=np.ones(len(own),np.int64)
        else:
            lengths=cache[direction+'_lengths']
            neighbor=impute(cache['raw'][cache[direction+'_ids']],p0,cap)
            x[:,:,:12]=((neighbor-scaler.mean_)/scaler.scale_).astype(np.float32)
            x[:,:,12]=cache[direction+'_logs']
            x[np.arange(17)[None,:]>=lengths[:,None]]=0
        np.testing.assert_allclose(x[:,0,:12],scaler.transform(own),rtol=1e-6,atol=1e-6)
        arrays[name]=(x,lengths)
    return ids,arrays,own,scaler,history_scaler,weights,p0,cap


def job(origin):
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    with np.load(RUN/'inputs.npz',allow_pickle=False) as stream:cache={k:stream[k] for k in stream.files}
    ids,arrays,own,scaler,hs,weights,p0,cap=origin_arrays(cache,origin)
    train,val,test=ids;week=origin['Test_Week']
    def split(name,indices):
        x,l=arrays[name]
        return torch.from_numpy(x[indices]),torch.from_numpy(l[indices]),torch.from_numpy(cache['y'][indices].astype(np.float32))
    tuning=[];validation={}
    for name in NEURAL[:2]:
        validation[name]=[]
        for grid_id,cfg in enumerate(GRID):
            path=RUN/f'{week}_{name.replace(" ","_")}_tune_{grid_id}.pt'
            if path.exists():fitted=load_pt(path)
            else:
                _,fitted=fit(cfg,42,split(name,train),split(name,val),weights);save_pt(path,fitted)
            validation[name].append(fitted['Validation_Macro_F1'])
            tuning.append(dict(Week_Start=week,Model=name,Grid_ID=grid_id,**cfg,**{k:v for k,v in fitted.items() if k!='state'}))
    joint=np.mean([validation[name] for name in NEURAL[:2]],axis=0)
    selected=int(np.argmax(joint));cfg=GRID[selected]
    logistic_tuning=[]
    train_hist=hs.transform(own[train,6:]);val_hist=hs.transform(own[val,6:]);test_hist=hs.transform(own[test,6:])
    for c in C_GRID:
        lr=LogisticRegression(C=c,class_weight={0:weights[0],1:weights[1]},max_iter=3000,random_state=42,solver='lbfgs')
        lr.fit(train_hist,cache['y'][train]);v=score(cache['y'][val],lr.predict_proba(val_hist)[:,1]>=.5,lr.predict_proba(val_hist)[:,1])['macro_F1']
        logistic_tuning.append((v,c))
        tuning.append(dict(Week_Start=week,Model=MODELS[0],C=c,Validation_Macro_F1=v))
    _,best_c=sorted(logistic_tuning,key=lambda v:(-v[0],v[1]))[0]
    rows=[];parameters=[]
    for name in MODELS:
        for seed in SEEDS:
            path=RUN/f'{week}_{name.replace(" ","_")}_seed_{seed}.pt'
            if path.exists():result=load_pt(path)
            elif name==MODELS[0]:
                lr=LogisticRegression(C=best_c,class_weight={0:weights[0],1:weights[1]},max_iter=3000,random_state=seed,solver='lbfgs')
                lr.fit(train_hist,cache['y'][train]);p1=lr.predict_proba(test_hist)[:,1]
                fitted=dict(Validation_Macro_F1=score(cache['y'][val],lr.predict_proba(val_hist)[:,1]>=.5,lr.predict_proba(val_hist)[:,1])['macro_F1'],Best_Epoch=0,Epochs_Run=0,
                            coefficients=lr.coef_,intercept=lr.intercept_,Iterations=int(lr.n_iter_[0]))
                result=dict(p1=p1,fit=fitted,seed=seed,test_users=cache['user'][test]);save_pt(path,result)
            else:
                net,fitted=fit(cfg,seed,split(name,train),split(name,val),weights)
                p1=p1_scores(net,*split(name,test)[:2])
                result=dict(p1=p1,fit=fitted,seed=seed,test_users=cache['user'][test]);save_pt(path,result)
            assert result['seed']==seed
            np.testing.assert_array_equal(result['test_users'],cache['user'][test])
            for cohort,mask in [('All eligible',np.ones(len(test),bool)),('History available',cache['x'][test,7]==1)]:
                assert mask.any()
                metrics=score(cache['y'][test][mask],result['p1'][mask]>=.5,result['p1'][mask])
                rows.append(dict(Week_Start=week,Model=name,Seed=seed,Cohort=cohort,Test_Users=int(mask.sum()),No_History_Users=int((cache['x'][test,7]==0).sum()),**metrics))
            fitted=result['fit']
            parameter=dict(Week_Start=week,Model=name,Seed=seed,Train_First_Week=origin['Train_First_Week'],Train_Last_Week=origin['Train_Last_Week'],Validation_Week=origin['Validation_Week'],
                           Train_Users=len(train),Validation_Users=len(val),Test_Users=len(test),Threshold=.5,Class_0_Weight=weights[0],Class_1_Weight=weights[1],
                           Imputation_Class_0_Rate=p0,Recency_Cap=cap,Scaler_Fit_Rows=len(train),
                           Scaler_Mean=json.dumps((hs if name==MODELS[0] else scaler).mean_.tolist()),Scaler_Scale=json.dumps((hs if name==MODELS[0] else scaler).scale_.tolist()),
                           Selected_Grid_ID=selected if name!=MODELS[0] else -1,Selected_C=best_c if name==MODELS[0] else np.nan,
                           Joint_Validation_Macro_F1=float(joint[selected]),Validation_Macro_F1=fitted['Validation_Macro_F1'],Best_Epoch=fitted['Best_Epoch'],Epochs_Run=fitted['Epochs_Run'],
                           **(cfg if name!=MODELS[0] else {}))
            parameters.append(parameter)
        print(f'{week}: {name}, ten seeds and both evaluation cohorts complete.',flush=True)
    return rows,parameters,tuning


def reference_records():
    with np.load(RUN/'inputs.npz',allow_pickle=False) as f:cache={k:f[k] for k in ['week','user','y','x']}
    old=pd.read_csv(ROOT/'data/processed/rolling_origin_baseline_predictions.csv',dtype={'User':str})
    old=old[old.Model.eq('Persistence')]
    original=pd.read_csv(ROOT/'results/tables/rolling_origin_baseline_metrics_by_week.csv')
    protocol=pd.read_csv(ROOT/'results/tables/rolling_origin_baseline_protocol.csv')
    rows=[]
    for week in protocol.Test_Week:
        mask=cache['week']==week
        users=cache['user'][mask];y=cache['y'][mask];available=cache['x'][mask,7]==1
        previous=old[old.Week_Start.eq(week)].set_index('User').loc[users]
        np.testing.assert_array_equal(previous.Orientation_Label,y)
        all_metrics=score(y,previous.Prediction.to_numpy(),previous.Probability_Class_1.to_numpy())
        for metric in METRICS:
            np.testing.assert_allclose(all_metrics[metric],original[original.Model.eq('Persistence')&original.Week_Start.eq(week)].iloc[0][metric],atol=1e-12)
        for cohort,keep in [('All eligible',np.ones(len(y),bool)),('History available',available)]:
            values=score(y[keep],previous.Prediction.to_numpy()[keep],previous.Probability_Class_1.to_numpy()[keep])
            rows.append(dict(Week_Start=week,Model='Persistence',Seed=np.nan,Cohort=cohort,Test_Users=int(keep.sum()),No_History_Users=int((~available).sum()),**values))
        for seed in SEEDS:
            path=ROOT/f'results/models/weekly_graph_forecasting_runs/{week}_GAT_reverse_primary_seed_{seed}.pt'
            saved=load_pt(path);np.testing.assert_array_equal(saved['test_users'],users)
            for cohort,keep in [('All eligible',np.ones(len(y),bool)),('History available',available)]:
                p1=saved['p1'][keep];values=score(y[keep],p1>=.5,p1)
                rows.append(dict(Week_Start=week,Model='Reversed GAT (no history)',Seed=seed,Cohort=cohort,Test_Users=int(keep.sum()),No_History_Users=int((~available).sum()),**values))
    return rows


def main():
    assert not (ROOT/'results/tables/history_graph_metrics_by_seed_week.csv').exists()
    manifest=json.loads((RUN/'manifest.json').read_text())
    assert all(sha256(ROOT/p)==h for p,h in manifest['protected_hashes'].items())
    config_path=RUN/'config.json'
    if config_path.exists():assert json.loads(config_path.read_text())==CONFIG
    else:
        with config_path.open('x',encoding='utf-8') as f:json.dump(CONFIG,f,indent=2)
    prepare()
    protocol=pd.read_csv(ROOT/'results/tables/rolling_origin_baseline_protocol.csv')
    assert protocol.Test_Week.tolist()==pd.date_range('2022-12-19','2023-03-27',freq='7D').strftime('%Y-%m-%d').tolist()
    rows=[];params=[];tuning=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(job,o) for o in protocol.to_dict('records')]
        for i,future in enumerate(as_completed(jobs),1):
            m,h,g=future.result();rows+=m;params+=h;tuning+=g
            print('Completed supplementary origins',i,'/15',flush=True)
    assert len(rows)==1200 and len(params)==600
    assert all(sha256(ROOT/p)==h for p,h in manifest['protected_hashes'].items())
    for name,frame in [('history_graph_metrics_by_seed_week.csv',pd.DataFrame(rows).sort_values(['Week_Start','Model','Cohort','Seed'])),
                       ('history_graph_hyperparameters.csv',pd.DataFrame(params)),('history_graph_validation_grid.csv',pd.DataFrame(tuning))]:
        with (ROOT/'results/tables'/name).open('x',encoding='utf-8',newline='') as f:frame.to_csv(f,index=False,na_rep='NA')
    print('All 600 supplementary fits and 1200 full/sensitivity metric records complete.',flush=True)


if __name__=='__main__':main()
