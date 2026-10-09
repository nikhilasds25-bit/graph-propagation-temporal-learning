"""Independent history joins, checkpoint replay and statistical validation."""
import json,sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import permutation_test
from sklearn.preprocessing import StandardScaler
from PIL import Image
from run_history_graph_supplement import ROOT,RUN,MODELS,NEURAL,HISTORY,GRID,HistoryGAT,origin_arrays,load_pt,p1_scores
from build_orientation_baselines import FEATURES,METRICS,score,sha256
from report_history_graph_supplement import PAIRS,STAT_METRICS


def features(cache):
    f=pd.read_csv(ROOT/'data/processed/weekly_temporal_node_features.csv').sort_values(['Week_Start','User']).reset_index(drop=True)
    f.User=f.User.str.casefold();f['Feature_Row']=f.index;f['Cutoff']=pd.to_datetime(f.Week_Start)
    labels=pd.read_csv(ROOT/'data/processed/weekly_user_orientation_labels.csv')
    labels=labels[labels.Orientation_Label.notna()].copy();labels.User=labels.User.str.casefold()
    labels=labels.sort_values(['User','Week_Start'])
    labels['Prior_Count']=labels.groupby('User').cumcount()+1
    labels['Zero']=labels.Orientation_Label.eq(0).astype(int)
    labels['Prior_Zeros']=labels.groupby('User').Zero.cumsum()
    labels['Label_Date']=pd.to_datetime(labels.Week_Start)
    labels=labels[['User','Label_Date','Orientation_Label','Prior_Count','Prior_Zeros']]
    joined=pd.merge_asof(f.sort_values('Cutoff'),labels.sort_values('Label_Date'),left_on='Cutoff',right_on='Label_Date',by='User',allow_exact_matches=False).sort_values('Feature_Row')
    known=joined.Label_Date.notna().to_numpy()
    expected=np.zeros((len(joined),6));expected[:,[2,3,5]]=np.nan
    expected[known,0]=joined.loc[known,'Orientation_Label']
    expected[known,1]=1
    expected[known,2]=joined.loc[known,'Prior_Zeros']/joined.loc[known,'Prior_Count']
    expected[known,3]=1-expected[known,2]
    expected[known,4]=joined.loc[known,'Prior_Count']
    expected[known,5]=(joined.loc[known,'Cutoff']-joined.loc[known,'Label_Date']).dt.days/7
    assert (joined.loc[known,'Label_Date']<joined.loc[known,'Cutoff']).all()
    np.testing.assert_allclose(cache['raw'][:,6:],expected,equal_nan=True)
    np.testing.assert_allclose(cache['raw'][:,:6],f[FEATURES].to_numpy())
    last=joined.Label_Date.dt.strftime('%Y-%m-%d').fillna('').to_numpy()
    np.testing.assert_array_equal(cache['last_label'],last[cache['target_ids']])
    np.testing.assert_allclose(cache['x'],cache['raw'][cache['target_ids']],equal_nan=True)
    with np.load(ROOT/'results/models/weekly_graph_forecasting_runs/inputs.npz',allow_pickle=False) as old:
        np.testing.assert_array_equal(cache['week'],old['week']);np.testing.assert_array_equal(cache['user'],old['user'])
        for direction in ['forward','reverse']:
            lengths=cache[direction+'_lengths'];ids=cache[direction+'_ids'];mask=np.arange(17)[None,:]<lengths[:,None]
            np.testing.assert_array_equal(lengths,old['gat_lengths_'+direction])
            np.testing.assert_allclose(cache['raw'][ids[:,0]],cache['x'],equal_nan=True)
            assert (f.Week_Start.to_numpy()[ids][mask]==np.broadcast_to(cache['week'][:,None],ids.shape)[mask]).all()
            neighbor_last=last[ids][mask];cutoff=np.broadcast_to(cache['week'][:,None],ids.shape)[mask];have=neighbor_last!=''
            assert (neighbor_last[have]<cutoff[have]).all()
    events=pd.read_csv(ROOT/'data/processed/reconstructed_temporal_edges.csv')
    events.Source=events.Source.str.casefold();events.Target=events.Target.str.casefold()
    events['Timestamp']=pd.to_datetime(events.Timestamp,utc=True,format='mixed',errors='coerce')
    user_array=f.User.to_numpy()
    for week in sorted(set(cache['week'])):
        prior=events[events.Timestamp<pd.Timestamp(str(week),tz='UTC')]
        counts=prior.groupby(['Source','Target']).size().to_dict()
        known=set(prior.Source)|set(prior.Target)
        samples=np.flatnonzero(cache['week']==week)
        for direction in ['forward','reverse']:
            for i in samples:
                receiver=cache['user'][i];length=cache[direction+'_lengths'][i]
                neighbors=user_array[cache[direction+'_ids'][i,:length]]
                assert receiver in known and set(neighbors)<=known
                assert neighbors[0]==receiver
                weights=[1+counts.get((receiver,receiver),0)]
                for neighbor in neighbors[1:]:
                    key=(neighbor,receiver) if direction=='forward' else (receiver,neighbor)
                    assert counts.get(key,0)>0
                    weights.append(counts[key])
                np.testing.assert_allclose(cache[direction+'_logs'][i,:length],np.log(weights),atol=1e-6)
    print('PASS: independent as-of join verifies history for all',len(f),'historical node rows; each sampled graph message verified against strictly prior raw interactions.',flush=True)


def main():
    torch.set_num_threads(1)
    with np.load(RUN/'inputs.npz',allow_pickle=False) as file:cache={k:file[k] for k in file.files}
    features(cache)
    if '--features-only' in sys.argv:return
    tables=ROOT/'results/tables';output=tables/'history_graph_independent_validation.txt'
    assert not output.exists()
    records=pd.read_csv(tables/'history_graph_metrics_by_seed_week.csv')
    hp=pd.read_csv(tables/'history_graph_hyperparameters.csv');grid=pd.read_csv(tables/'history_graph_validation_grid.csv')
    protocol=pd.read_csv(tables/'rolling_origin_baseline_protocol.csv')
    assert len(records)==1200 and len(hp)==600 and len(grid)==165
    assert not records.duplicated(['Week_Start','Model','Seed','Cohort']).any()
    replayed=0
    for origin in protocol.to_dict('records'):
        week=origin['Test_Week'];ids,arrays,own,scaler,hs,weights,p0,cap=origin_arrays(cache,origin)
        tr,va,te=ids
        assert origin['Train_Last_Week']<origin['Validation_Week']<week
        independent=StandardScaler().fit(own[tr]);np.testing.assert_allclose(independent.mean_,scaler.mean_)
        expected_weights=len(tr)/(2*np.bincount(cache['y'][tr],minlength=2))
        chosen=grid[grid.Week_Start.eq(week)&grid.Model.isin(NEURAL[:2])].groupby('Grid_ID').Validation_Macro_F1.mean().sort_index().to_numpy().argmax()
        logistic=grid[grid.Week_Start.eq(week)&grid.Model.eq(MODELS[0])].sort_values(['Validation_Macro_F1','C'],ascending=[False,True]).iloc[0].C
        for r in hp[hp.Week_Start.eq(week)].itertuples():
            assert r.Threshold==.5 and r.Scaler_Fit_Rows==len(tr) and r.Recency_Cap==cap
            assert np.isclose(r.Imputation_Class_0_Rate,p0)
            np.testing.assert_allclose([r.Class_0_Weight,r.Class_1_Weight],expected_weights)
            scale=hs if r.Model==MODELS[0] else scaler
            np.testing.assert_allclose(json.loads(r.Scaler_Mean),scale.mean_);np.testing.assert_allclose(json.loads(r.Scaler_Scale),scale.scale_)
            assert r.Selected_C==logistic if r.Model==MODELS[0] else r.Selected_Grid_ID==chosen
            path=RUN/f'{week}_{r.Model.replace(" ","_")}_seed_{r.Seed}.pt';saved=load_pt(path)
            assert saved['seed']==r.Seed
            np.testing.assert_array_equal(saved['test_users'],cache['user'][te])
            if r.Model==MODELS[0]:
                xx=hs.transform(own[te,6:]);logits=xx@saved['fit']['coefficients'].ravel()+saved['fit']['intercept'][0]
                p1=1/(1+np.exp(-logits))
            else:
                cfg=GRID[int(chosen)]
                for key,value in cfg.items():assert np.isclose(getattr(r,key),value)
                net=HistoryGAT(cfg);net.load_state_dict(saved['fit']['state'])
                xx,ll=arrays[r.Model]
                p1=p1_scores(net,torch.from_numpy(xx[te]),torch.from_numpy(ll[te]))
            np.testing.assert_allclose(p1,saved['p1'],rtol=1e-6,atol=1e-7)
            for cohort,keep in [('All eligible',np.ones(len(te),bool)),('History available',cache['x'][te,7]==1)]:
                row=records[records.Week_Start.eq(week)&records.Model.eq(r.Model)&records.Seed.eq(r.Seed)&records.Cohort.eq(cohort)].iloc[0]
                metric=score(cache['y'][te][keep],saved['p1'][keep]>=.5,saved['p1'][keep])
                assert row.Test_Users==keep.sum()
                for key in METRICS:np.testing.assert_allclose(row[key],metric[key],atol=1e-12,equal_nan=True)
            replayed+=1
        print('Replayed all predictions and metrics for',week,flush=True)
    weekly=pd.read_csv(tables/'history_graph_weekly_metrics.csv')
    pairs=pd.read_csv(tables/'history_graph_paired_comparisons.csv');blocks=pd.read_csv(tables/'history_graph_block_bootstrap.csv');tests=pd.read_csv(tables/'history_graph_holm_tests.csv')
    with np.load(RUN/'statistics_draws.npz') as file:draws={k:file[k] for k in file.files}
    weeks=protocol.Test_Week.tolist()
    for r in pairs.itertuples():
        g=weekly[weekly.Cohort.eq(r.Cohort)]
        a=g[g.Model.eq(r.Model_A)].set_index('Week_Start').loc[weeks,r.Metric].to_numpy()
        b=g[g.Model.eq(r.Model_B)].set_index('Week_Start').loc[weeks,r.Metric].to_numpy();delta=a-b
        np.testing.assert_allclose(r.Mean_A_Minus_B,delta.mean(),atol=1e-12)
        np.testing.assert_allclose([r.Ordinary_95_Lower,r.Ordinary_95_Upper],np.quantile(delta[draws['ordinary']].mean(1),[.025,.975]),atol=1e-12)
        for length in [3,4]:
            q=blocks[blocks.Cohort.eq(r.Cohort)&blocks.Model_A.eq(r.Model_A)&blocks.Model_B.eq(r.Model_B)&blocks.Metric.eq(r.Metric)&blocks.Block_Length.eq(length)].iloc[0]
            np.testing.assert_allclose([q.Moving_Block_95_Lower,q.Moving_Block_95_Upper],np.quantile(delta[draws[f'block_{length}']].mean(1),[.025,.975]),atol=1e-12)
        if r.Primary_Comparison:
            result=permutation_test((delta,),lambda x,axis:np.mean(x,axis=axis),permutation_type='samples',n_resamples=np.inf,alternative='two-sided',vectorized=True)
            q=tests[tests.Cohort.eq(r.Cohort)&tests.Model_A.eq(r.Model_A)&tests.Model_B.eq(r.Model_B)&tests.Metric.eq(r.Metric)].iloc[0]
            np.testing.assert_allclose(q.Uncorrected_P,result.pvalue,atol=1e-12)
    for _,g in tests.groupby('Cohort'):
        ordered=g.sort_values('Uncorrected_P');adjusted=[];largest=0
        for i,p in enumerate(ordered.Uncorrected_P):largest=max(largest,(20-i)*p);adjusted.append(min(1,largest))
        np.testing.assert_allclose(ordered.Holm_Adjusted_P,adjusted,atol=1e-12)
    manifest=json.loads((RUN/'manifest.json').read_text())
    assert all(sha256(ROOT/p)==h for p,h in manifest['protected_hashes'].items())
    for r in pd.read_csv(tables/'history_graph_figure_manifest.csv').itertuples():
        im=Image.open(ROOT/r.PNG);assert min(im.size)>=2000 and all(abs(v-400)<.1 for v in im.info['dpi'])
        assert '<svg' in (ROOT/r.SVG).read_text(encoding='utf-8')
    text=f'PASS: all 802811 historical node feature histories independently reconstructed by strict as-of joins.\nPASS: every sampled graph message/weight in both directions matches strictly prior raw interaction pairs and neighbor label cutoffs.\nPASS: {replayed} model checkpoints reproduce probabilities; all 1200 full/sensitivity metric records replayed.\nPASS: training-only defaults/scalers/weights, common validation-selected hyperparameters and threshold 0.5.\nPASS: all ordinary and block intervals replayed; exact p-values agree with SciPy; Holm 20-test families reproduced.\nPASS: four 400-DPI PNG/SVG pairs.\nPASS: all {len(manifest["protected_hashes"])} protected prior files remain byte-identical.\n'
    with output.open('x',encoding='utf-8') as f:f.write(text)
    print(text,flush=True)


if __name__=='__main__':main()
