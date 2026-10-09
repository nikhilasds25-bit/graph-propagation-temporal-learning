"""Independent replay of saved forecasting results and temporal provenance."""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIL import Image
from weekly_graph_forecasting import ROOT, RUN, GRID, SEEDS, load_pt, design, make_model, p1_scores
import torch
from build_orientation_baselines import FEATURES, METRICS, score, sha256


def main():
    torch.set_num_threads(1)
    tables=ROOT/'results/tables'
    output=tables/'graph_forecasting_independent_validation.txt'
    if output.exists():
        raise FileExistsError(output)
    records=pd.concat([pd.read_csv(tables/'graph_models_metrics_by_seed_week.csv'),pd.read_csv(tables/'graph_direction_ablation_metrics.csv')])
    hp=pd.concat([pd.read_csv(tables/'graph_model_hyperparameters.csv'),pd.read_csv(tables/'graph_direction_ablation_hyperparameters.csv')])
    grid=pd.read_csv(tables/'graph_models_validation_grid.csv')
    protocol=pd.read_csv(tables/'rolling_origin_baseline_protocol.csv')
    with np.load(RUN/'inputs.npz',allow_pickle=False) as archive:
        cache={name:archive[name] for name in archive.files}
    assert len(records)==len(hp)==2850 and len(grid)==315
    keys=['Week_Start','Model','Direction','Control','Seed']
    assert not records.duplicated(keys).any() and not hp.duplicated(keys).any()
    merged=records.merge(hp,on=keys,suffixes=('','_hp'),validate='one_to_one')
    replayed=0
    for origin in protocol.to_dict('records'):
        week=origin['Test_Week']
        training=(cache['week']>=origin['Train_First_Week'])&(cache['week']<=origin['Train_Last_Week'])
        test=cache['week']==week
        assert origin['Train_Last_Week']<origin['Validation_Week']<week
        scaler=StandardScaler().fit(cache['x'][training])
        weights=training.sum()/(2*np.bincount(cache['y'][training],minlength=2))
        for r in merged[merged.Week_Start.eq(week)].itertuples():
            assert r.Threshold==.5 and r.Scaler_Fit_Rows==training.sum()
            np.testing.assert_allclose(json.loads(r.Scaler_Mean),scaler.mean_)
            np.testing.assert_allclose(json.loads(r.Scaler_Scale),scaler.scale_)
            np.testing.assert_allclose([r.Class_0_Weight,r.Class_1_Weight],weights)
            choices=grid[grid.Week_Start.eq(week)&grid.Model.eq(r.Model)&grid.Direction.eq(r.Direction)]
            assert len(choices)==3
            selected=choices.sort_values(['Validation_Macro_F1','Grid_ID'],ascending=[False,True]).iloc[0]
            assert r.Selected_Grid_ID==selected.Grid_ID
            for k,v in GRID[int(r.Selected_Grid_ID)].items():
                assert np.isclose(getattr(r,k),v)
            path=RUN/f'{week}_{r.Model.replace(" ","_")}_{r.Direction}_{r.Control}_seed_{r.Seed}.pt'
            saved=load_pt(path)
            assert saved['seed']==r.Seed and saved['selected_grid_id']==r.Selected_Grid_ID
            np.testing.assert_array_equal(saved['test_users'],cache['user'][test])
            p1=saved['p1']
            assert len(p1)==r.Test_Users and np.isfinite(p1).all() and ((p1>=0)&(p1<=1)).all()
            metrics=score(cache['y'][test],(p1>=.5).astype(int),p1)
            if r.Seed==42 and (r.Control=='primary' or r.Model=='GRU (aligned)'):
                out,lengths=design(cache,r.Model,r.Direction,r.Control,scaler.mean_,scaler.scale_)
                net=make_model(r.Model,GRID[int(r.Selected_Grid_ID)])
                net.load_state_dict(saved['fit']['state'])
                replay=p1_scores(net,torch.from_numpy(out[test]),torch.from_numpy(lengths[test]))
                np.testing.assert_allclose(replay,p1,atol=1e-7)
                validation=cache['week']==origin['Validation_Week']
                vp=p1_scores(net,torch.from_numpy(out[validation]),torch.from_numpy(lengths[validation]))
                vf=score(cache['y'][validation],(vp>=.5).astype(int),vp)['macro_F1']
                np.testing.assert_allclose(vf,r.Validation_Macro_F1,atol=1e-12)
            for metric in METRICS:
                np.testing.assert_allclose(getattr(r,metric),metrics[metric],rtol=1e-9,atol=1e-10,equal_nan=True)
            assert 1<=r.Best_Epoch<=r.Epochs_Run<=30
            replayed+=1
        for _,g in records[records.Week_Start.eq(week)].groupby(['Model','Direction','Control']):
            assert sorted(g.Seed)==SEEDS
    print(f'Passed saved-model/metric replay for {replayed} trials.',flush=True)
    dataset=pd.read_csv(ROOT/'data/processed/weekly_leakage_controlled_prediction_dataset.csv').sort_values(['Week_Start','User']).reset_index(drop=True)
    assert list(cache['week'])==dataset.Week_Start.tolist()
    assert list(cache['user'])==dataset.User.tolist()
    np.testing.assert_allclose(cache['x'],dataset[FEATURES].to_numpy(),rtol=1e-6)
    prediction=pd.to_datetime(dataset.Week_Start,utc=True)
    assert (pd.to_datetime(dataset.Feature_Last_Interaction_UTC,utc=True)<prediction).all()
    assert (pd.to_datetime(dataset.Feature_Window_End_Exclusive,utc=True)==prediction).all()
    for i,length in enumerate(cache['lengths']):
        assert cache['sequence_mask'][i].sum()==length
        np.testing.assert_allclose(cache['raw_seq'][i,length-1],cache['x'][i])
    events=pd.read_csv(ROOT/'data/processed/reconstructed_temporal_edges.csv')
    events['Source']=events.Source.str.casefold()
    events['Target']=events.Target.str.casefold()
    events['Timestamp']=pd.to_datetime(events.Timestamp,utc=True,format='mixed',errors='coerce')
    audit=pd.read_csv(tables/'graph_forecasting_cutoff_audit.csv')
    mapping=pd.read_csv(ROOT/'data/processed/weekly_graph_learning_node_indices.csv')
    for r in audit.itertuples():
        prior=events[events.Timestamp<pd.Timestamp(r.Week_Start,tz='UTC')]
        assert len(prior)==r.Historical_Events
        users=set(prior.Source)|set(prior.Target)
        nodes=mapping[mapping.Week_Start.eq(r.Week_Start)]
        assert set(nodes.User)==users and len(nodes)==r.Known_Nodes
        assert nodes.User.tolist()==sorted(users) and nodes.Node_Index.tolist()==list(range(len(users)))
        assert r.Symmetric_GCN_Reversal_Identical and r.Rewired_In_Out_Strength_Preserved
        assert pd.Timestamp(r.Latest_Completed_Graph_Week)+pd.Timedelta(days=7)<=pd.Timestamp(r.Week_Start)
    weekly=pd.read_csv(tables/'graph_forecasting_all_model_weekly_metrics.csv')
    comparisons=pd.read_csv(tables/'graph_model_comparisons.csv')
    draws=np.random.default_rng(42).integers(0,15,(10000,15))
    weeks=protocol.Test_Week.tolist()
    for r in comparisons.itertuples():
        def values(name):
            metric=r.Metric
            if name=='No-skill':
                name,metric='Majority','no_skill_class_0_PR_AUC'
            return weekly[weekly.Model.eq(name)].set_index('Week_Start').loc[weeks,metric].to_numpy()
        delta=values(r.Model_A)-values(r.Model_B)
        np.testing.assert_allclose(r.Mean_A_Minus_B,delta.mean(),atol=1e-12)
        np.testing.assert_allclose([r.Paired_Bootstrap_95_Lower,r.Paired_Bootstrap_95_Upper],np.quantile(delta[draws].mean(1),[.025,.975]),atol=1e-12)
    figures=pd.read_csv(tables/'graph_forecasting_publication_figure_manifest.csv')
    assert len(figures)==5
    for r in figures.itertuples():
        im=Image.open(ROOT/r.PNG)
        assert min(im.size)>=2000 and all(abs(d-400)<.1 for d in im.info['dpi'])
        assert '<svg' in (ROOT/r.SVG).read_text(encoding='utf-8')
    manifest=json.loads((RUN/'manifest.json').read_text())
    assert all(sha256(ROOT/p)==h for p,h in manifest['protected_hashes'].items())
    text=f'PASS: replayed every metric for {replayed} saved seed/week/control predictions.\nPASS: training-only scalers and class weights, validation grid choice, fixed threshold, cohorts and seeds.\nPASS: own-week temporal cutoffs, latest sequence step, historical node identities and raw-event totals.\nPASS: all paired-week bootstrap intervals independently reproduced.\nPASS: five 400-DPI PNG/SVG figure pairs.\nPASS: all {len(manifest["protected_hashes"])} protected prior files retain identical SHA-256 hashes.\n'
    with output.open('x',encoding='utf-8') as f:
        f.write(text)
    print(text,flush=True)


if __name__=='__main__':
    main()
