"""CPU node-only neural rolling baselines; exclusive outputs and resumable trials."""
from pathlib import Path
import copy
import hashlib
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from build_orientation_baselines import FEATURES, METRICS, score, sha256

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(42, 52))
MAX_EPOCHS = 30
PATIENCE = 4
BATCH = 1024
BOOTSTRAPS = 10000
GRID = [
    dict(hidden=16, dropout=0., lr=.001, weight_decay=0.),
    dict(hidden=32, dropout=.2, lr=.001, weight_decay=.0001),
    dict(hidden=64, dropout=.5, lr=.001, weight_decay=.0001),
    dict(hidden=16, dropout=.2, lr=.0003, weight_decay=.0001),
    dict(hidden=32, dropout=.5, lr=.0003, weight_decay=0.),
    dict(hidden=64, dropout=0., lr=.0003, weight_decay=0.),
]
MODELS = ['Majority', 'Persistence', 'Logistic Regression', 'MLP', 'GRU']
COLORS = dict(zip(MODELS, ['#737373', '#B65B25', '#174A75', '#8665A0', '#27856F']))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, d = cfg['hidden'], cfg['dropout']
        self.net = nn.Sequential(nn.Linear(6, h), nn.ReLU(), nn.Dropout(d),
                                 nn.Linear(h, h), nn.ReLU(), nn.Dropout(d), nn.Linear(h, 1))

    def forward(self, x, lengths=None):
        return self.net(x).squeeze(-1)


class GRU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gru = nn.GRU(6, cfg['hidden'], batch_first=True)
        self.dropout = nn.Dropout(cfg['dropout'])
        self.classifier = nn.Linear(cfg['hidden'], 1)

    def forward(self, x, lengths):
        # Right padding cannot affect outputs at earlier, real time steps.
        output, _ = self.gru(x)
        hidden = output[torch.arange(len(x)), (lengths-1).clamp(min=0)]
        hidden = hidden * lengths.gt(0).unsqueeze(1)
        return self.classifier(self.dropout(hidden)).squeeze(-1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def probabilities(model, x, lengths):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), BATCH):
            output.append(torch.sigmoid(model(x[start:start+BATCH], lengths[start:start+BATCH])).numpy())
    return np.concatenate(output)


def fit(name, cfg, seed, train, validation, weights):
    seed_all(seed)
    model = (MLP if name == 'MLP' else GRU)(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    xt, lt, yt = train
    xv, lv, yv = validation
    best, best_epoch, stalled, state = -1., 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        # Preserve chronological sample ordering, including within minibatches.
        for start in range(0, len(xt), BATCH):
            xb, lb, yb = xt[start:start+BATCH], lt[start:start+BATCH], yt[start:start+BATCH]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, lb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none')
            loss = (loss * torch.where(yb.eq(0), weights[0], weights[1])).mean()
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        pv = probabilities(model, xv, lv)
        value = f1_score(yv.numpy(), pv >= .5, labels=[0, 1], average='macro', zero_division=0)
        if value > best + 1e-12:
            best, best_epoch, stalled = float(value), epoch, 0
            state = copy.deepcopy(model.state_dict())
        else:
            stalled += 1
        if stalled >= PATIENCE:
            break
    model.load_state_dict(state)
    return model, {'Validation_Macro_F1': best, 'Best_Epoch': best_epoch, 'Epochs_Run': epoch,
                   'Seconds': time.perf_counter()-started, 'state': state}


def write_trial(path, result):
    with path.open('xb') as f:
        torch.save(result, f)


def read_trial(path):
    # Only checkpoints authored by this run under a verified manifest are used.
    return torch.load(path, map_location='cpu', weights_only=False)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = True
    tables = ROOT / 'results/tables'
    figures = ROOT / 'results/figures'
    run = ROOT / 'results/models/node_only_neural_runs'
    paths = {
        'metrics': tables / 'node_only_neural_metrics_by_seed_week.csv',
        'summary': tables / 'node_only_neural_summary.csv',
        'comparisons': tables / 'node_only_model_comparisons.csv',
        'hyperparameters': tables / 'node_only_training_hyperparameters.csv',
        'grid': tables / 'node_only_validation_grid.csv',
        'weekly': tables / 'node_only_all_model_weekly_metrics.csv',
        'audit': tables / 'node_only_neural_leakage_audit.json',
        'report': tables / 'node_only_neural_report.txt',
    }
    figure_names = ['node_only_rolling_macro_f1', 'node_only_rolling_class_0_pr_auc',
                    'node_only_rolling_balanced_accuracy', 'node_only_comparison_forest']
    for name in figure_names:
        for ext in ['png', 'svg']:
            paths[name+'_'+ext] = figures / (name+'.'+ext)
    collisions = [str(p) for p in paths.values() if p.exists()]
    if collisions:
        raise FileExistsError('No writes: existing final outputs: '+', '.join(collisions))
    dataset_path = ROOT / 'data/processed/weekly_leakage_controlled_prediction_dataset.csv'
    features_path = ROOT / 'data/processed/weekly_temporal_node_features.csv'
    baseline_path = tables / 'rolling_origin_baseline_metrics_by_week.csv'
    protocol_path = tables / 'rolling_origin_baseline_protocol.csv'
    protected = [*list((ROOT/'data/raw').rglob('*')), *list((ROOT/'data/processed').glob('*.csv')),
                 *list((ROOT/'results').rglob('*'))]
    protected = [p for p in protected if p.is_file() and run not in p.parents]
    manifest_path = run / 'manifest.json'
    config = {'seeds': SEEDS, 'grid': GRID, 'max_epochs': MAX_EPOCHS, 'patience': PATIENCE,
              'batch': BATCH, 'features': FEATURES, 'threshold': .5, 'torch': torch.__version__}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['config'] == config
        hashes = manifest['protected_hashes']
        assert all(sha256(ROOT/p)==h for p,h in hashes.items()), 'Protected files changed since run started'
    else:
        hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(protected))}
        run.mkdir(parents=True, exist_ok=True)
        with manifest_path.open('x', encoding='utf-8') as f:
            json.dump({'config': config, 'protected_hashes': hashes}, f, indent=2)
    data = pd.read_csv(dataset_path, dtype={'User': str}).sort_values(['Week_Start', 'User']).reset_index(drop=True)
    feat = pd.read_csv(features_path, dtype={'User': str})
    feat['User'] = feat.User.str.casefold()
    feat = feat.sort_values(['User', 'Week_Start']).reset_index(drop=True)
    feat['Feature_Window_End_Exclusive'] = pd.to_datetime(feat.Feature_Window_End_Exclusive, utc=True)
    assert feat.Feature_Window_End_Exclusive.equals(pd.to_datetime(feat.Week_Start, utc=True))
    assert not feat.duplicated(['Week_Start', 'User']).any()
    assert not data.duplicated(['Week_Start', 'User']).any()
    assert pd.to_datetime(data.Feature_Last_Interaction_UTC, utc=True).lt(pd.to_datetime(data.Week_Start, utc=True)).all()
    baseline = pd.read_csv(baseline_path)
    baseline_summary = pd.read_csv(tables/'rolling_origin_baseline_summary.csv')
    assert set(baseline_summary.Model)==set(MODELS[:3])
    protocol = pd.read_csv(protocol_path)
    test_weeks = protocol.Test_Week.tolist()
    assert len(test_weeks)==15 and test_weeks==sorted(test_weeks)
    for name in MODELS[:3]:
        assert baseline.loc[baseline.Model.eq(name), 'Week_Start'].tolist()==test_weeks
    # GRU sequence membership uses only user rows with feature Week_Start < label week.
    max_length = int(feat.groupby('User').size().max())
    sequence_indices = np.full((len(data), max_length), -1, dtype=np.int64)
    lengths = np.zeros(len(data), dtype=np.int64)
    user_histories = {user: (g.Week_Start.to_numpy(), g.index.to_numpy()) for user,g in feat.groupby('User', sort=False)}
    for i, row in enumerate(data.itertuples(index=False)):
        history_weeks, indices = user_histories[row.User]
        count = np.searchsorted(history_weeks, row.Week_Start, side='left')
        lengths[i] = count
        sequence_indices[i, :count] = indices[:count]
        assert (history_weeks[:count] < row.Week_Start).all()
        if count:
            assert feat.loc[indices[:count], 'Feature_Window_End_Exclusive'].lt(pd.Timestamp(row.Week_Start, tz='UTC')).all()
    raw_features = feat[FEATURES].to_numpy(np.float64)
    assert np.isfinite(raw_features).all() and np.isfinite(data[FEATURES].to_numpy()).all()
    metrics, hyperparameters, tuning = [], [], []
    for origin in protocol.itertuples(index=False):
        masks = {
            'train': data.Week_Start.ge(origin.Train_First_Week)&data.Week_Start.le(origin.Train_Last_Week),
            'validation': data.Week_Start.eq(origin.Validation_Week),
            'test': data.Week_Start.eq(origin.Test_Week),
        }
        train_ids, validation_ids, test_ids = [np.flatnonzero(masks[k]) for k in ['train','validation','test']]
        assert len(train_ids)==origin.Train_Users and len(validation_ids)==origin.Validation_Users and len(test_ids)==origin.Test_Users
        assert data.iloc[train_ids].Week_Start.max()<origin.Validation_Week<origin.Test_Week
        assert data.iloc[train_ids].Week_Start.nunique()==origin.Training_Weeks
        yt = data.iloc[train_ids].Orientation_Label.to_numpy(np.int64)
        counts = np.bincount(yt, minlength=2)
        assert (counts>0).all()
        class_weights = len(yt)/(2*counts)
        assert np.allclose(class_weights, [len(train_ids)/(2*sum(yt==0)),len(train_ids)/(2*sum(yt==1))])
        for name in ['MLP','GRU']:
            if name=='MLP':
                training_values = data.iloc[train_ids][FEATURES].to_numpy(float)
                scaler = StandardScaler().fit(training_values)
                transformed = scaler.transform(data[FEATURES].to_numpy(float)).astype(np.float32)
                fit_rows = len(train_ids)
                feature_max_week = origin.Train_Last_Week
            else:
                # Unique real tokens from training sequences only, never validation/test histories.
                token_ids = np.unique(sequence_indices[train_ids])
                token_ids = token_ids[token_ids>=0]
                training_values = raw_features[token_ids]
                scaler = StandardScaler().fit(training_values)
                all_scaled = scaler.transform(raw_features).astype(np.float32)
                transformed = np.zeros((*sequence_indices.shape, 6), dtype=np.float32)
                real = sequence_indices>=0
                transformed[real] = all_scaled[sequence_indices[real]]
                fit_rows = len(token_ids)
                feature_max_week = feat.iloc[token_ids].Week_Start.max()
                assert feature_max_week<origin.Train_Last_Week
                assert (transformed[~real]==0).all()
            assert int(scaler.n_samples_seen_)==fit_rows
            assert np.allclose(scaler.mean_, training_values.mean(axis=0))
            def split(ids):
                return (torch.from_numpy(transformed[ids]), torch.from_numpy(lengths[ids]),
                        torch.from_numpy(data.iloc[ids].Orientation_Label.to_numpy(np.float32)))
            tr, va = split(train_ids), split(validation_ids)
            candidates = []
            for grid_id, cfg in enumerate(GRID):
                path = run / f'{origin.Test_Week}_{name}_tune_{grid_id}.pt'
                if path.exists():
                    fitted = read_trial(path)
                else:
                    _, fitted = fit(name, cfg, 42, tr, va, class_weights)
                    write_trial(path, fitted)
                candidates.append((fitted['Validation_Macro_F1'], grid_id))
                tuning.append({'Week_Start': origin.Test_Week, 'Validation_Week': origin.Validation_Week,
                               'Model': name, 'Tuning_Seed':42, 'Grid_ID':grid_id, **cfg,
                               **{k:v for k,v in fitted.items() if k!='state'}})
            _, selected_id = sorted(candidates, key=lambda v:(-v[0],v[1]))[0]
            cfg = GRID[selected_id]
            # Selection is frozen before any test predictions/metrics.
            te = split(test_ids)
            for seed in SEEDS:
                path = run / f'{origin.Test_Week}_{name}_seed_{seed}.pt'
                if path.exists():
                    result = read_trial(path)
                    fitted = result['fit']
                    result_metrics = result['metrics']
                else:
                    model, fitted = fit(name, cfg, seed, tr, va, class_weights)
                    p1 = probabilities(model, te[0], te[1])
                    result_metrics = score(te[2].numpy().astype(int), (p1>=.5).astype(int), p1)
                    write_trial(path, {'fit':fitted,'metrics':result_metrics,'p1':p1,'selected_grid_id':selected_id,
                                       'test_users':data.iloc[test_ids].User.tolist(),'seed':seed})
                metrics.append({'Week_Start':origin.Test_Week,'Model':name,'Seed':seed,'Test_Users':len(test_ids),
                                'GRU_Empty_Sequence_Users':int((lengths[test_ids]==0).sum()) if name=='GRU' else 0,
                                **result_metrics})
                hyperparameters.append({'Week_Start':origin.Test_Week,'Model':name,'Seed':seed,
                                        'Train_First_Week':origin.Train_First_Week,'Train_Last_Week':origin.Train_Last_Week,
                                        'Validation_Week':origin.Validation_Week,'Train_Users':len(train_ids),
                                        'Validation_Users':len(validation_ids),'Test_Users':len(test_ids),
                                        'Selected_Grid_ID':selected_id, **cfg,'Threshold':.5,
                                        'Class_0_Weight':class_weights[0],'Class_1_Weight':class_weights[1],
                                        'Scaler_Fit_Feature_Rows':fit_rows,'Scaler_Feature_Max_Week':feature_max_week,
                                        'Scaler_Mean':json.dumps(scaler.mean_.tolist()),'Scaler_Scale':json.dumps(scaler.scale_.tolist()),
                                        'Train_Empty_Sequences':int((lengths[train_ids]==0).sum()) if name=='GRU' else 0,
                                        'Test_Empty_Sequences':int((lengths[test_ids]==0).sum()) if name=='GRU' else 0,
                                        **{k:v for k,v in fitted.items() if k!='state'}})
                print(f'{origin.Test_Week} {name} seed={seed} selected={selected_id} epoch={fitted["Best_Epoch"]} completed',flush=True)
            del transformed
    metrics = pd.DataFrame(metrics)
    hyperparameters = pd.DataFrame(hyperparameters)
    assert len(metrics)==300 and not metrics.duplicated(['Week_Start','Model','Seed']).any()
    for name in ['MLP','GRU']:
        for seed in SEEDS:
            assert metrics.loc[metrics.Model.eq(name)&metrics.Seed.eq(seed),'Week_Start'].tolist()==test_weeks
    # Week means are the unit of paired comparisons; seeds are not extra weeks.
    neural_weekly = metrics.groupby(['Model','Week_Start'],sort=False)[METRICS].mean().reset_index()
    weekly = pd.concat([baseline[['Model','Week_Start',*METRICS]],neural_weekly],ignore_index=True)
    rng = np.random.default_rng(42)
    week_draws = rng.integers(0,15,size=(BOOTSTRAPS,15))
    seed_draws = rng.integers(0,10,size=(BOOTSTRAPS,10))
    summary_rows = []
    for name in MODELS:
        frame = weekly[weekly.Model.eq(name)].set_index('Week_Start').loc[test_weeks]
        for metric in METRICS:
            means = frame[metric].to_numpy()
            if name in ['MLP','GRU']:
                matrix = metrics[metrics.Model.eq(name)].pivot(index='Seed',columns='Week_Start',values=metric).loc[SEEDS,test_weeks].to_numpy()
                seed_means = matrix.mean(axis=1)
                across = matrix[seed_draws[:,:,None],week_draws[:,None,:]].mean(axis=(1,2))
                perweek_sd = matrix.std(axis=0,ddof=1).mean()
            else:
                seed_means = np.array([means.mean()])
                across = means[week_draws].mean(axis=1)
                perweek_sd = np.nan
            low,high = np.quantile(across,[.025,.975])
            seed_ci = np.quantile(seed_means[seed_draws].mean(axis=1),[.025,.975]) if len(seed_means)==10 else (np.nan,np.nan)
            summary_rows.append({'Model':name,'Metric':metric,'Mean':means.mean(),'SD_Over_Seed_Means':seed_means.std(ddof=1) if len(seed_means)>1 else np.nan,
                                 'Mean_Within_Week_Seed_SD':perweek_sd,'SD_Over_Week_Means':means.std(ddof=1),
                                 'Bootstrap_95_Lower':low,'Bootstrap_95_Upper':high,'Seed_Mean_95_Lower':seed_ci[0],'Seed_Mean_95_Upper':seed_ci[1],
                                 'Seeds':10 if name in ['MLP','GRU'] else 1,'Test_Weeks':15,'Bootstrap_Draws':BOOTSTRAPS})
    summary = pd.DataFrame(summary_rows)
    comparisons=[]
    pairs=[('MLP','Logistic Regression'),('GRU','MLP'),('GRU','Persistence'),('GRU','Logistic Regression'),('MLP','Persistence'),('MLP','No-skill'),('GRU','No-skill')]
    for a,b in pairs:
        for metric in ['macro_F1','class_0_PR_AUC','balanced_accuracy','MCC']:
            av = weekly[weekly.Model.eq(a)].set_index('Week_Start').loc[test_weeks,metric].to_numpy()
            if b=='No-skill':
                if metric!='class_0_PR_AUC':
                    continue
                bv = weekly[weekly.Model.eq('Majority')].set_index('Week_Start').loc[test_weeks,'no_skill_class_0_PR_AUC'].to_numpy()
            else:
                bv = weekly[weekly.Model.eq(b)].set_index('Week_Start').loc[test_weeks,metric].to_numpy()
            delta=av-bv
            lo,hi=np.quantile(delta[week_draws].mean(axis=1),[.025,.975])
            comparisons.append({'Model_A':a,'Model_B':b,'Metric':metric,'Mean_A_Minus_B':delta.mean(),
                                'Paired_Bootstrap_95_Lower':lo,'Paired_Bootstrap_95_Upper':hi,
                                'A_Better_Weeks':int((delta>0).sum()),'A_Worse_Weeks':int((delta<0).sum()),'Paired_Weeks':15,
                                'Consistently_Better_All_Weeks':bool((delta>0).all()),'Bootstrap_Draws':BOOTSTRAPS})
    comparisons=pd.DataFrame(comparisons)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    plot_weeks=pd.to_datetime(test_weeks)
    rendered={}
    for name,metric,title,ylabel in [
        ('node_only_rolling_macro_f1','macro_F1','Node-only rolling-origin macro-F1','Macro-F1'),
        ('node_only_rolling_class_0_pr_auc','class_0_PR_AUC','Node-only class-0 precision–recall performance','Class-0 average precision'),
        ('node_only_rolling_balanced_accuracy','balanced_accuracy','Node-only rolling balanced accuracy','Balanced accuracy')]:
        fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
        for model_name in MODELS:
            values=weekly[weekly.Model.eq(model_name)].set_index('Week_Start').loc[test_weeks,metric].to_numpy()
            ax.plot(plot_weeks,values,label=model_name,color=COLORS[model_name],linewidth=1.6,marker='o',markersize=3)
            if model_name in ['MLP','GRU']:
                matrix=metrics[metrics.Model.eq(model_name)].pivot(index='Seed',columns='Week_Start',values=metric).loc[SEEDS,test_weeks].to_numpy()
                # Pointwise seed mean uncertainty, not an extra user-level interval.
                lo,hi=np.quantile(matrix[seed_draws].mean(axis=1),[.025,.975],axis=0)
                ax.fill_between(plot_weeks,lo,hi,color=COLORS[model_name],alpha=.15)
        if metric=='class_0_PR_AUC':
            ax.plot(plot_weeks,weekly[weekly.Model.eq('Majority')].set_index('Week_Start').loc[test_weeks,'no_skill_class_0_PR_AUC'],color='#000000',linestyle='--',label='No-skill: class-0 prevalence')
        if metric=='balanced_accuracy':
            ax.axhline(.5,color='#000000',linestyle='--',label='No-skill: 0.5')
        locator=mdates.AutoDateLocator(minticks=5,maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set(title=title,xlabel='Test week beginning Monday (UTC)',ylabel=ylabel,ylim=(0,1))
        ax.grid(axis='y',alpha=.2)
        ax.legend(frameon=False,ncol=3,fontsize=8,loc='lower left')
        fig.suptitle('Neural lines: 10-seed means; shaded bands: pointwise seed-bootstrap 95% CIs',fontsize=9,color='#555555')
        rendered[name]=fig
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,metric in zip(axes.flat,['macro_F1','class_0_PR_AUC','balanced_accuracy','MCC']):
        frame=comparisons[comparisons.Metric.eq(metric)].reset_index(drop=True)
        y=np.arange(len(frame))
        ax.errorbar(frame.Mean_A_Minus_B,y,xerr=np.array([frame.Mean_A_Minus_B-frame.Paired_Bootstrap_95_Lower,frame.Paired_Bootstrap_95_Upper-frame.Mean_A_Minus_B]),fmt='o',color='#174A75',capsize=3)
        ax.axvline(0,color='#777777',linestyle='--',linewidth=1)
        ax.set_yticks(y,[f'{r.Model_A} − {r.Model_B}' for r in frame.itertuples()])
        ax.invert_yaxis()
        ax.set(title=metric.replace('_',' '),xlabel='Paired mean difference (positive favors first model)')
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Paired rolling-week comparisons with bootstrap 95% CIs',fontsize=14)
    rendered['node_only_comparison_forest']=fig
    winners={metric:summary[summary.Metric.eq(metric)].sort_values('Mean',ascending=False).iloc[0].Model for metric in ['macro_F1','class_0_PR_AUC','balanced_accuracy']}
    audit={'all_assertions_passed':True,'test_weeks':test_weeks,'seeds':SEEDS,'features':FEATURES,
           'feature_event_timestamps_before_MLP_label_week':True,'GRU_feature_row_weeks_strictly_before_each_label_week':True,
           'scalers_training_only':True,'class_weights_training_only':True,'validation_precedes_test':True,
           'threshold_fixed':.5,'validation_only_selection_and_early_stopping':True,
           'no_label_history_features':True,'no_adjacency_or_graph_convolution':True,'same_cohort_and_test_weeks':True,
           'protected_original_files_unchanged':True,'deterministic_cpu':True,'config':config,'protected_hashes':hashes}
    assumptions=[
        'Target remains the dataset binary orientation label, not verified sentiment. Exactly the saved 15 origins and the same eligible test users are used for all models; no future labels or identities become predictors.',
        'MLP uses the target-week feature row, whose events are strictly earlier than its Monday; GRU uses all available feature rows whose own Week_Start is strictly earlier than the target Monday. Each is cumulative historical structural data, not current-week graph data.',
        'The strict GRU row-week lag means its latest feature row is generally one week older than the MLP row. GRU-versus-MLP differences combine temporal representation and recency differences and cannot isolate a pure causal benefit of temporal modeling.',
        'GRU sequences begin at the first available historical feature row, preserve chronological order, use right zero padding after scaling, and use real sequence lengths to select the last real hidden state. Zero-length histories use a masked zero state and learned classifier intercept; no label history or adjacency is provided.',
        'MLP has two same-width Linear/ReLU/Dropout hidden layers and one logit. GRU has one recurrent layer and dropout on the final masked state before its classifier. Models use only six structural numeric features.',
        'The six predeclared configurations span all requested hidden sizes, dropout levels, learning rates and weight-decay choices but are a fractional grid, not an exhaustive 54-configuration search. Configuration selection uses seed 42 validation macro-F1 only, then is frozen across ten independent seed fits per origin.',
        'Each final seed reinitializes and trains independently; early stopping uses that seed validation macro-F1, restoring the earliest best epoch. Maximum 30 epochs, patience 4, batch size 1024, Adam, no chronological sample shuffle, CPU deterministic algorithms, seeds 42 through 51. Test scores never enter selection.',
        'The probability threshold is fixed at 0.5 for MLP, GRU and existing logistic regression. Training label weights are N/(2*N_class), applied to per-sample binary cross entropy. Validation/test labels do not determine class weights.',
        'MLP StandardScaler fits current feature rows of training samples only. GRU StandardScaler fits unique real feature rows referenced by training sequences only; padding is excluded. Held-out sequences are transformed with those frozen training statistics.',
        'Class-0 PR-AUC is average precision using 1-P(class 1); class-1 PR-AUC uses P(class 1). The test class-0 prevalence is stored for every model/seed/week. Existing metrics and thresholds are reused unchanged for classical comparisons.',
        'Neural summaries report mean and sample SD of each seed mean across 15 weeks, average within-week seed SD, and SD across week means. Overall neural percentile intervals resample both seeds and weeks with 10,000 draws; classical intervals resample weeks only. Seed-mean-only intervals are also reported.',
        'Paired comparisons average neural seeds within each test week then resample identical test weeks across models. Individual users and seed repeats are never treated as independent test weeks. These comparison CIs condition on the ten-seed average and omit residual seed uncertainty.',
        'Week-bootstrap intervals assume exchangeable weeks and do not correct temporal autocorrelation or repeated-user dependence. Pointwise figure bands resample seeds, not users. All intervals are descriptive, not proof of superiority.',
        'Consistent improvement means a strictly positive seed-averaged difference in every one of the 15 paired test weeks. Winners by a mean are descriptive; week win counts and paired intervals determine the qualified comparison narrative.',
        'The evaluated cohort excludes tied labels and missing graph-history users from the prior pipeline and conditions on current-week observed labels. No user-disjoint or full-population performance claim is made. Graph extraction and orientation-label validity limitations persist.',
        'No GNN, graph convolution, adjacency matrix or message passing is used. Original raw and baseline files are hash-protected. Trial checkpoints are new exclusive files and support resumption without overwriting completed trials.',
    ]
    selected=comparisons[comparisons.Model_B.isin(['Persistence','Logistic Regression'])]
    report=json.dumps({'test_weeks':test_weeks,'seeds':SEEDS,'best_mean_models':winners,
                       'GRU_consistently_beats_Persistence_all_four_metrics':bool(comparisons[comparisons.Model_A.eq('GRU')&comparisons.Model_B.eq('Persistence')].Consistently_Better_All_Weeks.all())},indent=2)
    report+='\n\nSelected paired comparisons:\n'+selected.to_string(index=False)
    report+='\n\nGRU versus MLP (temporal-history comparison with recency caveat):\n'+comparisons[comparisons.Model_A.eq('GRU')&comparisons.Model_B.eq('MLP')].to_string(index=False)
    report+='\n\nExact assumptions and limitations:\n'+'\n'.join(f'{i}. {v}' for i,v in enumerate(assumptions,1))
    assert all(sha256(ROOT/p)==h for p,h in hashes.items()), 'Original protected output modified'
    for p in paths.values():
        p.parent.mkdir(parents=True,exist_ok=True)
    for frame,key in [(metrics,'metrics'),(summary,'summary'),(comparisons,'comparisons'),(hyperparameters,'hyperparameters'),(pd.DataFrame(tuning),'grid'),(weekly,'weekly')]:
        with paths[key].open('x',encoding='utf-8',newline='') as f:
            frame.to_csv(f,index=False,na_rep='NA')
    for name,fig in rendered.items():
        for ext in ['png','svg']:
            with paths[name+'_'+ext].open('xb') as f:
                fig.savefig(f,format=ext,dpi=400,facecolor='white')
        plt.close(fig)
    with paths['audit'].open('x',encoding='utf-8') as f:
        json.dump(audit,f,indent=2)
    with paths['report'].open('x',encoding='utf-8') as f:
        f.write(report+'\n')
    print(report,flush=True)


if __name__=='__main__':
    main()
