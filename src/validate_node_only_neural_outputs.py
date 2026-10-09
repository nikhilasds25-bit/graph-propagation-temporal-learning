"""Independent checks of serialized neural trials, metrics and paired statistics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, balanced_accuracy_score, matthews_corrcoef
from run_node_only_neural_baselines import ROOT, FEATURES, SEEDS, GRID, MLP, GRU, seed_all
from build_orientation_baselines import sha256


def main():
    tables = ROOT / 'results/tables'
    output = tables / 'node_only_neural_saved_output_qa.txt'
    if output.exists():
        raise FileExistsError(output)
    metrics = pd.read_csv(tables / 'node_only_neural_metrics_by_seed_week.csv')
    hp = pd.read_csv(tables / 'node_only_training_hyperparameters.csv')
    grid = pd.read_csv(tables / 'node_only_validation_grid.csv')
    comparisons = pd.read_csv(tables / 'node_only_model_comparisons.csv')
    weekly = pd.read_csv(tables / 'node_only_all_model_weekly_metrics.csv')
    summary = pd.read_csv(tables / 'node_only_neural_summary.csv')
    protocol = pd.read_csv(tables / 'rolling_origin_baseline_protocol.csv')
    data = pd.read_csv(ROOT / 'data/processed/weekly_leakage_controlled_prediction_dataset.csv', dtype={'User': str}).sort_values(['Week_Start', 'User'])
    historic_features = pd.read_csv(ROOT / 'data/processed/weekly_temporal_node_features.csv', dtype={'User': str})
    historic_features['User'] = historic_features.User.str.casefold()
    baseline = pd.read_csv(tables / 'rolling_origin_baseline_metrics_by_week.csv')
    audit = json.loads((tables / 'node_only_neural_leakage_audit.json').read_text())
    assert len(metrics) == len(hp) == 300
    assert len(grid) == 180
    assert not metrics.duplicated(['Week_Start', 'Model', 'Seed']).any()
    assert pd.to_datetime(data.Feature_Last_Interaction_UTC, utc=True).lt(pd.to_datetime(data.Week_Start, utc=True)).all()
    for (week, model), group in metrics.groupby(['Week_Start', 'Model']):
        assert sorted(group.Seed) == SEEDS
        origin = protocol[protocol.Test_Week.eq(week)].iloc[0]
        train = data[data.Week_Start.between(origin.Train_First_Week, origin.Train_Last_Week)]
        test = data[data.Week_Start.eq(week)]
        assert train.Week_Start.max() < origin.Validation_Week < week
        assert group.Test_Users.eq(len(test)).all()
        selected = grid[grid.Week_Start.eq(week) & grid.Model.eq(model)].sort_values(['Validation_Macro_F1', 'Grid_ID'], ascending=[False, True]).iloc[0]
        parameters = hp[hp.Week_Start.eq(week) & hp.Model.eq(model)]
        assert parameters.Selected_Grid_ID.eq(selected.Grid_ID).all()
        assert parameters.Threshold.eq(.5).all()
        weights = [len(train)/(2*train.Orientation_Label.eq(k).sum()) for k in [0, 1]]
        assert np.allclose(parameters.Class_0_Weight, weights[0]) and np.allclose(parameters.Class_1_Weight, weights[1])
        if model == 'MLP':
            assert np.allclose(json.loads(parameters.iloc[0].Scaler_Mean), train[FEATURES].mean())
            assert parameters.Scaler_Fit_Feature_Rows.eq(len(train)).all()
        else:
            assert parameters.Scaler_Feature_Max_Week.lt(origin.Train_Last_Week).all()
            # Union of prior rows referenced by training sequences, independently
            # derived from each user's latest training label week.
            latest_train_week = train.groupby('User').Week_Start.max()
            limits = historic_features.User.map(latest_train_week)
            tokens = historic_features.loc[limits.notna() & historic_features.Week_Start.lt(limits)]
            assert parameters.Scaler_Fit_Feature_Rows.eq(len(tokens)).all()
            assert np.allclose(json.loads(parameters.iloc[0].Scaler_Mean), tokens[FEATURES].mean())
        state_hashes = []
        for row in group.itertuples(index=False):
            checkpoint = ROOT / f'results/models/node_only_neural_runs/{week}_{model}_seed_{row.Seed}.pt'
            trial = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert trial['seed'] == row.Seed and trial['selected_grid_id'] == selected.Grid_ID
            assert trial['test_users'] == test.User.tolist()
            p1 = trial['p1']
            y = test.Orientation_Label.to_numpy()
            predicted = p1 >= .5
            assert np.isclose(f1_score(y, predicted, average='macro'), row.macro_F1)
            assert np.isclose(average_precision_score(y == 0, 1-p1), row.class_0_PR_AUC)
            assert np.isclose(balanced_accuracy_score(y, predicted), row.balanced_accuracy)
            assert np.isclose(matthews_corrcoef(y, predicted), row.MCC)
            assert np.isclose(np.mean(y == 0), row.no_skill_class_0_PR_AUC)
            assert 1 <= trial['fit']['Best_Epoch'] <= trial['fit']['Epochs_Run'] <= 30
            state_hashes.append(b''.join(v.numpy().tobytes() for v in trial['fit']['state'].values()))
        assert len(set(state_hashes)) == 10, 'Seed fits are not independent parameter states'
    for name in ['Majority', 'Persistence', 'Logistic Regression']:
        for metric in ['macro_F1', 'class_0_PR_AUC', 'balanced_accuracy', 'MCC']:
            old = baseline[baseline.Model.eq(name)].sort_values('Week_Start')[metric].to_numpy()
            new = weekly[weekly.Model.eq(name)].sort_values('Week_Start')[metric].to_numpy()
            assert np.allclose(old, new, rtol=0, atol=1e-15)
    weeks = protocol.Test_Week.tolist()
    draws = np.random.default_rng(42).integers(0, 15, size=(10000, 15))
    for row in comparisons.itertuples(index=False):
        a = weekly[weekly.Model.eq(row.Model_A)].set_index('Week_Start').loc[weeks, row.Metric].to_numpy()
        b = weekly[weekly.Model.eq('Majority' if row.Model_B == 'No-skill' else row.Model_B)].set_index('Week_Start').loc[weeks, 'no_skill_class_0_PR_AUC' if row.Model_B == 'No-skill' else row.Metric].to_numpy()
        delta = a-b
        lo, hi = np.quantile(delta[draws].mean(axis=1), [.025, .975])
        assert np.allclose([lo, hi], [row.Paired_Bootstrap_95_Lower, row.Paired_Bootstrap_95_Upper])
        assert row.A_Better_Weeks == sum(delta > 0)
        assert np.isclose(row.Mean_A_Minus_B, delta.mean())
    for row in summary[summary.Model.isin(['MLP', 'GRU'])].itertuples(index=False):
        matrix = metrics[metrics.Model.eq(row.Model)].pivot(index='Seed', columns='Week_Start', values=row.Metric).loc[SEEDS, weeks].to_numpy()
        assert np.isclose(matrix.mean(), row.Mean)
        assert np.isclose(matrix.mean(axis=1).std(ddof=1), row.SD_Over_Seed_Means)
    for name in ['node_only_rolling_macro_f1', 'node_only_rolling_class_0_pr_auc', 'node_only_rolling_balanced_accuracy', 'node_only_comparison_forest']:
        path = ROOT / 'results/figures' / (name+'.png')
        with Image.open(path) as image:
            assert all(abs(v-400)<.1 for v in image.info['dpi'])
        assert path.with_suffix('.svg').exists()
    # Architecture and mask tests do not depend on trained-model scores.
    torch.set_num_threads(2)
    seed_all(42)
    g = GRU(GRID[1]).eval()
    x = torch.randn(3, 5, 6)
    lengths = torch.tensor([3, 0, 5])
    changed = x.clone()
    changed[0, 3:] = 100
    changed[1] = 100
    with torch.no_grad():
        assert torch.allclose(g(x, lengths), g(changed, lengths))
        assert torch.allclose(g(x, lengths)[1], g.classifier.bias[0])
    assert sum(isinstance(m, torch.nn.Linear) for m in MLP(GRID[0]).modules()) == 3
    assert sum(isinstance(m, torch.nn.Dropout) for m in MLP(GRID[0]).modules()) == 2
    assert all(sha256(ROOT/path) == h for path,h in audit['protected_hashes'].items())
    message = ('Independent saved-output QA passed: 300 model/seed/week trials, 180 tuning configurations, '
               'all paired cohorts, chronology, validation-only configuration selection, training class weights and MLP scalers, '
               'GRU historical scaling bounds, distinct seed parameter states, checkpoint prediction metrics, '
               'unchanged classical metrics, paired-bootstrap differences, seed SDs, padding/empty-sequence masks, '
               '400-DPI PNG and SVG availability, original-file hashes.\n')
    with output.open('x', encoding='utf-8') as f:
        f.write(message)
    print(message)


if __name__ == '__main__':
    main()
