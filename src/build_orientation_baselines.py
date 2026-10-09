"""Leakage-controlled weekly orientation labels and classical rolling baselines.

Run from this project with the bundled Python runtime. Existing outputs abort.
"""
from pathlib import Path
import hashlib
import json
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, matthews_corrcoef,
                             average_precision_score, roc_auc_score, brier_score_loss)
from sklearn.exceptions import ConvergenceWarning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
MIN_TRAIN_WEEKS = 8
C_GRID = [.01, .1, 1., 10., 100.]
BOOTSTRAPS = 10000
FEATURES = ['in_degree', 'out_degree', 'in_strength', 'out_strength', 'PageRank', 'reverse_PageRank']
MODELS = ['Majority', 'Persistence', 'Logistic Regression']
COLORS = {'Majority': '#737373', 'Persistence': '#B65B25', 'Logistic Regression': '#174A75'}
METRICS = ['accuracy', 'balanced_accuracy', 'macro_F1', 'class_0_precision', 'class_0_recall',
           'class_0_F1', 'class_1_F1', 'MCC', 'class_0_PR_AUC', 'class_1_PR_AUC', 'ROC_AUC',
           'Brier_score', 'no_skill_class_0_PR_AUC']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def week_of(dates):
    return dates.dt.normalize() - pd.to_timedelta(dates.dt.dayofweek, unit='D')


def normalize(users):
    return users.str.strip().str.lstrip('@').str.casefold()


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return max(0., center-half), min(1., center+half)


def score(y, predictions, p1):
    assert np.isfinite(p1).all() and ((p1 >= 0) & (p1 <= 1)).all()
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, labels=[0, 1], zero_division=0)
    both = len(np.unique(y)) == 2
    return {
        'accuracy': accuracy_score(y, predictions),
        'balanced_accuracy': balanced_accuracy_score(y, predictions) if both else np.nan,
        'macro_F1': f1_score(y, predictions, labels=[0, 1], average='macro', zero_division=0),
        'class_0_precision': precision[0], 'class_0_recall': recall[0],
        'class_0_F1': f1[0], 'class_1_F1': f1[1], 'MCC': matthews_corrcoef(y, predictions) if both else np.nan,
        'class_0_PR_AUC': average_precision_score(y == 0, 1-p1) if both else np.nan,
        'class_1_PR_AUC': average_precision_score(y == 1, p1) if both else np.nan,
        'ROC_AUC': roc_auc_score(y, p1) if both else np.nan,
        'Brier_score': brier_score_loss(y, p1),
        'no_skill_class_0_PR_AUC': float(np.mean(y == 0)),
    }


def format_axis(ax, weeks, partial_weeks):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlabel('Prediction week beginning Monday (UTC)')
    ax.grid(axis='y', alpha=.2, linewidth=.6)
    for week in partial_weeks:
        if week in weeks:
            ax.axvspan(week-pd.Timedelta(days=2), week+pd.Timedelta(days=2), color='#AAAAAA', alpha=.16, zorder=0)
            ax.text(week, 1.015, 'Partial', transform=ax.get_xaxis_transform(), ha='center', fontsize=8, color='#666666')


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tables = ROOT / 'results/tables'
    processed = ROOT / 'data/processed'
    figures = ROOT / 'results/figures'
    paths = {
        'labels': processed / 'weekly_user_orientation_labels.csv',
        'dataset': processed / 'weekly_leakage_controlled_prediction_dataset.csv',
        'balance': tables / 'weekly_orientation_balance.csv',
        'metrics': tables / 'rolling_origin_baseline_metrics_by_week.csv',
        'summary': tables / 'rolling_origin_baseline_summary.csv',
        'paired': tables / 'rolling_origin_baseline_paired_comparisons.csv',
        'protocol': tables / 'rolling_origin_baseline_protocol.csv',
        'tuning': tables / 'rolling_origin_logistic_validation.csv',
        'predictions': processed / 'rolling_origin_baseline_predictions.csv',
        'audit': tables / 'orientation_baseline_leakage_audit.json',
        'report': tables / 'orientation_baseline_report.txt',
    }
    figure_names = ['weekly_orientation_balance', 'rolling_origin_macro_f1', 'rolling_origin_class_0_pr_auc',
                    'rolling_origin_balanced_accuracy', 'rolling_origin_class_balance']
    for name in figure_names:
        for ext in ['png', 'svg']:
            paths[name + '_' + ext] = figures / f'{name}.{ext}'
    collisions = [str(p) for p in paths.values() if p.exists()]
    if collisions:
        raise FileExistsError('No writes: output paths already exist: ' + ', '.join(collisions))
    inputs = {
        'tweets': ROOT / 'data/raw/master_tweets.csv',
        'features': processed / 'weekly_temporal_node_features.csv',
        'graph_summary': tables / 'weekly_temporal_graph_summary.csv',
        # Ancillary provenance audit only; never current-week model input.
        'edge_provenance': processed / 'reconstructed_temporal_edges.csv',
    }
    protected = set(inputs.values()) | set((ROOT / 'data/raw').rglob('*'))
    protected = {p for p in protected if p.is_file()}
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(protected)}
    tweets = pd.read_csv(inputs['tweets'], usecols=['Date', 'User', 'Target'], dtype=str, keep_default_na=False)
    dates = pd.to_datetime(tweets.Date, format='mixed', utc=True, errors='coerce')
    target = pd.to_numeric(tweets.Target, errors='coerce')
    valid = dates.notna() & target.isin([0, 1]) & tweets.User.str.strip().ne('')
    rejected_tweets = int((~valid).sum())
    tweets = tweets.loc[valid].copy()
    dates = dates.loc[valid]
    tweets['Week_Start'] = week_of(dates)
    tweets['User'] = normalize(tweets.User)
    tweets['Target'] = target.loc[valid].astype(int)
    weeks = pd.date_range(tweets.Week_Start.min(), tweets.Week_Start.max(), freq='7D')
    labels = tweets.groupby(['Week_Start', 'User']).Target.agg(
        Tweet_Count='size', Target_0_Count=lambda s: int(s.eq(0).sum()), Target_1_Count=lambda s: int(s.eq(1).sum())).reset_index()
    labels['Target_0_Ratio'] = labels.Target_0_Count / labels.Tweet_Count
    labels['Target_1_Ratio'] = labels.Target_1_Count / labels.Tweet_Count
    labels['Orientation_Label'] = pd.Series(pd.NA, index=labels.index, dtype='Int64')
    labels.loc[labels.Target_0_Count.gt(labels.Target_1_Count), 'Orientation_Label'] = 0
    labels.loc[labels.Target_1_Count.gt(labels.Target_0_Count), 'Orientation_Label'] = 1
    assert labels.Tweet_Count.sum() == len(tweets)
    assert not labels.duplicated(['Week_Start', 'User']).any()
    assert (labels.Target_0_Count + labels.Target_1_Count).equals(labels.Tweet_Count)
    first_tweet_week = tweets.groupby('User').Week_Start.min()
    graph_summary = pd.read_csv(inputs['graph_summary'])
    assert pd.to_datetime(graph_summary.Week_Start, utc=True).tolist() == weeks.tolist()
    feature_rows = pd.read_csv(inputs['features'], dtype={'User': str})
    assert list(feature_rows) == ['Week_Start', 'User', 'Feature_Window_End_Exclusive', *FEATURES]
    feature_rows['Week_Start'] = pd.to_datetime(feature_rows.Week_Start, utc=True, errors='raise')
    feature_rows['User'] = normalize(feature_rows.User)
    feature_rows['Feature_Window_End_Exclusive'] = pd.to_datetime(feature_rows.Feature_Window_End_Exclusive, utc=True, errors='raise')
    assert not feature_rows.duplicated(['Week_Start', 'User']).any()
    assert feature_rows.Feature_Window_End_Exclusive.equals(feature_rows.Week_Start)
    assert np.isfinite(feature_rows[FEATURES].to_numpy()).all()
    assert (feature_rows[FEATURES] >= 0).all().all()
    # Validate actual feature event times, not merely the equal-to-Monday exclusive cutoff.
    edge_events = pd.read_csv(inputs['edge_provenance'], usecols=['Source', 'Target', 'Timestamp'], dtype=str)
    edge_events['Timestamp'] = pd.to_datetime(edge_events.Timestamp, format='mixed', utc=True, errors='raise')
    edge_events['Week_Start'] = week_of(edge_events.Timestamp)
    edge_events['Source'] = normalize(edge_events.Source)
    edge_events['Target'] = normalize(edge_events.Target)
    edge_groups = dict(tuple(edge_events.groupby('Week_Start')))
    features_by_week = dict(tuple(feature_rows.groupby('Week_Start')))
    latest = {}
    provenance_frames = []
    for week in weeks:
        f = features_by_week.get(week, feature_rows.iloc[:0])
        assert set(f.User) == set(latest), 'Historical feature universe differs from prior interactions'
        if len(f):
            times = f.User.map(latest)
            assert times.lt(week).all(), 'Feature provenance includes current/future events'
            provenance_frames.append(pd.DataFrame({'Week_Start': week, 'User': f.User.to_numpy(), 'Feature_Last_Interaction_UTC': times.to_numpy()}))
        current = edge_groups.get(week, edge_events.iloc[:0])
        for col in ['Source', 'Target']:
            for user, stamp in current.groupby(col).Timestamp.max().items():
                latest[user] = max(latest.get(user, stamp), stamp)
    provenance = pd.concat(provenance_frames, ignore_index=True)
    feature_rows = feature_rows.merge(provenance, on=['Week_Start', 'User'], validate='one_to_one')
    joined = labels.merge(feature_rows, on=['Week_Start', 'User'], how='left', validate='one_to_one', indicator=True)
    dataset = joined.loc[joined.Orientation_Label.notna() & joined._merge.eq('both'),
                         ['Week_Start', 'User', 'Orientation_Label', 'Feature_Window_End_Exclusive', 'Feature_Last_Interaction_UTC', *FEATURES]].copy()
    dataset['Orientation_Label'] = dataset.Orientation_Label.astype(int)
    assert not dataset.duplicated(['Week_Start', 'User']).any()
    assert pd.to_datetime(dataset.Feature_Last_Interaction_UTC, utc=True).lt(dataset.Week_Start).all()
    assert list(dataset[FEATURES]) == FEATURES
    assert not set(['Tweet_Count', 'Target_0_Count', 'Target_1_Count', 'Target_0_Ratio', 'Target_1_Ratio', 'Likes', 'Retweets', 'Target']).intersection(dataset.columns)
    balance_rows = []
    for week in weeks:
        group = joined.loc[joined.Week_Start.eq(week)]
        eligible = dataset.loc[dataset.Week_Start.eq(week)]
        labelled = group.Orientation_Label.notna()
        cold = labelled & group._merge.eq('left_only')
        first_seen = group.User.map(first_tweet_week).eq(week)
        n = len(eligible)
        k = int(eligible.Orientation_Label.eq(0).sum())
        low, high = wilson(k, n)
        balance_rows.append({
            'Week_Start': week, 'Evaluated_Users': n, 'Class_0_Count': k, 'Class_1_Count': n-k,
            'Class_0_Share': k/n if n else np.nan, 'Class_1_Share': (n-k)/n if n else np.nan,
            'Cold_Start_Count': int(cold.sum()), 'Tie_Count': int(group.Orientation_Label.isna().sum()),
            'Class_0_Wilson_95_Lower': low, 'Class_0_Wilson_95_Upper': high,
            'Active_User_Weeks': len(group), 'Active_Labelled_Users': int(labelled.sum()),
            'Users_With_Historical_Features': int(group._merge.eq('both').sum()),
            'Historical_Feature_Users_Total': len(features_by_week.get(week, [])),
            'Cold_Start_First_Tweet_Week_Count': int((cold & first_seen).sum()),
            'Cold_Start_Previously_Tweeted_Count': int((cold & ~first_seen).sum()),
        })
        assert n + int(cold.sum()) + int(group.Orientation_Label.isna().sum()) == len(group)
    balance = pd.DataFrame(balance_rows)
    eligible_weeks = sorted(dataset.Week_Start.unique())
    label_lookup = labels.loc[labels.Orientation_Label.notna()].sort_values(['Week_Start', 'User'])
    metric_rows, prediction_rows, protocol_rows, tuning_rows = [], [], [], []
    for i in range(MIN_TRAIN_WEEKS + 1, len(eligible_weeks)):
        test_week, val_week = eligible_weeks[i], eligible_weeks[i-1]
        train_weeks = eligible_weeks[:i-1]
        train = dataset.loc[dataset.Week_Start.isin(train_weeks)].copy()
        val = dataset.loc[dataset.Week_Start.eq(val_week)].copy()
        test = dataset.loc[dataset.Week_Start.eq(test_week)].copy()
        if train.Orientation_Label.nunique() != 2:
            continue
        assert train.Week_Start.max() < val_week < test_week
        assert not set(train.Week_Start).intersection([val_week, test_week])
        xtrain, xval, xtest = (frame[FEATURES].to_numpy(dtype=float) for frame in [train, val, test])
        ytrain, yval, ytest = (frame.Orientation_Label.to_numpy(dtype=int) for frame in [train, val, test])
        scaler = StandardScaler().fit(xtrain)
        assert int(scaler.n_samples_seen_) == len(train)
        assert np.allclose(scaler.mean_, xtrain.mean(axis=0), atol=1e-12)
        ztrain, zval = scaler.transform(xtrain), scaler.transform(xval)
        # Select solely by validation macro-F1, with smaller C breaking ties.
        candidates = []
        for c in C_GRID:
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model = LogisticRegression(C=c, class_weight='balanced', solver='lbfgs', max_iter=2000, random_state=SEED)
                model.fit(ztrain, ytrain)
            pval = model.predict_proba(zval)[:, list(model.classes_).index(1)]
            value = f1_score(yval, (pval >= .5).astype(int), labels=[0, 1], average='macro', zero_division=0)
            tuning_rows.append({'Test_Week': test_week, 'Validation_Week': val_week, 'C': c, 'Validation_Macro_F1': value})
            candidates.append((value, c, model))
        selected_value, selected_c, selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
        # Test transformation/inference begins only after the selection is fixed.
        ztest = scaler.transform(xtest)
        train_p1 = float(ytrain.mean())
        majority = 0 if np.mean(ytrain == 0) >= .5 else 1
        past = label_lookup.loc[label_lookup.Week_Start.lt(test_week)].drop_duplicates('User', keep='last').set_index('User')
        assert past.Week_Start.lt(test_week).all()
        prior_labels = test.User.map(past.Orientation_Label).to_numpy(dtype=float, na_value=np.nan)
        persistence_p1 = np.where(np.isnan(prior_labels), train_p1, prior_labels)
        persistence_predictions = np.where(np.isnan(prior_labels), majority, prior_labels).astype(int)
        probabilities = {
            'Majority': np.full(len(test), train_p1),
            'Persistence': persistence_p1,
            'Logistic Regression': selected.predict_proba(ztest)[:, list(selected.classes_).index(1)],
        }
        predictions = {
            'Majority': np.full(len(test), majority, dtype=int),
            'Persistence': persistence_predictions,
            'Logistic Regression': (probabilities['Logistic Regression'] >= .5).astype(int),
        }
        protocol_rows.append({
            'Test_Week': test_week, 'Train_First_Week': train.Week_Start.min(), 'Train_Last_Week': train.Week_Start.max(),
            'Training_Weeks': len(train_weeks), 'Validation_Week': val_week, 'Train_Users': len(train),
            'Validation_Users': len(val), 'Test_Users': len(test), 'Training_Class_0_Share': float(np.mean(ytrain == 0)),
            'Selected_C': selected_c, 'Selected_Validation_Macro_F1': selected_value,
            'Threshold_Class_1': .5, 'Scaler_Fit_Rows': int(scaler.n_samples_seen_),
            'Scaler_Mean': json.dumps(scaler.mean_.tolist()), 'Scaler_Scale': json.dumps(scaler.scale_.tolist()),
            'Persistence_Fallback_Users': int(np.isnan(prior_labels).sum()),
        })
        for name in MODELS:
            metric_rows.append({'Week_Start': test_week, 'Model': name, 'Test_Users': len(test),
                                'Test_Class_0_Count': int(np.sum(ytest == 0)), 'Test_Class_1_Count': int(np.sum(ytest == 1)),
                                **score(ytest, predictions[name], probabilities[name])})
            prediction_rows.append(pd.DataFrame({'Week_Start': test_week, 'User': test.User.to_numpy(), 'Model': name,
                                                 'Orientation_Label': ytest, 'Prediction': predictions[name],
                                                 'Probability_Class_0': 1-probabilities[name], 'Probability_Class_1': probabilities[name]}))
        print(f'{test_week:%Y-%m-%d}: train {len(train):,}, validation {len(val):,}, test {len(test):,}, selected C={selected_c:g}', flush=True)
    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise ValueError('No eligible rolling test origins; no output files written')
    test_weeks = sorted(metrics.Week_Start.unique())
    assert test_weeks == pd.DataFrame(protocol_rows).Test_Week.tolist()
    assert all(metrics.loc[metrics.Model.eq(name), 'Week_Start'].tolist() == test_weeks for name in MODELS)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    assert not predictions.duplicated(['Week_Start', 'User', 'Model']).any()
    # Paired week draws are shared by all models and metrics.
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(test_weeks), size=(BOOTSTRAPS, len(test_weeks)))
    summary_rows, comparison_rows = [], []
    for name in MODELS:
        frame = metrics.loc[metrics.Model.eq(name)].set_index('Week_Start').loc[test_weeks]
        for metric in METRICS:
            values = frame[metric].to_numpy(dtype=float)
            finite = np.isfinite(values)
            sampled = values[draws]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                means = np.nanmean(sampled, axis=1)
            low, high = np.nanquantile(means, [.025, .975]) if finite.any() else (np.nan, np.nan)
            summary_rows.append({'Model': name, 'Metric': metric, 'Test_Weeks': len(test_weeks), 'Valid_Weeks': int(finite.sum()),
                                 'Mean': float(np.nanmean(values)) if finite.any() else np.nan,
                                 'Std': float(np.nanstd(values, ddof=1)) if finite.sum() > 1 else np.nan,
                                 'Bootstrap_95_Lower': low, 'Bootstrap_95_Upper': high, 'Bootstrap_Draws': BOOTSTRAPS, 'Seed': SEED})
    for a, b in [('Persistence', 'Majority'), ('Logistic Regression', 'Majority'), ('Logistic Regression', 'Persistence')]:
        for metric in METRICS:
            av = metrics.loc[metrics.Model.eq(a)].set_index('Week_Start').loc[test_weeks, metric].to_numpy(float)
            bv = metrics.loc[metrics.Model.eq(b)].set_index('Week_Start').loc[test_weeks, metric].to_numpy(float)
            delta = av-bv
            finite = np.isfinite(delta)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                means = np.nanmean(delta[draws], axis=1)
            low, high = np.nanquantile(means, [.025, .975]) if finite.any() else (np.nan, np.nan)
            wins = delta < 0 if metric == 'Brier_score' else delta > 0
            comparison_rows.append({'Model_A': a, 'Model_B': b, 'Metric': metric,
                                    'Paired_Weeks': int(finite.sum()), 'Mean_A_Minus_B': float(np.nanmean(delta)) if finite.any() else np.nan,
                                    'Bootstrap_95_Lower': low, 'Bootstrap_95_Upper': high,
                                    'A_Better_Weeks': int((wins & finite).sum()), 'Bootstrap_Draws': BOOTSTRAPS, 'Seed': SEED})
    summary = pd.DataFrame(summary_rows)
    comparisons = pd.DataFrame(comparison_rows)
    # First/last coverage weeks are marked partial, based on observed collection boundaries.
    partial_weeks = []
    if dates.min() > weeks[0]:
        partial_weeks.append(weeks[0])
    if dates.max() < weeks[-1] + pd.Timedelta(days=7):
        partial_weeks.append(weeks[-1])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    rendered_figures = {}
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    axes[0].plot(weeks, balance.Active_Labelled_Users, label='Valid majority labels', color='#4D8064', linewidth=1.8)
    axes[0].plot(weeks, balance.Evaluated_Users, label='With historical features', color='#174A75', linewidth=1.8)
    axes[0].set(title='a  Weekly users with valid orientation labels', ylabel='Users')
    axes[0].legend(frameon=False)
    axes[0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    for c, color in [(0, '#174A75'), (1, '#B65B25')]:
        axes[1].plot(weeks, balance[f'Class_{c}_Share'], label=f'Class {c}', color=color, linewidth=1.8)
    axes[1].set(title='b  Class shares among users with historical features', ylabel='Share', ylim=(0, 1))
    axes[1].legend(frameon=False)
    axes[2].plot(weeks, balance.Class_0_Share, color='#174A75', linewidth=1.8)
    axes[2].fill_between(weeks, balance.Class_0_Wilson_95_Lower, balance.Class_0_Wilson_95_Upper, color='#174A75', alpha=.2, label='Wilson 95% CI')
    axes[2].set(title='c  Class-0 share with Wilson interval', ylabel='Class-0 share', ylim=(0, 1))
    axes[2].legend(frameon=False)
    for ax in axes:
        format_axis(ax, weeks, partial_weeks)
        ax.set_ylim(bottom=0)
    for ax in axes[1:]:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('Weekly binary orientation labels and class balance', fontsize=14)
    rendered_figures['weekly_orientation_balance'] = fig
    for name, metric, title, ylabel in [
        ('rolling_origin_macro_f1', 'macro_F1', 'Rolling-origin macro-F1', 'Macro-F1'),
        ('rolling_origin_class_0_pr_auc', 'class_0_PR_AUC', 'Rolling-origin class-0 precision–recall performance', 'Class-0 average precision'),
        ('rolling_origin_balanced_accuracy', 'balanced_accuracy', 'Rolling-origin balanced accuracy', 'Balanced accuracy')]:
        fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
        for model_name in MODELS:
            frame = metrics.loc[metrics.Model.eq(model_name)]
            ax.plot(frame.Week_Start, frame[metric], color=COLORS[model_name], label=model_name, marker='o', markersize=3, linewidth=1.7)
        if metric == 'class_0_PR_AUC':
            frame = metrics.loc[metrics.Model.eq('Majority')]
            ax.plot(frame.Week_Start, frame.no_skill_class_0_PR_AUC, color='#4D8064', linestyle='--', linewidth=1.8, label='No-skill: test class-0 prevalence')
        if metric == 'balanced_accuracy':
            ax.axhline(.5, color='#4D8064', linestyle='--', linewidth=1.4, label='No-skill: 0.5')
        ax.set(title=title, ylabel=ylabel, ylim=(0, 1))
        format_axis(ax, test_weeks, partial_weeks)
        ax.legend(frameon=False, fontsize=9, loc='lower left', ncol=2)
        rendered_figures[name] = fig
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for c, color in [(0, '#174A75'), (1, '#B65B25')]:
        axes[0].plot(weeks, balance[f'Class_{c}_Share'], label=f'Class {c}', color=color, linewidth=1.8)
    axes[0].set(title='a  Orientation class balance', ylabel='Share', ylim=(0, 1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False)
    axes[1].plot(weeks, balance.Evaluated_Users, color='#174A75', linewidth=1.8, label='Dataset-eligible users')
    axes[1].plot(weeks, balance.Active_Labelled_Users, color='#737373', linestyle='--', linewidth=1.5, label='All valid labels')
    axes[1].set(title='b  Users eligible for structural-feature prediction', ylabel='Users', ylim=(0, None))
    axes[1].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axes[1].legend(frameon=False)
    for ax in axes:
        format_axis(ax, weeks, partial_weeks)
    fig.suptitle('Orientation balance and evaluated-user counts', fontsize=14)
    rendered_figures['rolling_origin_class_balance'] = fig
    best_macro = summary.loc[summary.Metric.eq('macro_F1')].sort_values('Mean', ascending=False).iloc[0]
    best_ap = summary.loc[summary.Metric.eq('class_0_PR_AUC')].sort_values('Mean', ascending=False).iloc[0]
    lr = metrics.loc[metrics.Model.eq('Logistic Regression')].set_index('Week_Start')
    maj = metrics.loc[metrics.Model.eq('Majority')].set_index('Week_Start')
    wins = {
        'logistic_macro_f1_above_majority_weeks': int(lr.macro_F1.gt(maj.macro_F1).sum()),
        'logistic_class_0_ap_above_prevalence_weeks': int(lr.class_0_PR_AUC.gt(lr.no_skill_class_0_PR_AUC).sum()),
        'logistic_balanced_accuracy_above_half_weeks': int(lr.balanced_accuracy.gt(.5).sum()),
    }
    audit = {
        'all_assertions_passed': True, 'features': FEATURES,
        'feature_event_timestamps_strictly_before_label_week': True,
        'exclusive_feature_cutoff_equals_label_monday': True,
        'historical_node_membership_verified_from_prior_interactions': True,
        'training_strictly_before_validation_and_test': True,
        'scaler_training_only_sample_count_and_means_verified': True,
        'hyperparameters_selected_by_validation_macro_f1_only': True,
        'threshold_fixed_a_priori': .5, 'test_weeks_chronological_and_paired': True,
        'no_duplicate_user_week_records': True, 'no_same_week_target_features': True,
        'raw_and_existing_input_sha256_unchanged': True,
        'input_hashes': hashes, 'seed': SEED, 'minimum_training_weeks': MIN_TRAIN_WEEKS,
        'C_grid': C_GRID, 'bootstrap_draws': BOOTSTRAPS,
    }
    assumptions = [
        'Target is a dataset binary orientation label, not verified sentiment. Only Date, User, and Target are read from master_tweets for labels. Binary labels are majority votes within each UTC Monday-start week; ties remain NA and are excluded.',
        'Users are normalized by trimming whitespace and leading @ and casefolding. One capitalization alias in master_tweets is merged; handles are identifiers, not verified stable people.',
        'Dataset eligibility requires a non-tied current-week label and a same-Week_Start historical feature row. Only the six named structural features enter logistic regression. Counts, ratios, metadata, current edges, and labels are never predictor columns.',
        'The input feature cutoff is exclusive and equals the prediction Monday; actual most recent historical interaction timestamps are separately audited to be strictly earlier. Reconstructed edges are read only for provenance checks, never used as contemporaneous predictors.',
        'Cold_Start_Count counts valid-labelled user-weeks lacking graph history, not unique users. This includes first observed tweet weeks and users with earlier tweets but no prior graph interactions; both subsets are reported. Tie_Count includes all tied active user-weeks, independently of graph history.',
        'Active_Labelled_Users counts all valid labels; Users_With_Historical_Features counts active users with graph features, including ties; Evaluated_Users excludes ties and missing history. Wilson intervals and class shares use this eligible cohort, which can differ substantially from all labelled users.',
        'First and last observed collection weeks are marked partial. This is based on observed minimum/maximum dates, not independently confirmed collection coverage. Interior weeks are not assumed complete platform coverage. Weeks with no eligible users have undefined shares.',
        'Expanding rolling origins require at least eight eligible training weeks. Validation is the immediately previous eligible week; training is all earlier eligible weeks excluding validation; test is the next eligible week. Training must contain both classes. No random temporal split or training/validation refit is used.',
        'StandardScaler fits only training rows. Logistic Regression uses class_weight=balanced, lbfgs, max_iter=2000, seed=42. C in [0.01,0.1,1,10,100] is selected by validation macro-F1; smaller C breaks exact ties. Class-1 probability threshold is fixed at 0.5 and never tuned on test.',
        'Majority predicts the training majority (tie -> class 0), with training class prevalence as its probability. Persistence uses the most recent strictly prior non-tied user label, including prior validation labels and prior labels without graph features; missing prior labels fall back to training majority/prevalence. Persistence probabilities are hard 0/1 for observed labels, not calibrated confidence.',
        'Class 0 is designated the detection class as requested even when it is not empirically the minority in the selected cohort. Both class F1 scores are reported. ROC_AUC and Brier_score use class-1 probabilities. Class-0 PR-AUC uses 1-P(class 1).',
        'PR-AUC is sklearn average precision (step-weighted area), not trapezoidal integration. No-skill class-0 AP equals that test week class-0 prevalence. Undefined precision/recall/F1 uses zero_division=0; when a test week has only one class, balanced accuracy, MCC and discrimination AUC metrics are NA.',
        'Aggregate means weight test weeks equally; standard deviation is sample SD (ddof=1). Percentile 95% bootstrap intervals use 10,000 week resamples with seed 42, shared across models/metrics. Paired comparisons subtract metrics on identical test weeks; Brier improvement has negative difference.',
        'Week bootstrap intervals treat weeks as exchangeable and do not account for temporal autocorrelation, repeated users, changing cohort composition, or orientation-label validity. Wilson intervals similarly describe counts without correcting user dependence.',
        'Consistent no-skill improvement is defined strictly as improvement in every evaluated test week, separately for macro-F1 versus majority, class-0 AP versus prevalence, and balanced accuracy versus 0.5. Win counts and paired mean-difference intervals are reported; they are descriptive, not proof of generalization.',
        'Repeated users can appear across train/validation/test because the task predicts future activity for previously observed users. No user-disjoint generalization is claimed. Majority vote discards tweet-level variation; missing graph-history users and tied labels are not evaluated.',
        'Structural features inherit extraction uncertainties, inferred replies and missing invisible interactions. Prior labels are available to persistence only after their week has ended. No neural networks are trained, no GNN performance claims are made, and existing raw, graph, GCN/EvolveGCN outputs are not modified.',
    ]
    report_info = {
        'weekly_labelled_user_records_including_ties': len(labels),
        'valid_majority_label_records': int(labels.Orientation_Label.notna().sum()),
        'ties': int(labels.Orientation_Label.isna().sum()), 'cold_start_excluded_user_weeks': int(balance.Cold_Start_Count.sum()),
        'cold_start_first_tweet_week': int(balance.Cold_Start_First_Tweet_Week_Count.sum()),
        'cold_start_previously_tweeted': int(balance.Cold_Start_Previously_Tweeted_Count.sum()),
        'dataset_eligible_records': len(dataset), 'rejected_tweet_rows': rejected_tweets,
        'rolling_test_weeks': len(test_weeks), 'first_test_week': str(test_weeks[0]), 'last_test_week': str(test_weeks[-1]),
        'best_macro_f1_baseline': best_macro.Model, 'best_mean_macro_f1': float(best_macro.Mean),
        'best_class_0_ap_baseline': best_ap.Model, 'best_mean_class_0_ap': float(best_ap.Mean),
        **wins,
        'logistic_exceeds_all_three_no_skill_checks_every_week': all(n == len(test_weeks) for n in wins.values()),
    }
    report = json.dumps(report_info, indent=2) + '\n\nTest-week class balance:\n' + balance.loc[balance.Week_Start.isin(test_weeks), ['Week_Start', 'Evaluated_Users', 'Class_0_Count', 'Class_1_Count', 'Class_0_Share']].to_string(index=False)
    report += '\n\nBaseline aggregate metrics:\n' + summary.loc[summary.Metric.isin(['macro_F1', 'class_0_PR_AUC', 'balanced_accuracy'])].to_string(index=False)
    report += '\n\nExact assumptions and limitations:\n' + '\n'.join(f'{i}. {a}' for i, a in enumerate(assumptions, 1))
    report += '\n\nLeakage audit:\n' + json.dumps(audit, indent=2)
    assert all(sha256(ROOT / p) == h for p, h in hashes.items()), 'Protected input changed'
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    def save_csv(frame, path):
        # Date-only week keys, full UTC timestamp provenance.
        out = frame.copy()
        for column in out.columns:
            if column.endswith('Week') or column == 'Week_Start':
                if isinstance(out[column].dtype, pd.DatetimeTZDtype):
                    out[column] = out[column].dt.strftime('%Y-%m-%d')
        with path.open('x', encoding='utf-8', newline='') as f:
            out.to_csv(f, index=False, na_rep='NA')
    for frame, name in [(labels, 'labels'), (dataset, 'dataset'), (balance, 'balance'), (metrics, 'metrics'),
                        (summary, 'summary'), (comparisons, 'paired'), (pd.DataFrame(protocol_rows), 'protocol'),
                        (pd.DataFrame(tuning_rows), 'tuning'), (predictions, 'predictions')]:
        save_csv(frame, paths[name])
    for name, fig in rendered_figures.items():
        for ext in ['png', 'svg']:
            with paths[name+'_'+ext].open('xb') as f:
                fig.savefig(f, format=ext, dpi=400, facecolor='white')
        plt.close(fig)
    # Check serialization and row accounting, preserving identifiers such as 000... handles.
    reread = pd.read_csv(paths['dataset'], dtype={'User': str})
    assert len(reread) == len(dataset) and not reread.duplicated(['Week_Start', 'User']).any()
    assert pd.to_datetime(reread.Feature_Last_Interaction_UTC, utc=True).lt(pd.to_datetime(reread.Week_Start, utc=True)).all()
    assert list(reread[FEATURES]) == FEATURES
    assert all(sha256(ROOT / p) == h for p, h in hashes.items())
    with paths['audit'].open('x', encoding='utf-8') as f:
        json.dump(audit, f, indent=2)
    with paths['report'].open('x', encoding='utf-8') as f:
        f.write(report + '\n')
    print(report, flush=True)


if __name__ == '__main__':
    main()
