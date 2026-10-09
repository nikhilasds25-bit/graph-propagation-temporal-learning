"""Reconstruct text-supported directed interactions. Never overwrite outputs."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import random
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
MENTION = re.compile(r'(?<![A-Za-z0-9_@])@([A-Za-z0-9_]{1,15})(?![A-Za-z0-9_])')
RETWEET = re.compile(r'(?<![A-Za-z0-9_])RT\s+@([A-Za-z0-9_]{1,15})(?![A-Za-z0-9_])', re.I)
LEADING = re.compile(r'^\s*@([A-Za-z0-9_]{1,15})(?![A-Za-z0-9_])')


def key(value):
    return str(value).strip().lstrip('@').casefold()


def interactions(text):
    """One event per target per tweet: retweet > inferred reply > mention."""
    result = {key(m.group(1)): 'mention' for m in MENTION.finditer(text)}
    lead = LEADING.match(text)
    if lead:
        result[key(lead.group(1))] = 'reply_inferred'
    for m in RETWEET.finditer(text):
        result[key(m.group(1))] = 'retweet'
    return result


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    raw = [ROOT / 'data/raw/master_tweets.csv', ROOT / 'data/raw/network_edges.csv']
    outputs = {
        'edges': ROOT / 'data/processed/reconstructed_temporal_edges.csv',
        'summary': ROOT / 'data/processed/reconstructed_temporal_edges_summary.csv',
        'weekly': ROOT / 'data/processed/reconstructed_weekly_edge_counts.csv',
        'figure': ROOT / 'results/figures/reconstructed_weekly_edge_counts.png',
        'vector': ROOT / 'results/figures/reconstructed_weekly_edge_counts.svg',
        'report': ROOT / 'results/tables/reconstructed_temporal_edges_report.txt',
    }
    collisions = [str(p) for p in outputs.values() if p.exists()]
    if collisions:
        raise FileExistsError('No files written; existing outputs: ' + ', '.join(collisions))
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in raw}
    tweets, supplied = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in raw]
    print('Exact tweet columns:', tweets.columns.tolist())
    print('Exact network columns:', supplied.columns.tolist())
    assert tweets.columns.tolist() == ['Date', 'User', 'Tweet', 'Likes', 'Retweets', 'Target', 'Year', 'Month', 'Day']
    assert supplied.columns.tolist() == ['Source', 'Target', 'Weight']
    # Most frequent supplied spelling per case-insensitive key; lexical tie break.
    spellings = {}
    for value in pd.concat([supplied.Source, supplied.Target]):
        spellings.setdefault(key(value), Counter())[value.strip().lstrip('@')] += 1
    canonical = {k: sorted(c, key=lambda s: (-c[s], s))[0] for k, c in spellings.items()}
    display = lambda k: canonical.get(k, k)
    records = []
    no_interactions = 0
    for row in tweets.itertuples(index=False):
        found = interactions(row.Tweet)
        no_interactions += not bool(found)
        for target, kind in found.items():
            records.append((display(key(row.User)), display(target), row.Date, kind))
    edges = pd.DataFrame(records, columns=['Source', 'Target', 'Timestamp', 'Interaction_Type'])
    counts = Counter((key(s), key(t)) for s, t in zip(edges.Source, edges.Target))
    weights = pd.to_numeric(supplied.Weight, errors='coerce')
    network = pd.DataFrame({'Source': supplied.Source.map(key), 'Target': supplied.Target.map(key), 'Weight': weights})
    aggregated = network.groupby(['Source', 'Target']).Weight.sum(min_count=1)
    reconstructed_pairs = set(counts)
    supplied_pairs = set(aggregated.index)
    overlap = reconstructed_pairs & supplied_pairs
    dates = pd.to_datetime(edges.Timestamp, utc=True, errors='coerce')
    all_dates = pd.to_datetime(tweets.Date, utc=True, errors='coerce')
    week_start = lambda d: d.dt.tz_localize(None).dt.normalize() - pd.to_timedelta(d.dt.dayofweek, unit='D')
    first, last = week_start(all_dates).min(), week_start(all_dates).max()
    weeks = pd.date_range(first, last, freq='7D')
    weekly = pd.crosstab(week_start(dates), edges.Interaction_Type).reindex(weeks, fill_value=0)
    weekly = weekly.reindex(columns=['mention', 'reply_inferred', 'retweet'], fill_value=0)
    weekly['Total'] = weekly.sum(axis=1)
    weekly.index.name = 'Week_Start_UTC'
    metrics = []
    def add(name, value, definition):
        metrics.append({'Metric': name, 'Value': value, 'Definition': definition})
    ratio = lambda n, d: n / d if d else float('nan')
    add('tweet_rows', len(tweets), 'Input tweet records; no row deduplication')
    add('tweets_without_interactions', no_interactions, 'No recoverable text interaction')
    add('reconstructed_interactions', len(edges), 'One target event per input tweet')
    for kind, n in edges.Interaction_Type.value_counts().items():
        add(kind + '_interactions', int(n), 'Retweet > inferred reply > mention priority')
    add('reconstructed_unique_edges', len(counts), 'Distinct directed case-insensitive Source,Target pairs')
    add('supplied_unique_edges', len(supplied_pairs), 'Distinct directed normalized network pairs')
    add('unique_edge_overlap', len(overlap), 'Size of intersection of directed pair sets')
    add('precision_style_overlap', ratio(len(overlap), len(counts)), 'Intersection / reconstructed unique pairs; supplied network is not ground truth')
    add('coverage', ratio(len(overlap), len(supplied_pairs)), 'Intersection / supplied unique pairs')
    add('jaccard_overlap', ratio(len(overlap), len(reconstructed_pairs | supplied_pairs)), 'Intersection / union of unique directed pairs')
    add('interaction_weighted_overlap', ratio(sum(counts[p] for p in overlap), len(edges)), 'Reconstructed events whose directed pair is supplied / all reconstructed events')
    for label, pairs in [('matched', sorted(overlap)), ('supplied_zero_filled', sorted(supplied_pairs))]:
        data = pd.DataFrame({'Count': [counts.get(p, 0) for p in pairs], 'Weight': [aggregated.loc[p] for p in pairs]})
        data = data[np.isfinite(data.Weight)]
        add('correlation_' + label + '_n', len(data), 'Finite aggregate Weight pairs; missing reconstructed counts set to zero only for supplied_zero_filled')
        for method in ['pearson', 'spearman']:
            r = data.Count.corr(data.Weight, method=method) if len(data) > 1 and data.Count.nunique() > 1 and data.Weight.nunique() > 1 else float('nan')
            add(method + '_' + label, r, 'Count vs summed supplied Weight; descriptive association, not temporal validation')
    add('invalid_tweet_dates', int(all_dates.isna().sum()), 'Original Date retained; invalid dates excluded from weekly figure')
    add('duplicate_tweet_records', int(tweets.duplicated(['Date', 'User', 'Tweet']).sum()), 'Repeated Date,User,Tweet rows retained because no tweet ID exists')
    add('self_interactions', sum(s == t for s, t in zip(edges.Source.map(key), edges.Target.map(key))), 'Self mentions retained')
    add('invalid_network_handle_endpoints', sum((~supplied[c].str.fullmatch(r'@?[A-Za-z0-9_]{1,15}')).sum() for c in ['Source', 'Target']), 'Nonstandard supplied identifiers retained for comparison; text extraction uses standard handles')
    add('invalid_weights', int(weights.isna().sum()), 'Non-numeric weights excluded from correlations')
    add('supplied_duplicate_normalized_pairs', len(network) - len(aggregated), 'Duplicate pair weights summed')
    add('seed', SEED, 'Python and NumPy seeded; no random sampling used')
    assumptions = [
        'Source is tweet User; Target is a recoverable text username; direction is author to mentioned/retweeted/replied-to user.',
        'Usernames are stripped of whitespace and leading @, compared case-insensitively. Output uses most frequent supplied-network spelling (lexical tie break); unmatched handles use lowercase.',
        'Extract ASCII Twitter handles of 1–15 letters, digits or underscores. Mention boundaries exclude email-like tokens and overlong handles.',
        'RT is a standalone, case-insensitive token followed by whitespace and @handle. Embedded RT patterns are included; textual quotations cannot be distinguished from actual retweets.',
        'Only the first leading @handle is reply_inferred. Other leading handles remain mentions. Leading mentions do not prove replies; no explicit reply or retweet metadata exists.',
        'One event per target per tweet; retweet takes priority over reply_inferred, which takes priority over mention. Repeated mentions to the same target do not multiply counts.',
        'Tweet Target contains binary values 0.0 and 1.0 and is not a reply username. Likes and Retweets are engagement totals, not actor identities.',
        'Date strings are preserved exactly as Timestamp. Weekly aggregation uses UTC Monday 00:00 through next Monday exclusive; missing weeks within the full tweet date span are zero-filled. Boundary weeks may be partial.',
        'Self interactions and repeated input rows are retained. No tweet ID is available to identify duplicate posts. Invalid dates, if any, remain in the edge CSV but are excluded from weekly counts.',
        'Mentions inside quoted/retweeted text are attributed to the row author because attribution cannot be recovered. Native retweets, replies, quote tweets, deleted/truncated text and interactions without visible handles may be missed.',
        'Overlap uses normalized directed unique pairs. Precision-style overlap is intersection/reconstructed pairs; coverage is intersection/supplied pairs. The supplied network is a comparator, not verified ground truth.',
        'Pair interaction counts sum all types. Supplied duplicate pair Weight values are summed. Pearson and Spearman are reported on matched pairs and on all supplied pairs with absent reconstructed counts zero-filled. Non-finite weights are excluded; undefined correlations are NA.',
        'The supplied network has no timestamps or documented Weight semantics, so correlations and overlap do not establish temporal completeness or causality.',
        'No random sampling is used; Python and NumPy seeds are 42. Raw input SHA-256 hashes are checked before and after execution. Existing output paths cause failure before writing.',
    ]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    ax.plot(weeks, weekly.Total, color='#174A75', linewidth=1.8)
    ax.fill_between(weeks, weekly.Total, color='#174A75', alpha=.12)
    ax.set(title='Weekly reconstructed user interactions', xlabel='Week beginning Monday (UTC)', ylabel='Reconstructed interaction events')
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=.22, linewidth=.6)
    fig.suptitle('One directed event per target per tweet; all interaction types combined', fontsize=9, color='#555555')
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    with outputs['edges'].open('x', encoding='utf-8', newline='') as f:
        edges.to_csv(f, index=False)
    with outputs['summary'].open('x', encoding='utf-8', newline='') as f:
        pd.DataFrame(metrics).to_csv(f, index=False, na_rep='NA')
    with outputs['weekly'].open('x', encoding='utf-8', newline='') as f:
        weekly.to_csv(f, date_format='%Y-%m-%d')
    for name in ['figure', 'vector']:
        with outputs[name].open('xb') as f:
            fig.savefig(f, format='png' if name == 'figure' else 'svg', dpi=400, facecolor='white')
    plt.close(fig)
    assert weekly.Total.sum() == dates.notna().sum()
    assert all(sha256(p) == hashes[str(p.relative_to(ROOT))] for p in raw), 'Raw input changed during run'
    report = '\n'.join([
        'Exact input columns:', json.dumps(tweets.columns.tolist()), json.dumps(supplied.columns.tolist()),
        'Results:', pd.DataFrame(metrics).to_string(index=False),
        'Raw SHA-256 hashes (unchanged):', json.dumps(hashes, indent=2),
        'Exact assumptions and limitations:', *[f'{i}. {a}' for i, a in enumerate(assumptions, 1)],
    ])
    with outputs['report'].open('x', encoding='utf-8') as f:
        f.write(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
