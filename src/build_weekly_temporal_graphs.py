"""Build weekly graphs and strictly lagged cumulative structural features."""
from pathlib import Path
import hashlib
import random
import json

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data/processed/reconstructed_temporal_edges.csv'
SEED = 42


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def monday(dates):
    return dates.dt.normalize() - pd.to_timedelta(dates.dt.dayofweek, unit='D')


def graph_from_pairs(pairs):
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(pairs.itertuples(index=False, name=None))
    return graph


def graph_summary(graph, week, events):
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    nonself = [(s, t) for s, t in graph.edges if s != t]
    return {
        'Week_Start': week.strftime('%Y-%m-%d'), 'Nodes': n,
        'Directed_Edges': m, 'Interaction_Events': events,
        'Density': len(nonself) / (n * (n - 1)) if n > 1 else 0.0,
        'Reciprocity': sum(graph.has_edge(t, s) for s, t in nonself) / len(nonself) if nonself else float('nan'),
        'Largest_WCC_Nodes': max(map(len, nx.weakly_connected_components(graph)), default=0),
        'Largest_SCC_Nodes': max(map(len, nx.strongly_connected_components(graph)), default=0),
        'Mean_In_Degree': m / n if n else 0.0,
        'Mean_Out_Degree': m / n if n else 0.0,
        'Mean_In_Strength': events / n if n else 0.0,
        'Mean_Out_Strength': events / n if n else 0.0,
    }


def historical_features(history, week):
    if not history:
        return []
    pr = nx.pagerank(history, alpha=.85, weight='weight', tol=1e-10, max_iter=1000)
    reverse = nx.pagerank(history.reverse(copy=False), alpha=.85, weight='weight', tol=1e-10, max_iter=1000)
    rows = []
    for node in sorted(history):
        rows.append({
            'Week_Start': week.strftime('%Y-%m-%d'), 'User': node,
            'Feature_Window_End_Exclusive': week.isoformat(),
            'in_degree': history.in_degree(node), 'out_degree': history.out_degree(node),
            'in_strength': history.in_degree(node, weight='weight'),
            'out_strength': history.out_degree(node, weight='weight'),
            'PageRank': pr[node], 'reverse_PageRank': reverse[node],
        })
    return rows


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    before = digest(INPUT)
    data = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
    assert list(data) == ['Source', 'Target', 'Timestamp', 'Interaction_Type']
    dates = pd.to_datetime(data.Timestamp, format='mixed', utc=True, errors='coerce')
    identifiers_valid = data.Source.str.strip().ne('') & data.Target.str.strip().ne('')
    valid = dates.notna() & identifiers_valid
    rejected = data.loc[~valid].copy()
    rejected['Reason'] = np.where(dates.loc[~valid].isna(), 'invalid_timestamp', 'empty_endpoint')
    events = data.loc[valid].copy()
    if events.empty:
        raise ValueError('No valid timestamped interactions; no outputs written')
    events['Week'] = monday(dates.loc[valid])
    weeks = pd.date_range(events.Week.min(), events.Week.max(), freq='7D')
    # Validate the existing spelling convention instead of silently splitting identities.
    endpoints = pd.concat([events.Source, events.Target]).drop_duplicates()
    if endpoints.str.strip().ne(endpoints).any() or endpoints.str.startswith('@').any():
        raise ValueError('Input handle normalization differs from reconstruction')
    if endpoints.str.casefold().duplicated().any():
        raise ValueError('Case aliases in input require an explicit canonical mapping')
    paths = {
        'summary': ROOT / 'results/tables/weekly_temporal_graph_summary.csv',
        'features': ROOT / 'data/processed/weekly_temporal_node_features.csv',
        'types': ROOT / 'results/tables/weekly_temporal_interaction_types.csv',
        'rejected': ROOT / 'results/tables/weekly_temporal_rejected_events.csv',
        'report': ROOT / 'results/tables/weekly_temporal_graph_validation.txt',
        'png': ROOT / 'results/figures/weekly_graph_structure.png',
        'svg': ROOT / 'results/figures/weekly_graph_structure.svg',
    }
    edge_paths = {week: ROOT / f'data/processed/temporal_graphs/edges_{week:%Y-%m-%d}.csv' for week in weeks}
    collisions = [str(p) for p in [*paths.values(), *edge_paths.values()] if p.exists()]
    if collisions:
        raise FileExistsError('No files written. Existing outputs: ' + ', '.join(collisions))
    history = nx.DiGraph()
    summary, features, snapshots = [], [], {}
    grouped = dict(tuple(events.groupby('Week', sort=True)))
    for week in weeks:
        # Compute BEFORE adding this week's interactions; no future node universe.
        features.extend(historical_features(history, week))
        current = grouped.get(week, events.iloc[:0])
        pairs = current.groupby(['Source', 'Target'], sort=True).size().reset_index(name='Weight')
        graph = graph_from_pairs(pairs)
        assert sum(nx.get_edge_attributes(graph, 'weight').values()) == len(current)
        assert graph.number_of_edges() == len(pairs)
        summary.append(graph_summary(graph, week, len(current)))
        snapshots[week] = pairs
        for s, t, weight in pairs.itertuples(index=False, name=None):
            history.add_edge(s, t, weight=history.get_edge_data(s, t, {}).get('weight', 0) + weight)
        print(f'{week:%Y-%m-%d}: {len(graph):,} nodes, {len(pairs):,} edges, {len(current):,} events', flush=True)
    summary = pd.DataFrame(summary)
    feature_columns = ['Week_Start', 'User', 'Feature_Window_End_Exclusive', 'in_degree', 'out_degree', 'in_strength', 'out_strength', 'PageRank', 'reverse_PageRank']
    features = pd.DataFrame(features, columns=feature_columns)
    types = pd.crosstab(events.Week, events.Interaction_Type).reindex(weeks, fill_value=0)
    types.index.name = 'Week_Start'
    assert summary.Interaction_Events.sum() == len(events) == types.to_numpy().sum()
    assert not features.duplicated(['Week_Start', 'User']).any()
    for week, f in features.groupby('Week_Start'):
        cutoff = pd.Timestamp(week, tz='UTC')
        prior = events[events.Week < cutoff]
        assert set(f.User) == set(prior.Source) | set(prior.Target)
        assert f.in_strength.sum() == len(prior) == f.out_strength.sum()
        assert np.isclose(f.PageRank.sum(), 1) and np.isclose(f.reverse_PageRank.sum(), 1)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ['#174A75', '#B65B25', '#4D8064', '#8665A0']
    ax = axes[0, 0]
    for col, color in zip(['Nodes', 'Directed_Edges'], colors):
        ax.plot(weeks, summary[col], label=col.replace('_', ' '), color=color, linewidth=1.8)
    ax.set(title='a  Weekly nodes and directed edges', ylabel='Count')
    ax.legend(frameon=False)
    ax = axes[0, 1]
    ax.plot(weeks, summary.Density, color=colors[0], label='Density', linewidth=1.8)
    ax.set(title='b  Density and reciprocity', ylabel='Density (excluding self loops)')
    ax.tick_params(axis='y', labelcolor=colors[0])
    right = ax.twinx()
    right.plot(weeks, summary.Reciprocity, color=colors[1], linestyle='--', label='Reciprocity', linewidth=1.8)
    right.set_ylabel('Reciprocity (excluding self loops)', color=colors[1])
    right.tick_params(axis='y', labelcolor=colors[1])
    ax.legend([ax.lines[0], right.lines[0]], ['Density', 'Reciprocity'], frameon=False)
    ax = axes[1, 0]
    for col, label, color in zip(['Largest_WCC_Nodes', 'Largest_SCC_Nodes'], ['Largest WCC', 'Largest SCC'], colors):
        ax.plot(weeks, summary[col], label=label, color=color, linewidth=1.8)
    ax.set(title='c  Largest connected components', ylabel='Nodes')
    ax.legend(frameon=False)
    ax = axes[1, 1]
    names = sorted(types.columns)
    ax.stackplot(weeks, *[types[n].to_numpy() for n in names], labels=[n.replace('_', ' ') for n in names], colors=colors[:len(names)], alpha=.9)
    ax.set(title='d  Weekly interactions by type', ylabel='Interaction events')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    for ax in axes.flat:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_xlabel('Week beginning Monday (UTC)')
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.2, linewidth=.6)
    right.set_ylim(bottom=0)
    for ax in [axes[0, 0], axes[1, 0], axes[1, 1]]:
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    fig.suptitle('Weekly directed graph structure and interaction activity', fontsize=14)
    assumptions = [
        'Timestamp parsing uses pandas mixed-format parsing with errors coerced and UTC conversion. Naive timestamps, if present, are assumed UTC. Invalid dates or blank endpoints are excluded and saved with reasons.',
        'Weeks begin Monday 00:00 UTC and end at the next Monday exclusive. Every week from first through last valid interaction is included. Boundary weeks can be partial. Empty weeks have header-only edge files.',
        'Snapshot nodes are endpoints active in that week; no globally fixed future node universe or inactive historical isolates is added. Source -> Target direction and canonical handle spelling are preserved and checked for case aliases.',
        'Each snapshot is a weighted DiGraph. Weight counts input interaction rows per directed pair across all types. Self loops are retained. Repeated input rows remain separate interaction events.',
        'Directed_Edges and degrees include self loops; each loop contributes one in-degree and one out-degree. Density is non-self directed edges / N(N-1), or zero for N<2. Reciprocity is reciprocated non-self directed edges / non-self directed edges, and NA when that denominator is zero.',
        'WCC and SCC sizes count nodes; empty graphs have size zero. Mean degree is directed edges / active nodes; mean strength is events / active nodes; both are zero for empty graphs.',
        'Feature Week_Start denotes the week to predict. Features use a cumulative historical graph containing timestamps strictly before that Monday. Its weights sum prior events; degree counts distinct historical neighbors. Current-week snapshot statistics are descriptive and must not be used to predict that same week.',
        'Feature rows include only users observed strictly before the prediction week, including historically inactive users. The first week has no feature rows. Newly appearing users have no prior feature row; cold-start handling is left to future model design. No future node identities are used.',
        'PageRank and reverse_PageRank are weight-aware on the prior cumulative graph and its edge reversal, alpha=0.85, tol=1e-10, max_iter=1000, uniform teleportation and dangling distribution. They are deterministic. Python/NumPy seeds are 42; no stochastic sampling is performed.',
        'No labels, sentiment values, tweet Target columns, future labels, or same-week target-derived features are read or created. Target here is solely the destination username in the interaction edge list.',
        'Interaction_Type distinctions inherit reconstruction uncertainty: replies are inferred from leading handles, RT patterns may be quoted, and native or invisible interactions may be absent. The supplied reconstruction is not a complete record of platform activity.',
        'Counts, per-week weights, component sizes, historical node availability, PageRank normalization, feature strength totals, and exact time cutoffs are validated. Files are created exclusively; collisions fail before any output writes. The input SHA-256 is verified unchanged.',
        'The four figure panels describe contemporaneous snapshots, not prediction features. No GCN, EvolveGCN, or other model is trained. No raw file is read or modified.',
    ]
    info = {
        'weekly_snapshots': len(weeks), 'first_week': f'{weeks[0]:%Y-%m-%d}', 'last_week': f'{weeks[-1]:%Y-%m-%d}',
        'average_nodes_per_week': float(summary.Nodes.mean()), 'average_edges_per_week': float(summary.Directed_Edges.mean()),
        'empty_weeks': summary.loc[summary.Interaction_Events.eq(0), 'Week_Start'].tolist(),
        'valid_interaction_events': len(events), 'rejected_events': len(rejected), 'feature_rows': len(features),
        'self_loop_events': int(events.Source.eq(events.Target).sum()), 'input_sha256': before,
    }
    report = json.dumps(info, indent=2) + '\n\nExact assumptions and limitations:\n' + '\n'.join(f'{i}. {a}' for i, a in enumerate(assumptions, 1))
    assert digest(INPUT) == before, 'Input changed during processing'
    for p in [*paths.values(), *edge_paths.values()]:
        p.parent.mkdir(parents=True, exist_ok=True)
    def save_csv(frame, path, index=False):
        with path.open('x', encoding='utf-8', newline='') as f:
            frame.to_csv(f, index=index, na_rep='NA', date_format='%Y-%m-%d')
    for week, pairs in snapshots.items():
        save_csv(pairs, edge_paths[week])
    save_csv(summary, paths['summary'])
    save_csv(features, paths['features'])
    save_csv(types, paths['types'], index=True)
    save_csv(rejected, paths['rejected'])
    for name in ['png', 'svg']:
        with paths[name].open('xb') as f:
            fig.savefig(f, format=name, dpi=400, facecolor='white')
    plt.close(fig)
    # Validate serialized snapshot files against independently grouped input.
    for week, path in edge_paths.items():
        saved = pd.read_csv(path, dtype={'Source': str, 'Target': str})
        assert list(saved) == ['Source', 'Target', 'Weight']
        pd.testing.assert_frame_equal(saved, snapshots[week], check_dtype=False)
    saved_features = pd.read_csv(paths['features'])
    assert len(saved_features) == len(features)
    assert digest(INPUT) == before
    with paths['report'].open('x', encoding='utf-8') as f:
        f.write(report + '\n\nAll construction and serialized-output validations passed.\n')
    print(report, flush=True)
    print('All construction and serialized-output validations passed.', flush=True)


if __name__ == '__main__':
    main()
