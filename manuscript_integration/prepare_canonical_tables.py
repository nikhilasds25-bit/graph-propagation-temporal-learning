"""Typeset existing evidence only; no statistics, experiments or result-file edits."""
from pathlib import Path
import hashlib,json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
TABLES=ROOT/'results/tables'
FILES=['all_forecasting_models_reconciled.csv','matching_sensitivity_comparison.csv',
       'target_textblob_agreement_recheck.csv','typed_layer_network_statistics.csv',
       'layer_orientation_permutation_tests.csv','time_respecting_reachability_summary.csv',
       'secondary_reach_simulation_results.csv','competition_exposure_overlap.csv',
       'spectral_threshold_audit.csv','reviewer_correction_master_report.md',
       'reviewer_correction_issue_index.csv']

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def write(name,text):
    with (OUT/name).open('x',encoding='utf-8') as f: f.write(text+'\n')

def esc(v):
    s=str(v)
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]: s=s.replace(a,b)
    return s

def val(v,d=4): return 'NA' if pd.isna(v) else f'{float(v):.{d}f}'

def ci(r,col,d=4):
    if pd.isna(r.get(col)): return 'NA'
    m=val(r[col],d); lo=r.get(col+'_ci95_low'); hi=r.get(col+'_ci95_high')
    if pd.isna(lo) or pd.isna(hi): return m+r'\newline (CI unavailable)'
    return m+r'\newline ['+val(lo,d)+', '+val(hi,d)+']'

def main():
    paths=[TABLES/n for n in FILES]+sorted((ROOT/'results/figures/reviewer_corrected').glob('*'))
    protected={str(p.relative_to(ROOT)):digest(p) for p in paths if p.is_file()}
    write('canonical_input_hashes.json',json.dumps(protected,indent=2))
    f=pd.read_csv(TABLES/'all_forecasting_models_reconciled.csv')
    # Keep every saved cohort; no averaging across populations or inference units.
    rows=[]
    columns=['macro_F1','balanced_accuracy','MCC','class_0_PR_AUC','ROC_AUC','Brier_score']
    latex=[r'% Requires booktabs, longtable, array, pdflscape; place inside landscape.',
           r'\begin{landscape}',r'\begingroup\small\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.3}',
           r'\begin{longtable}{p{3.8cm}p{2.2cm}*{6}{p{2.25cm}}}',
           r'\caption{Weekly forecasting and cohort-specific comparison results. Entries are saved means with 95\% confidence intervals. No results are pooled across cohorts.}\label{tab:forecasting}\\',
           r'\toprule Model & Cohort & Macro-F1 & Balanced accuracy & MCC & Class-0 AP & AUROC & Brier score\\\midrule',
           r'\endfirsthead',r'\toprule Model & Cohort & Macro-F1 & Balanced accuracy & MCC & Class-0 AP & AUROC & Brier score\\\midrule',r'\endhead',
           r'\midrule\multicolumn{8}{r}{Continued on next page}\\\endfoot',r'\bottomrule\endlastfoot']
    for _,r in f.iterrows():
        cohort=str(r.cohort)
        label='Eligible weeks' if cohort.startswith('All eligible') else 'History available' if cohort=='History available' else 'Monthly node task'
        model=str(r.model)
        if model=='History Logistic': model='History-only logistic'
        if model=='Original GRU': model='Node-only GRU (original GRU)'
        # Protocol identifiers remain in the provenance JSON, not revision narration.
        if label=='Monthly node task':
            model=model.replace('_',' ')
            label+='; '+('class-balanced' if 'imbalance_corrected' in cohort else 'unweighted')
        latex.append(esc(model)+' & '+esc(label)+' & '+' & '.join(ci(r,c) for c in columns)+r'\\')
        rows.append({k:(None if pd.isna(v) else v) for k,v in r.to_dict().items()})
    latex += [r'\end{longtable}',r'\endgroup',r'\end{landscape}',
              r'% Text must cite Table~\ref{tab:forecasting}. Cohort counts and information boundaries are separate.']
    write('forecasting_results_table.tex','\n'.join(latex)); write('forecasting_table_provenance.json',json.dumps(rows,indent=2))
    primary=f[f.cohort.str.startswith('All eligible')].iloc[0]
    pop=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Analysis populations and information boundaries}\label{tab:populations}',
         r'\begin{tabular}{p{3.1cm}p{2.4cm}p{6.8cm}}\toprule Population & Size & Analysis and information boundary\\\midrule',
         r'Literal-matching WCC & 17,598 users; 29,715 directed edges & Static network structure and propagation simulation. Seed-orientation eligibility uses observations through November 2022; the graph is a static full-period interaction comparator.\\',
         r'Normalized matching sensitivity & 17,636 users; 29,825 directed edges & Matching sensitivity only; completed simulation and forecasting cohorts are not reassigned.\\',
         r'Reconstructed temporal endpoints & 60,219 accounts & Typed interaction networks and time-respecting reachability over reconstructed timestamped events; accounts need not belong to the literal-matching WCC.\\',
         'Weekly forecasting & '+f'{int(primary.test_user_weeks):,} eligible user-weeks; {int(primary.test_unique_users):,} users'+r' & Fifteen test weeks, 19 December 2022--27 March 2023. Non-tie weekly operational labels and strictly earlier structural inputs; cold-start user-weeks excluded by eligibility.\\',
         r'\bottomrule\end{tabular}',r'\end{table}']
    write('analysis_populations_table.tex','\n'.join(pop))
    t=pd.read_csv(TABLES/'time_respecting_reachability_summary.csv')
    t=t[(t.layer=='pooled')&(t.metric=='temporal_static_ratio')]
    tex=[r'\begin{table}[htbp]\centering\small',r'\caption{Temporal/static descendant ratio by seed stratum in the reconstructed endpoint universe. Ratios exclude static sinks; intervals are 95\% seed-bootstrap intervals for the mean.}\label{tab:temporal}',
         r'\begin{tabular}{p{2.0cm}p{3.2cm}rrp{3.2cm}}\toprule Direction & Stratum & Defined ratios & Static sinks & Mean [95\% CI]\\\midrule']
    for _,r in t.iterrows(): tex.append(esc(r.direction.replace('_',' '))+' & '+esc(r.stratum.replace('_',' '))+' & '+str(int(r['n']))+' & '+str(int(r.zero_static_seeds))+' & '+val(100*r['mean'],2)+r'\% ['+val(100*r.ci95_low,2)+', '+val(100*r.ci95_high,2)+r']\%\\')
    tex += [r'\bottomrule\end{tabular}\end{table}']; write('temporal_reach_table.tex','\n'.join(tex))
    s=pd.read_csv(TABLES/'secondary_reach_simulation_results.csv'); e=pd.read_csv(TABLES/'competition_exposure_overlap.csv')
    select=s[(s.configuration=='full_balanced')&(s.direction=='recorded')].copy()
    exp=e[(e.configuration=='full_balanced')&(e.direction=='recorded')].copy()
    numerical=dict(temporal=t.to_dict('records'),balanced_full_simulation=select.to_dict('records'),balanced_full_exposure=exp.to_dict('records'))
    write('exact_numerical_evidence.json',json.dumps(numerical,indent=2))
    assert all(digest(ROOT/n)==h for n,h in protected.items())
    write('preparation_status.md','# Manuscript integration preparation\n\nCanonical tables have been typeset from saved results only. No experiment, model fit or analysis-output write was performed. The current manuscript source is required before authorship, bibliography, section integration, compilation and the final Overleaf ZIP can be completed. These files are integration fragments, not a substitute final manuscript. Input hashes are in canonical_input_hashes.json; all listed files were verified unchanged after preparation. The forecasting fragment preserves all saved cohorts and unavailable values; final placement/widths must be checked against the supplied manuscript class.')
    print('Prepared manuscript-only table fragments. Canonical evidence unchanged.')

if __name__=='__main__': main()
