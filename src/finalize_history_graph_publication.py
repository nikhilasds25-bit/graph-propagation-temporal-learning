"""New forest layout and metric-specific findings, preserving original reports."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_history_graph_supplement import ROOT,PAIRS,STAT_METRICS,MODELS

tables=ROOT/'results/tables';figures=ROOT/'results/figures'
paths=[figures/f'history_graph_incremental_forest_publication.{ext}' for ext in ['png','svg']]
manifest=tables/'history_graph_publication_figure_manifest.csv'
findings=tables/'history_graph_findings_addendum.txt'
assert not any(p.exists() for p in [*paths,manifest,findings])
pairs=pd.read_csv(tables/'history_graph_paired_comparisons.csv')
blocks=pd.read_csv(tables/'history_graph_block_bootstrap.csv')
tests=pd.read_csv(tables/'history_graph_holm_tests.csv')
summary=pd.read_csv(tables/'history_graph_summary.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
short={MODELS[0]:'History LR',MODELS[1]:'History MLP',MODELS[2]:'History reversed GAT','Persistence':'Persistence','Reversed GAT (no history)':'GAT without history'}
for ax,metric in zip(axes.flat,STAT_METRICS):
    for j,(a,b) in enumerate(PAIRS):
        r=pairs[pairs.Cohort.eq('All eligible')&pairs.Model_A.eq(a)&pairs.Model_B.eq(b)&pairs.Metric.eq(metric)].iloc[0]
        for offset,method,color in [(-.15,'Ordinary','#174A75'),(0,'Block 3','#27856F'),(.15,'Block 4','#B65B25')]:
            if method=='Ordinary':lo,hi=r.Ordinary_95_Lower,r.Ordinary_95_Upper
            else:
                length=int(method[-1]);q=blocks[blocks.Cohort.eq('All eligible')&blocks.Model_A.eq(a)&blocks.Model_B.eq(b)&blocks.Metric.eq(metric)&blocks.Block_Length.eq(length)].iloc[0]
                lo,hi=q.Moving_Block_95_Lower,q.Moving_Block_95_Upper
            ax.plot([lo,hi],[j+offset,j+offset],color=color,linewidth=2,label=method if j==0 else None)
            ax.plot(r.Mean_A_Minus_B,j+offset,'o',color=color,markersize=4)
    ax.set_yticks(range(5),[short[a]+' − '+short[b] for a,b in PAIRS])
    ax.set_ylim(4.7,-.95)  # Reserve a real legend band above the first comparison.
    ax.axvline(0,color='#777777',linestyle='--');ax.grid(axis='x',alpha=.2)
    ax.set(title=metric.replace('_',' '),xlabel='Paired mean difference')
    ax.legend(frameon=False,ncol=3,fontsize=8,loc='upper left')
fig.suptitle('Historical-label graph contribution: ordinary and moving-block 95% CIs',fontsize=14)
for path in paths:
    with path.open('xb') as f:fig.savefig(f,format=path.suffix[1:],dpi=400,facecolor='white')
plt.close(fig)
frame=pd.read_csv(tables/'history_graph_figure_manifest.csv')
frame.loc[frame.Figure.eq('C'),'PNG']=str(paths[0].relative_to(ROOT));frame.loc[frame.Figure.eq('C'),'SVG']=str(paths[1].relative_to(ROOT))
with manifest.open('x',encoding='utf-8',newline='') as f:frame.to_csv(f,index=False)
text='Metric-specific supplement to history_graph_report.txt. Original report and figures are retained.\n\n'
for metric in STAT_METRICS:
    best=summary[summary.Cohort.eq('All eligible')&summary.Metric.eq(metric)].sort_values('Mean',ascending=False).iloc[0]
    text+=f'Highest descriptive mean {metric}: {best.Model}, {best.Mean:.6f}.\n'
text+='\nPredeclared primary comparisons, all eligible:\n'
for r in tests[tests.Cohort.eq('All eligible')].itertuples():
    q=pairs[pairs.Cohort.eq(r.Cohort)&pairs.Model_A.eq(r.Model_A)&pairs.Model_B.eq(r.Model_B)&pairs.Metric.eq(r.Metric)].iloc[0]
    bs=blocks[blocks.Cohort.eq(r.Cohort)&blocks.Model_A.eq(r.Model_A)&blocks.Model_B.eq(r.Model_B)&blocks.Metric.eq(r.Metric)]
    positive=q.Ordinary_95_Lower>0 and bs.Moving_Block_95_Lower.gt(0).all() and r.Holm_Adjusted_P<.05
    negative=q.Ordinary_95_Upper<0 and bs.Moving_Block_95_Upper.lt(0).all() and r.Holm_Adjusted_P<.05
    status='positive across all interval/corrected-test checks' if positive else ('negative across all interval/corrected-test checks' if negative else 'no directional conclusion across all checks')
    text+=f'{r.Model_A} minus {r.Model_B}, {r.Metric}: delta={r.Mean_A_Minus_B:.6f}, Holm p={r.Holm_Adjusted_P:.6g}; {status}.\n'
text+='\nHistory-only logistic improves class-0 average precision over Persistence under the predeclared checks. History + Structural MLP improves class-0 average precision, balanced accuracy and MCC. Neither establishes corrected macro-F1 superiority over Persistence.\n'
text+='The tested reversed graph-message model does not add incremental value beyond history/node features: it has negative paired effects across all four metrics, all three intervals and Holm tests. This finding is confined to this implementation/cohort, not a general claim that all graph methods fail.\n'
text+='Forward GAT is a secondary direction control. Its higher descriptive mean than reversed GAT does not establish a primary corrected improvement over the matched MLP.\n'
text+='Exact assumptions remain in the original report: 15 weeks, dependent-week caveats, evaluation-only history sensitivity, common neural tuning, sampled neighbors, and old-GAT sampling/tuning confounding.\n'
with findings.open('x',encoding='utf-8') as f:f.write(text)
print(text,flush=True)
