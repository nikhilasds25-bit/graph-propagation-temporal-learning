"""Create a new Figure E with distinguishable directed-model line styles."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from report_weekly_graph_forecasting import ROOT,PRIMARY,COLORS

tables=ROOT/'results/tables'
figures=ROOT/'results/figures'
paths=[figures/f'graph_forecasting_static_vs_temporal_publication.{ext}' for ext in ['png','svg']]
manifest=tables/'graph_forecasting_publication_figure_manifest.csv'
assert not any(p.exists() for p in [*paths,manifest])
weekly=pd.read_csv(tables/'graph_forecasting_all_model_weekly_metrics.csv')
summary=pd.read_csv(tables/'graph_models_summary.csv')
static=[*PRIMARY[:3],'GraphSAGE (reversed)','GAT (reversed)']
best=summary[summary.Metric.eq('macro_F1')&summary.Model.isin(static)].sort_values('Mean',ascending=False).iloc[0].Model
models=list(dict.fromkeys([*PRIMARY,best,'GRU (aligned)']))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,1,figsize=(11,8))
fig.subplots_adjust(left=.1,right=.98,bottom=.1,top=.87,hspace=.35)
for ax,metric,label in zip(axes,['macro_F1','class_0_PR_AUC'],['Macro-F1','Class-0 average precision']):
    for model in models:
        frame=weekly[weekly.Model.eq(model)].sort_values('Week_Start')
        ax.plot(pd.to_datetime(frame.Week_Start),frame[metric],label=model,color=COLORS[model],linestyle='--' if '(reversed)' in model else '-',linewidth=1.8)
    locator=mdates.AutoDateLocator(minticks=5,maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set(ylabel=label,xlabel='Prediction week beginning Monday (UTC)',ylim=(0,1))
    ax.grid(axis='y',alpha=.2)
    ax.legend(frameon=False,ncol=3,fontsize=9,loc='lower left')
fig.text(.5,.96,'Static versus temporal graph forecasting',ha='center',fontsize=15)
fig.text(.5,.91,'10-seed means; dashed line denotes reversed messages',ha='center',fontsize=10,color='#555555')
for path in paths:
    with path.open('xb') as f:
        fig.savefig(f,format=path.suffix[1:],dpi=400,facecolor='white')
plt.close(fig)
frame=pd.read_csv(tables/'graph_forecasting_figure_manifest.csv')
frame.loc[frame.Figure.eq('E'),'PNG']=str(paths[0].relative_to(ROOT))
frame.loc[frame.Figure.eq('E'),'SVG']=str(paths[1].relative_to(ROOT))
with manifest.open('x',encoding='utf-8',newline='') as f:
    frame.to_csv(f,index=False)
print('Saved refined Figure E and publication figure manifest; original figure retained.')
