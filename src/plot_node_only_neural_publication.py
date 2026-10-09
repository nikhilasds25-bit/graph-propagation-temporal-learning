"""Correct subtitle margins in new figure files, never replace outputs."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from run_node_only_neural_baselines import ROOT, MODELS, COLORS, SEEDS


def main():
    metrics = pd.read_csv(ROOT / 'results/tables/node_only_neural_metrics_by_seed_week.csv')
    weekly = pd.read_csv(ROOT / 'results/tables/node_only_all_model_weekly_metrics.csv')
    weeks = sorted(weekly.Week_Start.unique())
    dates = pd.to_datetime(weeks)
    # Replay the same seed resamples used in the original figure calculation.
    rng = np.random.default_rng(42)
    rng.integers(0, 15, size=(10000, 15))
    draws = rng.integers(0, 10, size=(10000, 10))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'})
    for name, metric, title, ylabel in [
        ('node_only_rolling_class_0_pr_auc_publication','class_0_PR_AUC','Node-only class-0 precision–recall performance','Class-0 average precision'),
        ('node_only_rolling_balanced_accuracy_publication','balanced_accuracy','Node-only rolling balanced accuracy','Balanced accuracy')]:
        paths = [ROOT / 'results/figures' / (name+'.'+ext) for ext in ['png','svg']]
        if any(p.exists() for p in paths):
            raise FileExistsError('Publication variant already exists')
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=.10,right=.98,bottom=.16,top=.82)
        for model in MODELS:
            values = weekly[weekly.Model.eq(model)].set_index('Week_Start').loc[weeks,metric].to_numpy()
            ax.plot(dates,values,label=model,color=COLORS[model],linewidth=1.6,marker='o',markersize=3)
            if model in ['MLP','GRU']:
                matrix=metrics[metrics.Model.eq(model)].pivot(index='Seed',columns='Week_Start',values=metric).loc[SEEDS,weeks].to_numpy()
                lo,hi=np.quantile(matrix[draws].mean(axis=1),[.025,.975],axis=0)
                ax.fill_between(dates,lo,hi,color=COLORS[model],alpha=.15)
        if metric=='class_0_PR_AUC':
            prevalence=weekly[weekly.Model.eq('Majority')].set_index('Week_Start').loc[weeks,'no_skill_class_0_PR_AUC']
            ax.plot(dates,prevalence,color='#000000',linestyle='--',label='No-skill: class-0 prevalence')
        else:
            ax.axhline(.5,color='#000000',linestyle='--',label='No-skill: 0.5')
        locator=mdates.AutoDateLocator(minticks=5,maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set(xlabel='Test week beginning Monday (UTC)',ylabel=ylabel,ylim=(0,1))
        ax.grid(axis='y',alpha=.2)
        ax.legend(frameon=False,ncol=3,fontsize=8,loc='lower left')
        fig.text(.5,.91,title,ha='center',va='center',fontsize=14)
        fig.text(.5,.965,'Neural lines: 10-seed means; shaded bands: pointwise seed-bootstrap 95% CIs',ha='center',va='center',fontsize=9,color='#555555')
        for p, ext in zip(paths,['png','svg']):
            with p.open('xb') as f:
                fig.savefig(f,format=ext,dpi=400,facecolor='white')
        plt.close(fig)
        print('Created new publication variant:',name)


if __name__=='__main__':
    main()
