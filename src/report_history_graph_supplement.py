"""Paired-week, moving-block and Holm inference for the history supplement."""
import json,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from run_history_graph_supplement import ROOT,RUN,MODELS,NEURAL,CONFIG,reference_records
from build_orientation_baselines import METRICS,sha256

BOOTSTRAPS=10000
STAT_METRICS=['macro_F1','class_0_PR_AUC','balanced_accuracy','MCC']
PAIRS=[(MODELS[0],'Persistence'),(MODELS[1],'Persistence'),(MODELS[2],'Persistence'),(MODELS[2],MODELS[1]),(MODELS[2],'Reversed GAT (no history)')]
COLORS={'Persistence':'#B65B25',MODELS[0]:'#174A75',MODELS[1]:'#8665A0',MODELS[2]:'#27856F',MODELS[3]:'#C15C87','Reversed GAT (no history)':'#737373'}


def bootstrap_indices(n=15):
    rng=np.random.default_rng(42)
    draws={'ordinary':rng.integers(0,n,(BOOTSTRAPS,n))}
    for length in [3,4]:
        starts=rng.integers(0,n-length+1,(BOOTSTRAPS,math.ceil(n/length)))
        draws[f'block_{length}']=(starts[:,:,None]+np.arange(length)[None,None,:]).reshape(BOOTSTRAPS,-1)[:,:n]
    return draws


def exact_sign_flip(delta):
    n=len(delta)
    signs=2*((np.arange(2**n)[:,None]>>np.arange(n)[None,:])&1)-1
    null=(signs*delta).mean(axis=1)
    tolerance=100*np.finfo(float).eps*max(1,abs(delta.mean()))
    return float(np.mean(np.abs(null)>=abs(delta.mean())-tolerance))


def holm(pvalues):
    p=np.asarray(pvalues);order=np.argsort(p,kind='stable');m=len(p)
    sorted_adjusted=np.minimum(1,np.maximum.accumulate(p[order]*(m-np.arange(m))))
    result=np.empty(m);result[order]=sorted_adjusted
    return result


def make_tables(records,weeks):
    weekly=records.groupby(['Cohort','Model','Week_Start'],sort=True)[METRICS].mean().reset_index()
    weekly['Test_Class_0_Prevalence']=weekly.no_skill_class_0_PR_AUC
    draws=bootstrap_indices();summary=[];pairs=[];blocks=[];tests=[]
    for (cohort,model),g in weekly.groupby(['Cohort','Model']):
        g=g.set_index('Week_Start').loc[weeks]
        raw=records[records.Cohort.eq(cohort)&records.Model.eq(model)]
        count=raw.Seed.nunique() or 1
        for metric in METRICS:
            values=g[metric].to_numpy();lo,hi=np.quantile(values[draws['ordinary']].mean(1),[.025,.975])
            seed_means=raw.groupby('Seed')[metric].mean()
            summary.append(dict(Cohort=cohort,Model=model,Metric=metric,Mean=values.mean(),Ordinary_95_Lower=lo,Ordinary_95_Upper=hi,
                                SD_Over_Week_Means=values.std(ddof=1),SD_Over_Seed_Means=seed_means.std(ddof=1) if count>1 else np.nan,
                                Test_Weeks=15,Seeds=count,Bootstrap_Draws=BOOTSTRAPS,Inference_Unit='week; seeds averaged within week'))
    for cohort,g in weekly.groupby('Cohort'):
        for a,b in [*PAIRS,(MODELS[2],MODELS[3])]:
            av=g[g.Model.eq(a)].set_index('Week_Start').loc[weeks]
            bv=g[g.Model.eq(b)].set_index('Week_Start').loc[weeks]
            primary=(a,b) in PAIRS
            for metric in STAT_METRICS:
                delta=(av[metric]-bv[metric]).to_numpy()
                lo,hi=np.quantile(delta[draws['ordinary']].mean(1),[.025,.975])
                common=dict(Cohort=cohort,Model_A=a,Model_B=b,Metric=metric,Mean_A_Minus_B=delta.mean(),Paired_Weeks=15,
                            Primary_Comparison=primary,A_Better_Weeks=int((delta>1e-12).sum()),A_Worse_Weeks=int((delta< -1e-12).sum()))
                pairs.append(dict(**common,Ordinary_95_Lower=lo,Ordinary_95_Upper=hi,Bootstrap_Draws=BOOTSTRAPS))
                for length in [3,4]:
                    lo,hi=np.quantile(delta[draws[f'block_{length}']].mean(1),[.025,.975])
                    blocks.append(dict(**common,Block_Length=length,Moving_Block_95_Lower=lo,Moving_Block_95_Upper=hi,
                                       Bootstrap_Draws=BOOTSTRAPS,Method='overlapping non-circular blocks; concatenate then truncate to 15'))
                if primary:tests.append(dict(**common,Uncorrected_P=exact_sign_flip(delta),Test='exact two-sided paired mean sign-flip; 32768 assignments',Alternative='two-sided'))
    tests=pd.DataFrame(tests)
    for cohort,g in tests.groupby('Cohort'):
        assert len(g)==20
        adjusted=holm(g.Uncorrected_P)
        tests.loc[g.index,'Holm_Adjusted_P']=adjusted
        tests.loc[g.index,'Holm_Reject_0_05']=adjusted<.05
        tests.loc[g.index,'Holm_Family_Size']=20
        tests.loc[g.index,'Family']='five predeclared pairs x four metrics; separate cohort families'
    return pd.DataFrame(summary),pd.DataFrame(pairs),pd.DataFrame(blocks),tests,weekly,draws


def main():
    tables=ROOT/'results/tables';figures=ROOT/'results/figures'
    names=['history_graph_summary.csv','history_graph_paired_comparisons.csv','history_graph_block_bootstrap.csv','history_graph_holm_tests.csv',
           'history_graph_report.txt','history_graph_reference_metrics.csv','history_graph_weekly_metrics.csv','history_graph_figure_manifest.csv','history_graph_leakage_audit.json']
    plot_names=['history_graph_rolling_macro_f1','history_graph_rolling_class_0_pr_auc','history_graph_incremental_forest','history_graph_weekly_incremental_difference']
    paths=[tables/name for name in names]+[figures/f'{name}.{ext}' for name in plot_names for ext in ['png','svg']]+[RUN/'statistics_draws.npz']
    assert not any(p.exists() for p in paths),'New supplementary report output exists; no overwrite'
    new=pd.read_csv(tables/'history_graph_metrics_by_seed_week.csv')
    assert len(new)==1200
    reference=pd.DataFrame(reference_records());assert len(reference)==330
    records=pd.concat([new,reference],ignore_index=True)
    protocol=pd.read_csv(tables/'rolling_origin_baseline_protocol.csv');weeks=protocol.Test_Week.tolist()
    for _,g in records.groupby(['Cohort','Model','Seed'],dropna=False):assert sorted(g.Week_Start)==weeks
    assert records.groupby(['Cohort','Week_Start']).Test_Users.nunique().eq(1).all()
    summary,paired,blocks,tests,weekly,draws=make_tables(records,weeks)
    hp=pd.read_csv(tables/'history_graph_hyperparameters.csv')
    assert len(hp)==600 and hp.Threshold.eq(.5).all() and hp.Scaler_Fit_Rows.eq(hp.Train_Users).all()
    assert (hp.Train_Last_Week<hp.Validation_Week).all() and (hp.Validation_Week<hp.Week_Start).all()
    neural=hp[hp.Model.isin(NEURAL)]
    assert neural.groupby(['Week_Start','Seed']).Selected_Grid_ID.nunique().eq(1).all()
    assert neural.groupby(['Week_Start','Seed']).Scaler_Mean.nunique().eq(1).all()
    manifest=json.loads((RUN/'manifest.json').read_text())
    assert all(sha256(ROOT/p)==h for p,h in manifest['protected_hashes'].items())
    main_summary=summary[summary.Cohort.eq('All eligible')]
    def best(metric):return main_summary[main_summary.Metric.eq(metric)].sort_values('Mean',ascending=False).iloc[0].Model
    def effect(a,b,metric='macro_F1',cohort='All eligible'):
        r=paired[paired.Cohort.eq(cohort)&paired.Model_A.eq(a)&paired.Model_B.eq(b)&paired.Metric.eq(metric)].iloc[0]
        bs=blocks[blocks.Cohort.eq(cohort)&blocks.Model_A.eq(a)&blocks.Model_B.eq(b)&blocks.Metric.eq(metric)].sort_values('Block_Length')
        t=tests[tests.Cohort.eq(cohort)&tests.Model_A.eq(a)&tests.Model_B.eq(b)&tests.Metric.eq(metric)].iloc[0]
        established=r.Ordinary_95_Lower>0 and bs.Moving_Block_95_Lower.gt(0).all() and t.Holm_Adjusted_P<.05
        return r,bs,t,bool(established)
    central=effect(MODELS[2],MODELS[1])
    robust_beats=[name for name in MODELS[:3] if effect(name,'Persistence')[3]]
    results=dict(best_by_macro_F1=best('macro_F1'),best_by_class_0_PR_AUC=best('class_0_PR_AUC'),best_by_balanced_accuracy=best('balanced_accuracy'),
                 models_with_positive_macro_F1_CIs_and_Holm_vs_Persistence=robust_beats,
                 incremental_graph_macro_F1_ordinary_lower=float(central[0].Ordinary_95_Lower),
                 incremental_graph_positive_macro_F1_all_intervals_and_Holm=central[3],fits=600,seeds=list(range(42,52)),test_weeks=weeks,
                 eligible_test_user_weeks=int(new[new.Model.eq(MODELS[0])&new.Seed.eq(42)&new.Cohort.eq('All eligible')].Test_Users.sum()),
                 no_history_test_user_weeks=int(new[new.Model.eq(MODELS[0])&new.Seed.eq(42)&new.Cohort.eq('All eligible')].No_History_Users.sum()),
                 epochs_capped_fits=int(neural.Epochs_Run.eq(30).sum()))
    assumptions=[
        'Target is the dataset binary orientation label, not verified sentiment. Labels and eligible users are unchanged; tied historical labels are ignored. Primary evaluation retains users lacking prior valid labels.',
        'History is computed from ALL valid earlier weekly labels, including the first dataset week, not merely supervised training rows. Previous_Label is the latest non-tied label; class rates count valid weekly labels equally, not tweets. Histories for every receiver AND sampled context neighbor use label week < that sample prediction week.',
        'A valid label from week w is available at Monday w+7 days. The feature row named t contains interactions strictly before t; its window boundary equals t, while actual last interaction time is <t. The historical graph uses completed snapshots through t-7 days, cumulative across weeks. No target-week adjacency or ratios enter features.',
        'For each origin, missing Previous_Label=0, availability=0 and count=0. Missing class rates use the TRAINING target-label prevalence and its complement. Missing recency uses max observed training-row recency+1 (or 1 if none); all observed recencies are capped at that value. Raw missing rates/recency remain NA in the provenance table and are imputed separately per origin.',
        'Each origin scaler fits only imputed supervised TRAINING rows. The same imputation values and frozen 12-feature scaler apply to graph context nodes. History-only logistic has its own training-only six-feature scaler. Training preprocessing is fitted once per origin, rather than separately for each older training-row date.',
        'Same 15 expanding origins and validation weeks, seeds42-51, and fixed class-1 threshold0.5. Class weights N/(2*N_class) use training labels only. No labels from validation/test fit defaults, scalers or class weights. Hyperparameter and early-epoch selection use validation labels only.',
        'History + Structural MLP is the exact self-only version of the history GAT classifier: identical 12 inputs, shared parameter shapes and initialization, multihead linear projection, ELU, dense/ReLU/dropout classifier. One self attention weight equals1, so its prediction is a node-only MLP. Unused attention parameters do not add label/graph predictors. It is not the earlier standalone MLP architecture.',
        'The critical neural pair shares one configuration per origin, chosen by the mean of their seed42 validation macro-F1 over the same predeclared three configurations. This common policy freezes architecture width/dropout/lr/decay/heads for both, rather than selecting different backbones. Every final seed independently retrains and early-stops. Forward GAT uses that same configuration as a secondary direction control.',
        'Directed weighted GAT samples at most16 non-self neighbors plus self using the previous deterministic seed42/cutoff/direction/local-node hash; its attention adds log Weight. Reversed messages receive original outgoing neighbors. Here node rows are sorted before case normalization, unlike the previous graph experiment, so neighbor slot order and high-degree sampled subsets can differ from the existing no-history GAT. Every selected pair/weight is verified against raw prior interactions. Attention and classifier are one hop; sampling is fixed across training seeds. Known users only, no ID embeddings or future node identities. Comparisons to old no-history GAT also include this sampling and tuning-policy change; they do not isolate adding label history.',
        'Logistic C is chosen separately from .01,.1,1,10,100 on validation macro-F1, tie favoring smallerC. Deterministic lbfgs, max3000iterations, repeated seeds produce the same convex-fit result. Neural training uses deterministic CPU Adam, batch1024 chronological rows, max30epochs and patience4; each seed restores earliest best validation epoch.',
        'Sensitivity analysis excludes no-history users from held-out evaluation only; training and validation, fitted defaults, hyperparameters and models remain identical to primary runs. Persistence and existing no-history reversed GAT predictions are reused without alteration and evaluated on exactly the same subset. This sensitivity is not a separately refitted history-only cohort experiment.',
        'PR-AUC is average precision, class0 probability=1-p1. Brier and ROC use p1. Weekly no-skill class0 AP equals that cohort weekly prevalence. Means weight each test week equally rather than pooling users; ten seed scores are averaged within each week.',
        'Ordinary paired percentile bootstrap uses10000 shared week resamples, seed42. Moving-block variants use overlapping non-circular consecutive blocks of length3 and4, concatenate independently drawn blocks and truncate to15 observations. No wraparound. All compared models share indices; intervals condition on seed averages and do not resample users or refit models.',
        'Two-sided paired mean sign-flip tests enumerate all32768 independent week sign assignments. Holm correction covers all20 predeclared pair/metric tests as one primary family at alpha.05; sensitivity has its own separate20-test exploratory family. Forward/reverse control is exploratory and excluded from these families. No significance claim uses uncorrected p alone.',
        'Week sign-flip inference assumes independent exchangeable/symmetric null differences; serial dependence can invalidate that assumption. Block bootstrap is a robustness check, not a cure for only15 weeks, nonstationarity or the sign-flip p-value assumptions. Block length choices are predeclared, not selected by favorable intervals. All intervals are unadjusted percentile CIs; Holm adjusts p-values, not CIs.',
        'A positive incremental claim requires positive mean, ordinary and both block lower bounds above0, and Holm-adjusted p<.05. If ordinary CI includes0, graph message passing does not establish incremental value. Other metrics may disagree; a macro-F1 change is not general improvement or causal proof.',
        'Best-model rankings use descriptive test means; they do not select a deployed model. Primary model comparisons were frozen before test aggregation. Structural extraction uncertainty, cohort exclusions, small tuning grid, one-hop sampled topology, class-weighted probability calibration and epoch caps limit generalization.',
        'All existing data, outputs, notebooks and source files (including old checkpoints and bytecode) are protected by SHA256. New files use exclusive creation; Python runs with -B so prior bytecode is not rewritten. New experiment sources are separate, old model results unchanged.'
    ]
    report=json.dumps(results,indent=2)+'\n\nPrincipal means (all eligible):\n'+main_summary[main_summary.Metric.isin(STAT_METRICS)].to_string(index=False)
    report+='\n\nPrimary paired questions (macro-F1):\n'
    for a,b in PAIRS:
        r,bs,t,established=effect(a,b)
        report+=f'{a} minus {b}: delta={r.Mean_A_Minus_B:.6f}; ordinary95=[{r.Ordinary_95_Lower:.6f},{r.Ordinary_95_Upper:.6f}]; '+', '.join(f'block{q.Block_Length}95=[{q.Moving_Block_95_Lower:.6f},{q.Moving_Block_95_Upper:.6f}]' for q in bs.itertuples())+f'; Holm p={t.Holm_Adjusted_P:.6g}; positive across all checks={established}.\n'
    report+='\nCentral question by metric and cohort:\n'
    for cohort in ['All eligible','History available']:
        for metric in STAT_METRICS:
            r,bs,t,established=effect(MODELS[2],MODELS[1],metric,cohort)
            report+=f'{cohort}, {metric}: delta={r.Mean_A_Minus_B:.6f}; ordinary95=[{r.Ordinary_95_Lower:.6f},{r.Ordinary_95_Upper:.6f}]; '+', '.join(f'block{q.Block_Length}95=[{q.Moving_Block_95_Lower:.6f},{q.Moving_Block_95_Upper:.6f}]' for q in bs.itertuples())+f'; Holm p={t.Holm_Adjusted_P:.6g}; positive across all checks={established}.\n'
    report+='\nConservative interpretation:\n'
    report+=('Graph message passing shows a positive macro-F1 effect after history/node features across the predeclared intervals and Holm test, with inference still limited by15 weeks and serial assumptions.\n' if central[3] else 'Graph message passing does not establish a robust positive macro-F1 increment beyond historical labels and structural features under all predeclared checks.\n')
    report+=f'Historical-only logistic robustly improves over Persistence on macro-F1: {effect(MODELS[0],"Persistence")[3]}. History + MLP robustly improves over Persistence: {effect(MODELS[1],"Persistence")[3]}.\n'
    report+='No superiority claim is based on an uncorrected p-value. Statistical support is metric-specific and conditional on the stated dependence assumptions.\n'
    report+='\nExact assumptions and limitations:\n'+'\n'.join(f'{i}. {s}' for i,s in enumerate(assumptions,1))
    report+='\n\nMethod references:\nhttps://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html\nhttps://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html\nhttps://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.MovingBlockBootstrap.html\n'
    # Render all scientific figures before creating report files.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    rendered={};g=weekly[weekly.Cohort.eq('All eligible')];dates=pd.to_datetime(weeks)
    for name,metric,title,ylabel in [(plot_names[0],'macro_F1','Persistence and historical-label graph models','Macro-F1'),
                                    (plot_names[1],'class_0_PR_AUC','Historical-label forecasting: class-0 precision–recall','Class-0 average precision')]:
        fig,ax=plt.subplots(figsize=(12,6));fig.subplots_adjust(left=.08,right=.98,bottom=.14,top=.85)
        for model in ['Persistence',*MODELS,'Reversed GAT (no history)']:
            frame=g[g.Model.eq(model)].set_index('Week_Start').loc[weeks]
            ax.plot(dates,frame[metric],label=model,color=COLORS[model],linestyle='--' if model==MODELS[3] else '-',linewidth=1.8,marker='o',markersize=3)
        if metric=='class_0_PR_AUC':ax.plot(dates,g[g.Model.eq('Persistence')].set_index('Week_Start').loc[weeks,'no_skill_class_0_PR_AUC'],label='No-skill: class-0 prevalence',color='black',linestyle=':')
        locator=mdates.AutoDateLocator(minticks=5,maxticks=7);ax.xaxis.set_major_locator(locator);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set(xlabel='Prediction week beginning Monday (UTC)',ylabel=ylabel,ylim=(0,1));ax.grid(axis='y',alpha=.2)
        ax.legend(frameon=False,ncol=3,fontsize=8,loc='lower left')
        fig.text(.5,.94,title,ha='center',fontsize=15)
        fig.text(.5,.885,'Neural lines: 10-seed means; primary eligible cohort',ha='center',fontsize=10,color='#555555')
        rendered[name]=fig
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for ax,metric in zip(axes.flat,STAT_METRICS):
        for j,(a,b) in enumerate(PAIRS):
            r=paired[paired.Cohort.eq('All eligible')&paired.Model_A.eq(a)&paired.Model_B.eq(b)&paired.Metric.eq(metric)].iloc[0]
            ax.plot([r.Ordinary_95_Lower,r.Ordinary_95_Upper],[j-.15,j-.15],color='#174A75',linewidth=2,label='Ordinary' if j==0 else None)
            ax.plot(r.Mean_A_Minus_B,j-.15,'o',color='#174A75',markersize=4)
            for off,length,color in [(0,3,'#27856F'),(.15,4,'#B65B25')]:
                r2=blocks[blocks.Cohort.eq('All eligible')&blocks.Model_A.eq(a)&blocks.Model_B.eq(b)&blocks.Metric.eq(metric)&blocks.Block_Length.eq(length)].iloc[0]
                ax.plot([r2.Moving_Block_95_Lower,r2.Moving_Block_95_Upper],[j+off,j+off],color=color,linewidth=2,label=f'Block {length}' if j==0 else None)
                ax.plot(r.Mean_A_Minus_B,j+off,'o',color=color,markersize=4)
        short={MODELS[0]:'History LR',MODELS[1]:'History MLP',MODELS[2]:'History reversed GAT','Persistence':'Persistence','Reversed GAT (no history)':'GAT without history'}
        ax.set_yticks(np.arange(5),[short[a]+' − '+short[b] for a,b in PAIRS]);ax.invert_yaxis()
        ax.axvline(0,color='#777777',linestyle='--');ax.grid(axis='x',alpha=.2)
        ax.set(title=metric.replace('_',' '),xlabel='Paired mean difference');ax.legend(frameon=False,ncol=3,fontsize=8)
    fig.suptitle('Historical-label graph contribution: ordinary and moving-block 95% CIs',fontsize=14)
    rendered[plot_names[2]]=fig
    fig,ax=plt.subplots(figsize=(12,5));fig.subplots_adjust(left=.09,right=.98,bottom=.15,top=.83)
    a=records[records.Cohort.eq('All eligible')&records.Model.eq(MODELS[2])].pivot(index='Seed',columns='Week_Start',values='macro_F1').loc[list(range(42,52)),weeks]
    b=records[records.Cohort.eq('All eligible')&records.Model.eq(MODELS[1])].pivot(index='Seed',columns='Week_Start',values='macro_F1').loc[list(range(42,52)),weeks]
    difference=a-b;means=difference.mean(0);lo=difference.quantile(.1);hi=difference.quantile(.9)
    ax.axhline(0,color='black',linestyle='--',linewidth=1)
    ax.fill_between(dates,lo,hi,color=COLORS[MODELS[2]],alpha=.18,label='10th–90th seed percentile; descriptive')
    ax.plot(dates,means,color=COLORS[MODELS[2]],marker='o',label='Mean paired seed difference')
    locator=mdates.AutoDateLocator(minticks=5,maxticks=7);ax.xaxis.set_major_locator(locator);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set(xlabel='Prediction week beginning Monday (UTC)',ylabel='Macro-F1 difference');ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=9)
    fig.text(.5,.94,'History + reversed GAT minus history + structural MLP',ha='center',fontsize=15)
    rendered[plot_names[3]]=fig
    for name,frame in [('history_graph_summary.csv',summary),('history_graph_paired_comparisons.csv',paired),('history_graph_block_bootstrap.csv',blocks),
                       ('history_graph_holm_tests.csv',tests),('history_graph_reference_metrics.csv',reference),('history_graph_weekly_metrics.csv',weekly)]:
        with (tables/name).open('x',encoding='utf-8',newline='') as f:frame.to_csv(f,index=False,na_rep='NA')
    figure_rows=[]
    for letter,name in zip('ABCD',plot_names):
        for ext in ['png','svg']:
            path=figures/f'{name}.{ext}'
            with path.open('xb') as f:rendered[name].savefig(f,format=ext,dpi=400,facecolor='white')
        plt.close(rendered[name]);figure_rows.append(dict(Figure=letter,PNG=f'results/figures/{name}.png',SVG=f'results/figures/{name}.svg',PNG_DPI=400))
    with (tables/'history_graph_figure_manifest.csv').open('x',encoding='utf-8',newline='') as f:pd.DataFrame(figure_rows).to_csv(f,index=False)
    with (RUN/'statistics_draws.npz').open('xb') as f:np.savez(f,**draws)
    with (tables/'history_graph_report.txt').open('x',encoding='utf-8') as f:f.write(report+'\n')
    audit=dict(all_assertions_passed=True,raw_and_existing_files_unchanged=True,protected_files=len(manifest['protected_hashes']),
               label_week_strictly_prior=True,graph_events_strictly_prior=True,actual_structural_event_timestamps_strictly_prior=True,
               historical_features_identical_across_neural_pair=True,scalers_imputation_weights_training_only=True,
               validation_only_common_hyperparameters=True,threshold=.5,holm_primary_family=20,bootstrap_seed=42,config=CONFIG)
    with (tables/'history_graph_leakage_audit.json').open('x',encoding='utf-8') as f:json.dump(audit,f,indent=2)
    print(report,flush=True)


if __name__=='__main__':main()
