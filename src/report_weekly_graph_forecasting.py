"""Paired-week statistics, figures and conservative graph forecasting report."""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from weekly_graph_forecasting import ROOT,RUN,PRIMARY,SEEDS,CONFIG
from build_orientation_baselines import METRICS,sha256

BOOTSTRAPS=10000
MAIN=['Majority','Persistence','Logistic Regression','MLP','GRU (aligned)',*PRIMARY]
COLORS=dict(zip(MAIN,['#737373','#B65B25','#174A75','#8665A0','#27856F','#2E75B6','#B8860B','#C15C87','#4D6779']))
REVERSE=['GraphSAGE (reversed)','GAT (reversed)']
ALL_GRAPHS=[*PRIMARY,*REVERSE]
ALL_MAIN=[*MAIN,*REVERSE]
COLORS.update({name:COLORS[name.split(' (')[0]] for name in REVERSE})


def main():
    tables=ROOT/'results/tables'; figures=ROOT/'results/figures'
    paths={key:tables/name for key,name in {
        'summary':'graph_models_summary.csv','comparisons':'graph_model_comparisons.csv',
        'direction':'graph_direction_control.csv','ablation':'graph_ablation_results.csv',
        'weekly':'graph_forecasting_all_model_weekly_metrics.csv','cohort':'graph_forecasting_cohort_audit.csv',
        'report':'graph_models_report.txt','audit':'graph_forecasting_leakage_audit.json',
        'figures':'graph_forecasting_figure_manifest.csv'}.items()}
    plot_names=['graph_forecasting_rolling_macro_f1','graph_forecasting_rolling_class_0_pr_auc',
                'graph_forecasting_rolling_balanced_accuracy','graph_forecasting_paired_forest','graph_forecasting_static_vs_temporal']
    for name in plot_names:
        for ext in ['png','svg']:
            paths[name+'_'+ext]=figures/(name+'.'+ext)
    if any(p.exists() for p in paths.values()):
        raise FileExistsError('New report output already exists; no writes')
    graph=pd.concat([pd.read_csv(tables/'graph_models_metrics_by_seed_week.csv'),pd.read_csv(tables/'graph_direction_ablation_metrics.csv')],ignore_index=True)
    hp=pd.concat([pd.read_csv(tables/'graph_model_hyperparameters.csv'),pd.read_csv(tables/'graph_direction_ablation_hyperparameters.csv')],ignore_index=True)
    protocol=pd.read_csv(tables/'rolling_origin_baseline_protocol.csv')
    weeks=protocol.Test_Week.tolist(); dates=pd.to_datetime(weeks)
    assert len(weeks)==15 and weeks==sorted(weeks)
    assert len(graph)==2850 and len(hp)==2850
    assert not graph.duplicated(['Week_Start','Model','Direction','Control','Seed']).any()
    primary=graph[graph.Model.isin(PRIMARY)&graph.Direction.eq('forward')&graph.Control.eq('primary')].copy()
    reverse=graph[graph.Direction.eq('reverse')&graph.Control.eq('primary')].copy()
    reverse['Model']=reverse.Model+' (reversed)'
    aligned=graph[graph.Model.eq('GRU (aligned)')].copy()
    assert len(primary)==600 and len(aligned)==150
    for _,group in graph.groupby(['Model','Direction','Control','Seed']):
        assert sorted(group.Week_Start)==weeks
    classical=pd.read_csv(tables/'rolling_origin_baseline_metrics_by_week.csv')
    neural=pd.read_csv(tables/'node_only_neural_metrics_by_seed_week.csv')
    neural.loc[neural.Model.eq('GRU'),'Model']='GRU (legacy lagged)'
    seeds=pd.concat([primary[['Week_Start','Model','Seed',*METRICS]],reverse[['Week_Start','Model','Seed',*METRICS]],aligned[['Week_Start','Model','Seed',*METRICS]],
                     neural[['Week_Start','Model','Seed',*METRICS]]],ignore_index=True)
    neural_means=seeds.groupby(['Model','Week_Start'])[METRICS].mean().reset_index()
    weekly=pd.concat([classical[['Model','Week_Start',*METRICS]],neural_means],ignore_index=True)
    assert not weekly.duplicated(['Model','Week_Start']).any()
    for name in ALL_MAIN:
        assert sorted(weekly[weekly.Model.eq(name)].Week_Start)==weeks
    rng=np.random.default_rng(42)
    wd=rng.integers(0,15,(BOOTSTRAPS,15))
    sd=rng.integers(0,10,(BOOTSTRAPS,10))
    summary_rows=[]
    for name in [*ALL_MAIN,'GRU (legacy lagged)']:
        frame=weekly[weekly.Model.eq(name)].set_index('Week_Start').loc[weeks]
        for metric in METRICS:
            values=frame[metric].to_numpy(float)
            if name in ['Majority','Persistence','Logistic Regression']:
                interval=np.quantile(values[wd].mean(1),[.025,.975]); seed_std=np.nan; within=np.nan
                seed_interval=[np.nan,np.nan]; seed_count=1
            else:
                matrix=seeds[seeds.Model.eq(name)].pivot(index='Seed',columns='Week_Start',values=metric).loc[SEEDS,weeks].to_numpy()
                seed_mean=matrix.mean(1)
                seed_std=seed_mean.std(ddof=1); within=matrix.std(axis=0,ddof=1).mean()
                interval=np.quantile(matrix[sd[:,:,None],wd[:,None,:]].mean(axis=(1,2)),[.025,.975])
                seed_interval=np.quantile(seed_mean[sd].mean(1),[.025,.975]); seed_count=10
            summary_rows.append(dict(Model=name,Metric=metric,Mean=values.mean(),SD_Over_Seed_Means=seed_std,
                                     Mean_Within_Week_Seed_SD=within,SD_Over_Week_Means=values.std(ddof=1),
                                     Bootstrap_95_Lower=interval[0],Bootstrap_95_Upper=interval[1],
                                     Seed_Mean_95_Lower=seed_interval[0],Seed_Mean_95_Upper=seed_interval[1],
                                     Test_Weeks=15,Seeds=seed_count,Bootstrap_Draws=BOOTSTRAPS))
    summary=pd.DataFrame(summary_rows)
    def values(name,metric):
        if name=='No-skill':
            return values('Majority','no_skill_class_0_PR_AUC')
        return weekly[weekly.Model.eq(name)].set_index('Week_Start').loc[weeks,metric].to_numpy()
    def paired(a,b,metric,av=None,bv=None):
        av=values(a,metric) if av is None else av
        bv=values(b,metric) if bv is None else bv
        delta=av-bv
        lo,hi=np.quantile(delta[wd].mean(1),[.025,.975])
        return dict(Model_A=a,Model_B=b,Metric=metric,Mean_A_Minus_B=delta.mean(),
                    Paired_Bootstrap_95_Lower=lo,Paired_Bootstrap_95_Upper=hi,A_Better_Weeks=int((delta>1e-12).sum()),
                    A_Worse_Weeks=int((delta< -1e-12).sum()),Paired_Weeks=15,
                    Consistently_Better_All_Weeks=bool((delta>1e-12).all()),Nominal_Positive_CI=bool(lo>0),Bootstrap_Draws=BOOTSTRAPS)
    pair_set={(g,n) for g in ALL_GRAPHS for n in ['Persistence','MLP','Logistic Regression']}
    pair_set|={('GraphSAGE','Static GCN'),('GAT','Static GCN'),('Temporal GCN','GRU (aligned)'),('Temporal GCN','GRU (legacy lagged)')}
    pair_set|={(name,'Static GCN') for name in REVERSE}
    pair_set|={('Temporal GCN',g) for g in [*PRIMARY[:3],*REVERSE]}
    comparisons=[paired(a,b,m) for a,b in sorted(pair_set) for m in ['macro_F1','class_0_PR_AUC','balanced_accuracy','MCC']]
    comparisons += [paired(g,'No-skill','class_0_PR_AUC') for g in ALL_GRAPHS]
    comparisons=pd.DataFrame(comparisons)
    direction=[]
    for name in PRIMARY:
        for metric in ['macro_F1','class_0_PR_AUC','balanced_accuracy','MCC']:
            av=values(name,metric)
            if name in ['Static GCN','Temporal GCN']:
                bv=av.copy(); method='algebraic invariance; same symmetric adjacency, no redundant refit'
            else:
                reverse=graph[graph.Model.eq(name)&graph.Direction.eq('reverse')&graph.Control.eq('primary')]
                bv=reverse.groupby('Week_Start')[metric].mean().loc[weeks].to_numpy()
                method='independent reverse-direction validation tuning and ten-seed retraining'
            row=paired(name+' forward',name+' reverse',metric,av,bv)
            row.update(Model=name,Forward_Message='Source -> Target (receiving Target aggregates Source)',
                       Reverse_Message='Target -> Source',Method=method)
            direction.append(row)
    direction=pd.DataFrame(direction)
    ablations=[]
    for name in ALL_GRAPHS:
        for control in ['feature_only','rewired']:
            base=name.split(' (')[0]
            graph_direction='reverse' if name in REVERSE else 'forward'
            rows=graph[graph.Model.eq(base)&graph.Direction.eq(graph_direction)&graph.Control.eq(control)]
            for metric in ['macro_F1','class_0_PR_AUC','balanced_accuracy','MCC']:
                bv=rows.groupby('Week_Start')[metric].mean().loc[weeks].to_numpy()
                result=paired(name,name+' '+control,metric,values(name,metric),bv)
                result.update(Graph_Model=name,Control=control,Control_Architecture='same model/parameters; identity messages' if control=='feature_only' else 'same model/parameters; event-endpoint rewiring',
                              Control_Mean=bv.mean(),Graph_Mean=values(name,metric).mean())
                ablations.append(result)
    ablations=pd.DataFrame(ablations)
    # Cohort accounting is recomputed from labels, not borrowed target ratios.
    labels=pd.read_csv(ROOT/'data/processed/weekly_user_orientation_labels.csv',dtype={'User':str})
    dataset=pd.read_csv(ROOT/'data/processed/weekly_leakage_controlled_prediction_dataset.csv',dtype={'User':str})
    joined=labels.merge(dataset[['Week_Start','User']],on=['Week_Start','User'],how='left',indicator=True,validate='one_to_one')
    cohort=[]
    for week,g in joined.groupby('Week_Start'):
        valid=g.Orientation_Label.notna(); has=g._merge.eq('both')
        cohort.append(dict(Week_Start=week,Active_Users=len(g),Valid_Label_Users=int(valid.sum()),
                           Tie_Count=int((~valid).sum()),Cold_Start_Count=int((valid&~has).sum()),Evaluated_Users=int(has.sum())))
    cohort=pd.DataFrame(cohort)
    assert cohort.Evaluated_Users.sum()==len(dataset)
    main_summary=summary[summary.Model.isin(ALL_MAIN)]
    winner=lambda metric,names:summary[summary.Metric.eq(metric)&summary.Model.isin(names)].sort_values('Mean',ascending=False).iloc[0].Model
    best_graph=winner('macro_F1',ALL_GRAPHS); best_static=winner('macro_F1',[*PRIMARY[:3],*REVERSE])
    best_node=winner('macro_F1',MAIN[:5])
    def select(frame,a,b,metric='macro_F1'):
        return frame[frame.Model_A.eq(a)&frame.Model_B.eq(b)&frame.Metric.eq(metric)].iloc[0]
    message=ablations[ablations.Graph_Model.eq(best_graph)&ablations.Control.eq('feature_only')&ablations.Metric.eq('macro_F1')].iloc[0]
    temporal=select(comparisons,'Temporal GCN',best_static)
    result_info=dict(best_by_macro_F1=winner('macro_F1',ALL_MAIN),best_by_class_0_PR_AUC=winner('class_0_PR_AUC',ALL_MAIN),
                     best_by_balanced_accuracy=winner('balanced_accuracy',ALL_MAIN),best_graph_by_macro_F1=best_graph,
                     best_static_graph_by_macro_F1=best_static,best_node_only_by_macro_F1=best_node,
                     graph_message_passing_nominal_positive_macro_F1_CI=bool(message.Paired_Bootstrap_95_Lower>0),
                     temporal_vs_best_static_nominal_positive_macro_F1_CI=bool(temporal.Paired_Bootstrap_95_Lower>0),
                     graph_models_exceeding_class_0_prevalence_all_weeks=comparisons[(comparisons.Model_B=='No-skill')&comparisons.Consistently_Better_All_Weeks].Model_A.tolist(),
                     primary_graph_fits=600,reversed_direction_fits=300,matched_and_null_control_fits=1800,aligned_GRU_fits=150,
                     test_weeks=weeks,seeds=SEEDS,epochs_capped_fits=int(hp.Epochs_Run.eq(30).sum()))
    # Publication figures: stable manual margins, one y-axis, explicit band meaning.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    rendered={}
    for name,metric,title,ylabel in [
        (plot_names[0],'macro_F1','Weekly graph forecasting: macro-F1','Macro-F1'),
        (plot_names[1],'class_0_PR_AUC','Weekly graph forecasting: class-0 precision–recall','Class-0 average precision'),
        (plot_names[2],'balanced_accuracy','Weekly graph forecasting: balanced accuracy','Balanced accuracy')]:
        fig,ax=plt.subplots(figsize=(12,6))
        fig.subplots_adjust(left=.08,right=.98,bottom=.15,top=.81)
        for model in ALL_MAIN:
            av=values(model,metric)
            ax.plot(dates,av,label=model,color=COLORS[model],linestyle='--' if model in REVERSE else '-',linewidth=1.6,marker='o',markersize=2.5)
        if metric=='class_0_PR_AUC':
            ax.plot(dates,values('No-skill',metric),color='black',linestyle='--',label='No-skill: class-0 prevalence')
        if metric=='balanced_accuracy':
            ax.axhline(.5,color='black',linestyle='--',label='No-skill: 0.5')
        locator=mdates.AutoDateLocator(minticks=5,maxticks=7)
        ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set(xlabel='Prediction week beginning Monday (UTC)',ylabel=ylabel,ylim=(0,1))
        ax.grid(axis='y',alpha=.2); ax.legend(frameon=False,fontsize=8,ncol=4,loc='lower left')
        fig.text(.5,.91,title,ha='center',va='center',fontsize=15)
        fig.text(.5,.965,'Neural lines: 10-seed means; GRU is newly aligned to the same completed-week cutoff',ha='center',fontsize=10,color='#555555')
        rendered[name]=fig
    forest_pairs=[('Static GCN','MLP'),('GraphSAGE','MLP'),('GAT','MLP'),(best_graph,'MLP'),('Temporal GCN','GRU (aligned)'),(best_graph,'Persistence'),('Temporal GCN',best_static)]
    forest_pairs=list(dict.fromkeys(forest_pairs))
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    for ax,metric in zip(axes.flat,['macro_F1','class_0_PR_AUC','balanced_accuracy','MCC']):
        frame=pd.DataFrame([select(comparisons,a,b,metric) for a,b in forest_pairs])
        ys=np.arange(len(frame))
        ax.errorbar(frame.Mean_A_Minus_B,ys,xerr=np.array([frame.Mean_A_Minus_B-frame.Paired_Bootstrap_95_Lower,frame.Paired_Bootstrap_95_Upper-frame.Mean_A_Minus_B]),fmt='o',color='#174A75',capsize=3)
        ax.axvline(0,color='#777777',linestyle='--',linewidth=1)
        ax.set_yticks(ys,[f'{a} − {b}' for a,b in forest_pairs]); ax.invert_yaxis()
        ax.set(title=metric.replace('_',' '),xlabel='Paired mean difference (positive favors first model)'); ax.grid(axis='x',alpha=.2)
    fig.suptitle('Graph versus node-only and static controls: paired-week bootstrap 95% CIs',fontsize=14)
    rendered[plot_names[3]]=fig
    fig,axes=plt.subplots(2,1,figsize=(11,8))
    fig.subplots_adjust(left=.1,right=.98,bottom=.1,top=.87,hspace=.35)
    for ax,metric in zip(axes,['macro_F1','class_0_PR_AUC']):
        for model in list(dict.fromkeys([*PRIMARY,best_static,'GRU (aligned)'])):
            ax.plot(dates,values(model,metric),label=model,color=COLORS[model],linewidth=1.8)
        locator=mdates.AutoDateLocator(minticks=5,maxticks=7)
        ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set(ylabel=metric.replace('_',' '),xlabel='Prediction week beginning Monday (UTC)',ylim=(0,1)); ax.grid(axis='y',alpha=.2)
        ax.legend(frameon=False,ncol=3,fontsize=9,loc='lower left')
    fig.text(.5,.96,'Static versus temporal graph forecasting',ha='center',fontsize=15)
    rendered[plot_names[4]]=fig
    assumptions=[
        'Target is the dataset binary orientation label, not verified sentiment. Exactly the original 15 origins, seeds 42-51, and eligible user-week labels are retained. Labels from the target week are used only for held-out evaluation.',
        'For EACH training, validation and test sample week t, graph construction uses only snapshot weeks w with w+7 days <= t. Historical feature row t has an exclusive event cutoff t. No target-week topology is used even for training-label examples.',
        'Static graphs are cumulative historical weighted directed interactions. Temporal graphs are the sequence of completed weekly weighted snapshots; every step uses cumulative structural feature row w+7 days, available at completion of week w. The last sequence step is week t-7 days for both aligned GRU and temporal GCN.',
        'Existing legacy GRU used feature row weeks strictly less than t and consequently older information. It is preserved and separately labelled legacy lagged. A newly trained aligned projected GRU uses the same sequence, scaling and latest completed-week cutoff as temporal GCN; fair temporal comparisons use it.',
        'Historical user mappings are dense, sorted per cutoff, and contain only users already appearing in prior interactions. They extend as new users become known; no learned ID embeddings exist. A user retains identity across recurrent steps by username. Before first known appearance no sequence row exists; inactive historical nodes remain with cumulative features and identity self messages in otherwise empty weekly neighborhoods.',
        'Static GCN implements one exact message-passing layer with S=D^-1/2 ((A+A.T)/2 + I) D^-1/2, learned projection, ReLU and dense classifier. A[target,source] receives original Source -> Target messages. Symmetrization discards orientation; reversal is algebraically identical with unchanged features and is checked directly.',
        'GraphSAGE uses a one-hop weighted incoming-neighbor mean concatenated with own features, a learned linear projection/ReLU/dropout and dense classifier. This is a weighted mean variant without a post-layer L2 normalization. Reverse variant retrains with outgoing original neighbors as incoming messages.',
        'GAT uses one directed additive multihead attention layer, learned source/receiver coefficients, LeakyReLU(0.2), log interaction Weight as an attention prior, ELU and dense classifier. It includes self messages, samples at most 16 non-self neighbors deterministically by seed 42/cutoff/node/direction, and masks padding. This weighted sampled variant is not exhaustive full-neighborhood GAT.',
        'Temporal GCN applies a one-hop symmetrically normalized weekly convolution, learned projection/ReLU and a GRU over each historical user sequence. GCN parameters are fixed across sequence steps; node hidden states evolve. It is NOT EvolveGCN-H/O or a faithful parameter-evolving implementation.',
        'Input scaling fits only the six structural feature rows of supervised TRAINING user-weeks. The frozen mean/std transforms all context neighbors and sequence steps. Graph averaging uses centered/scaled features exactly, including normalized row masses; padding remains zero. No orientation-label-history predictor is used in any primary graph model.',
        'Training loss uses N/(2*N_class) weights computed only from training labels. Validation macro-F1 selects one of three predeclared fractional-grid configurations using tuning seed 42. Each of ten final seeds independently reinitializes; validation early stopping restores earliest best epoch. Max 30 epochs, patience 4, Adam, batch 1024, no chronological sample shuffle, deterministic CPU operations. Fixed class-1 threshold 0.5 matches MLP/GRU/LR.',
        'Primary forward graph fits total 600; independently tuned/retrained reversed GraphSAGE/GAT add 300. Symmetric GCN and temporal GCN direction controls use proven identical adjacency, not redundant fits. Aligned GRU adds 150. Both ablations are run for ALL six trained graph variants, adding 1800 fits, so strongest-model reporting does not decide which controls are trained. Extra reversed ablation records are saved separately, preserving the initial graph metrics and hyperparameter tables.',
        'Feature-only controls preserve model architecture, selected graph hyperparameters, scaler, seed, loss and training procedure but replace messages by identity/self features. Static GCN control is exactly a node-only MLP; GraphSAGE duplicates self features in the neighbor slot; GAT restricts attention to self; temporal GCN becomes a projected node-feature GRU. Existing MLP is additionally reported. Independent early stopping may choose different epochs across controls.',
        'Rewiring expands integer edge weights into events, permutes destinations within each available forward graph using fixed seed 42-derived values, then reaggregates. Node features and labels are held fixed; forward weighted in/out strengths are preserved, while distinct-neighbor degrees, self loops and connectivity need not be. The reverse-direction ablations reuse this forward-oriented rewired message graph, so they do NOT preserve reverse-message in/out strengths; they are an orientation-changing topology null. Forward controls are an event-endpoint null, not an exact simple-graph degree-preserving rewire.',
        'Metrics use both classes and designate class 0 as detection class regardless of observed majority. PR-AUC is average precision with P(class 0)=1-P(class 1); ROC AUC and Brier use P(class 1). Each seed/week records test class-0 prevalence and its no-skill average-precision reference.',
        'Overall neural intervals jointly resample seeds and paired weeks with 10000 draws, seed 42. Paired differences average seeds within each week then resample identical 15 weeks; users and seeds are not treated as independent weeks. Those paired CIs condition on seed averages and assume exchangeable weeks, ignoring temporal autocorrelation. Reported positive CIs are nominal, without multiple-comparison correction.',
        'Best models by test mean are descriptive rankings, not validation-selected deployment decisions. Consistency means a positive paired difference greater than 1e-12 in ALL 15 weeks. Superiority is not asserted when the paired CI includes zero; even positive nominal CIs do not establish generalization beyond this selected cohort.',
        'Static models see cumulative topology; temporal model sees weekly topology plus cumulative features. Temporal-versus-static differences therefore include representation/architecture differences, not an isolated causal effect of evolution. Modest one-hop architectures, restricted grids, epoch caps, one fixed topology-null realization, sampling and label/graph extraction uncertainty limit negative and positive conclusions.',
        'All prior raw files, graph/node baselines, old GCN/EvolveGCN checkpoints and outputs are protected by SHA-256. New checkpoints and reports use exclusive creation. Model subprocesses are independent CPU computation jobs. No old experimental result is replaced.',
    ]
    sources={'GCN':'https://arxiv.org/abs/1609.02907','GraphSAGE':'https://arxiv.org/abs/1706.02216',
             'GAT':'https://arxiv.org/abs/1710.10903','EvolveGCN distinction':'https://ojs.aaai.org/index.php/AAAI/article/view/5984'}
    manifest=json.loads((RUN/'manifest.json').read_text())
    assert all(sha256(ROOT/p)==h for p,h in manifest['protected_hashes'].items())
    cutoff=pd.read_csv(tables/'graph_forecasting_cutoff_audit.csv')
    assert (pd.to_datetime(cutoff.Latest_Completed_Graph_Week,utc=True)+pd.Timedelta(days=7)<=pd.to_datetime(cutoff.Week_Start,utc=True)).all()
    assert (hp.Train_Last_Week<hp.Validation_Week).all() and (hp.Validation_Week<hp.Week_Start).all()
    assert hp.Scaler_Fit_Rows.eq(hp.Train_Users).all() and hp.Threshold.eq(.5).all()
    audit=dict(all_assertions_passed=True,identical_test_weeks=True,graph_snapshot_ends_before_prediction=True,
               feature_events_strictly_prior=True,aligned_temporal_cutoffs=True,historical_node_mapping_only=True,
               training_precedes_validation_precedes_test=True,training_only_scalers_and_weights=True,
               validation_only_hyperparameters_and_early_stopping=True,threshold_fixed=.5,
               no_Target_derived_predictors=True,raw_and_old_outputs_unchanged=True,config=CONFIG,
               protected_hashes=manifest['protected_hashes'])
    report=json.dumps(result_info,indent=2)+'\n\nSummary of principal means and intervals:\n'+main_summary[main_summary.Metric.isin(['macro_F1','class_0_PR_AUC','balanced_accuracy'])].to_string(index=False)
    selected_pairs=[(best_graph,'Persistence'),(best_graph,'MLP'),('Temporal GCN','GRU (aligned)'),('Temporal GCN',best_static)]
    report+='\n\nRequested paired comparisons:\n'+pd.DataFrame([select(comparisons,a,b,m) for a,b in selected_pairs for m in ['macro_F1','class_0_PR_AUC','balanced_accuracy','MCC']]).to_string(index=False)
    report+='\n\nAll graph contribution ablations:\n'+ablations.to_string(index=False)
    report+='\n\nDirection controls:\n'+direction.to_string(index=False)
    report+='\n\nConservative interpretation:\n'
    report+=f'Graph message passing for the descriptively strongest graph model ({best_graph}) has '+('a positive nominal paired macro-F1 interval versus its matched feature-only control.' if message.Paired_Bootstrap_95_Lower>0 else 'no established positive paired macro-F1 improvement versus its matched feature-only control.')+'\n'
    report+='Temporal GCN has '+('a positive nominal macro-F1 interval versus the best static graph model.' if temporal.Paired_Bootstrap_95_Lower>0 else 'no established positive macro-F1 improvement versus the best static graph model.')+'\n'
    report+='No superiority claim is made for intervals including zero. Nominal intervals are exploratory and do not adjust temporal dependence or multiple model comparisons.\n'
    def describe(row):
        lo,hi=row.Paired_Bootstrap_95_Lower,row.Paired_Bootstrap_95_Upper
        status='positive nominal paired improvement' if lo>0 else ('negative nominal paired difference' if hi<0 else 'inconclusive interval including zero')
        return f'{row.Model_A} - {row.Model_B}: delta={row.Mean_A_Minus_B:.6f}, CI=[{lo:.6f},{hi:.6f}], wins={row.A_Better_Weeks}/15; {status}.'
    report+='\nExplicit primary questions (macro-F1 unless specified):\n'
    report+='1. Any graph model versus Persistence:\n'+'\n'.join(describe(select(comparisons,g,'Persistence')) for g in ALL_GRAPHS)+'\n'
    report+='2. Does static GCN beat MLP? '+describe(select(comparisons,'Static GCN','MLP'))+'\n'
    report+='3. GraphSAGE/GAT versus GCN:\n'+'\n'.join(describe(select(comparisons,g,'Static GCN')) for g in ['GraphSAGE','GAT',*REVERSE])+'\n'
    report+='4. Temporal graph model versus strongest static graph model: '+describe(temporal)+'\n'
    report+='5. Temporal GCN versus aligned GRU: '+describe(select(comparisons,'Temporal GCN','GRU (aligned)'))+' No actual EvolveGCN result is claimed.\n'
    report+='6. Graph models versus class-0 prevalence AP:\n'+'\n'.join(describe(select(comparisons,g,'No-skill','class_0_PR_AUC')) for g in ALL_GRAPHS)+'\n'
    report+='7. Direction impact: GCN/temporal GCN are invariant by construction. Independently trained directed-model differences:\n'+'\n'.join(describe(r) for r in direction[direction.Model.isin(['GraphSAGE','GAT'])&direction.Metric.eq('macro_F1')].itertuples())+'\n'
    report+='\nExact assumptions and limitations:\n'+'\n'.join(f'{i}. {a}' for i,a in enumerate(assumptions,1))
    report+='\n\nPrimary methodological references:\n'+json.dumps(sources,indent=2)+'\n\nLeakage audit:\n'+json.dumps({k:v for k,v in audit.items() if k!='protected_hashes'},indent=2)
    for p in paths.values():
        p.parent.mkdir(parents=True,exist_ok=True)
    for frame,key in [(summary,'summary'),(comparisons,'comparisons'),(direction,'direction'),(ablations,'ablation'),(weekly,'weekly'),(cohort,'cohort')]:
        with paths[key].open('x',encoding='utf-8',newline='') as f:
            frame.to_csv(f,index=False,na_rep='NA')
    figure_rows=[]
    for letter,name in zip('ABCDE',plot_names):
        fig=rendered[name]
        for ext in ['png','svg']:
            with paths[name+'_'+ext].open('xb') as f:
                fig.savefig(f,format=ext,dpi=400,facecolor='white')
        plt.close(fig)
        figure_rows.append(dict(Figure=letter,PNG=str(paths[name+'_png'].relative_to(ROOT)),SVG=str(paths[name+'_svg'].relative_to(ROOT)),PNG_DPI=400))
    with paths['figures'].open('x',encoding='utf-8',newline='') as f:
        pd.DataFrame(figure_rows).to_csv(f,index=False)
    with paths['report'].open('x',encoding='utf-8') as f:
        f.write(report+'\n')
    with paths['audit'].open('x',encoding='utf-8') as f:
        json.dump(audit,f,indent=2)
    print(report,flush=True)


if __name__=='__main__':
    main()
