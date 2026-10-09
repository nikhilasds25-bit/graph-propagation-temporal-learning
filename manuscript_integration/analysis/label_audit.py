from pathlib import Path
import pandas as pd, numpy as np, json
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
ROOT=Path(__file__).resolve().parents[1]; data=ROOT/'data_raw'; out=ROOT/'source'; tab=out/'results_tables_revised'; fig=out/'figures_revised'
tw=pd.read_csv(data/'master_tweets.csv')
# compute lexical polarity full data
pol=[]
for i,t in enumerate(tw['Tweet'].fillna('').astype(str)):
    try: pol.append(TextBlob(t).sentiment.polarity)
    except Exception: pol.append(np.nan)
tw['textblob_polarity']=pol
tw['textblob_sign']=pd.cut(tw.textblob_polarity,[-np.inf,-1e-12,1e-12,np.inf],labels=['negative','neutral','positive'])
ct=pd.crosstab(tw.Target,tw.textblob_sign,dropna=False); ct.to_csv(tab/'target_vs_textblob_full_crosstab.csv')
means=tw.groupby('Target').textblob_polarity.agg(['count','mean','median','std']); means.to_csv(tab/'target_textblob_full_summary.csv')
# check deterministic sign mapping accuracy for target1=positive, target0=non-positive/negative; two possible rules
pred_pos=(tw.textblob_polarity>0).astype(int); eq=(pred_pos==tw.Target).mean()
pred_nonneg=(tw.textblob_polarity>=0).astype(int); eq2=(pred_nonneg==tw.Target).mean()
summary={'agreement_target1_with_polarity_gt0':float(eq),'agreement_target1_with_polarity_ge0':float(eq2),'n':int(len(tw)),'mean_by_target':means['mean'].to_dict()}
(tab/'label_audit_summary.json').write_text(json.dumps(summary,indent=2))
# figure density hist with zero share annotation
figu,axs=plt.subplots(1,2,figsize=(7.2,3.0))
bins=np.linspace(-1,1,61)
for target,c,label in [(0,'#D55E00','Target 0'),(1,'#0072B2','Target 1')]:
    vals=tw.loc[tw.Target==target,'textblob_polarity'].dropna(); axs[0].hist(vals,bins=bins,density=True,histtype='step',lw=1.5,color=c,label=f'{label} (n={len(vals):,})')
axs[0].axvline(0,color='0.4',ls='--',lw=.8); axs[0].set_xlabel('TextBlob polarity'); axs[0].set_ylabel('Density'); axs[0].set_title('(a) Full-data lexical polarity'); axs[0].legend(frameon=False,fontsize=7)
row=ct.div(ct.sum(axis=1),axis=0)
im=axs[1].imshow(row.values,vmin=0,vmax=1,cmap='Blues'); axs[1].set_xticks(range(len(row.columns))); axs[1].set_xticklabels(row.columns); axs[1].set_yticks(range(len(row.index))); axs[1].set_yticklabels([f'Target {i}' for i in row.index]); axs[1].set_xlabel('TextBlob sign'); axs[1].set_ylabel('Dataset label'); axs[1].set_title('(b) Label vs lexical sign')
for i in range(row.shape[0]):
  for j in range(row.shape[1]): axs[1].text(j,i,f'{row.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
figu.colorbar(im,ax=axs[1],fraction=.046,pad=.04,label='Row fraction')
figu.tight_layout(); figu.savefig(fig/'fig_label_audit_textblob.pdf',bbox_inches='tight'); figu.savefig(fig/'fig_label_audit_textblob.png',dpi=220,bbox_inches='tight'); plt.close(figu)
print(summary); print(ct); print(means)
