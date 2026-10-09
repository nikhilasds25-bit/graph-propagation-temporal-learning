"""Copy saved numerical results into LaTeX; no estimation or model execution."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript_integration/final_manuscript'
T=ROOT/'results/tables'
s=pd.read_csv(T/'secondary_reach_simulation_results.csv')
e=pd.read_csv(T/'competition_exposure_overlap.csv')
def get(df,c,m,d='recorded'):
    return df[(df.configuration==c)&(df.metric==m)&(df.direction==d)].iloc[0]
macros=[]
for name,df,metric,digits,multiply in [
    ('Secondary',s,'secondary_total',6,1),('Amplification',s,'amplification_total',8,1),
    ('ContestedPercent',e,'contested_node_fraction',8,100),
    ('Resolution',e,'competition_resolution_events',6,1),
    ('SecondaryOutcome',s,'C0_minus_C1_secondary',6,1)]:
    r=get(df,'full_balanced',metric)
    for suffix,col in [('Mean','mean'),('Low','ci95_low'),('High','ci95_high')]:
        macros.append(r'\newcommand{\Evidence'+name+suffix+'}{'+f'{r[col]*multiply:.{digits}f}'+'}')
(OUT/'evidence_numbers.tex').write_text('\n'.join(macros)+'\n',encoding='utf-8')
names=[('baseline_beta_005','Edge-only, low transmission','0.05 / 0.05','25 / 25'),
 ('symmetric_baseline','Edge-only, balanced','0.14 / 0.14','25 / 25'),
 ('baseline_beta_030','Edge-only, high transmission','0.30 / 0.30','25 / 25'),
 ('full_balanced','Full, balanced','0.14 / 0.14','25 / 25'),
 ('Target_0_advantage','Full, Target-0 advantage','0.28 / 0.14','25 / 25'),
 ('Target_1_advantage','Full, Target-1 advantage','0.14 / 0.28','25 / 25'),
 ('asymmetric_40_10','Full, asymmetric allocation','0.14 / 0.14','40 / 10'),
 ('asymmetric_10_40','Full, asymmetric allocation','0.14 / 0.14','10 / 40')]
lines=[r'\begin{table}[htbp]\centering\small',
 r'\caption{Recorded-direction secondary propagation on the 17,598-user WCC. Each row contains 2,000 runs with quiescence probability 0.25 and a 25-step horizon. Intervals quantify Monte Carlo uncertainty conditional on the graph and configuration.}\label{tab:simulation}',
 r'\begin{tabularx}{\textwidth}{@{}Xccp{3.4cm}r@{}}\toprule Configuration & $\beta_0/\beta_1$ & Seeds 0/1 & Secondary mean [95\% CI] & Amplification\\\midrule']
for c,name,beta,seeds in names:
    r=get(s,c,'secondary_total'); a=get(s,c,'amplification_total')
    lines.append(f'{name} & {beta} & {seeds} & '+f'{r["mean"]:.4f} [{r.ci95_low:.4f}, {r.ci95_high:.4f}] & {a["mean"]:.5f}'+r'\\')
lines += [r'\bottomrule\end{tabularx}\end{table}']
(OUT/'simulation_results_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Typeset saved simulation values and exact result bindings.')
