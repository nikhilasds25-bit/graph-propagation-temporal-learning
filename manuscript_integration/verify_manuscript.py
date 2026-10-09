"""Verify manuscript integration and protected bytes; no analysis execution."""
from pathlib import Path
import hashlib,json,re
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'manuscript_integration'; OUT=BASE/'final_manuscript'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def audit():
    ledger=json.loads((BASE/'protected_integration_hashes.json').read_text())
    changes=[n for n,h in ledger.items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
    checks={'protected_files_unchanged':not changes,'protected_file_count':len(ledger)}
    checks['primary_main_tex_unchanged']=sha(BASE/'main.tex')==sha(BASE/'main_before_integration.tex')
    main=(OUT/'main.tex').read_text(encoding='utf-8')
    backup=(BASE/'source_main.tex').read_text(encoding='utf-8')
    for cmd in ['title','author','date']:
        extract=lambda t:re.search(r'\\'+cmd+r'\{(.+)\}',t).group(1)
        checks[cmd+'_preserved']=extract(main)==extract(backup)
    checks['bibliography_byte_identical']=sha(OUT/'references.bib')==sha(BASE/'source_references.bib')
    source=main+'\n'+'\n'.join((OUT/(n+'.tex')).read_text(encoding='utf-8') for n in re.findall(r'\\input\{([^}]+)\}',main))
    labels=set(re.findall(r'\\label\{([^}]+)\}',source))
    figure_calls=re.findall(r'\\resultfigure\{([^}]+)\}\{.*?\}\{(fig:[^}]+)\}',main)
    for block in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',main,re.S):
        asset=re.search(r'\\includegraphics\[[^\]]*\]\{figures/([^}]+)\}',block)
        label=re.search(r'\\label\{(fig:[^}]+)\}',block)
        if asset and label:figure_calls.append((asset.group(1),label.group(1)))
    labels.update(label for _,label in figure_calls)
    labels.discard('#3')
    refs=set(re.findall(r'\\ref\{([^}]+)\}',source))
    checks['undefined_source_references']=sorted(refs-labels)
    checks['uncited_figures_or_tables']=sorted(l for l in labels if l.startswith(('fig:','tab:')) and l not in refs)
    checks['figure_count']=len(figure_calls)
    checks['figure_files_present']=all((OUT/'figures'/n).is_file() for n,_ in figure_calls)
    checks['figure_source_data_present']=all((OUT/'supporting_evidence'/(Path(n).stem+'_source.csv')).is_file() for n,_ in figure_calls)
    checks['no_keywords']='Keywords' not in main
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',main,re.S).group(1)
    words=re.findall(r"\S+",abstract)
    checks['abstract_words']=len(words); checks['abstract_length_170_220']=170<=len(words)<=220
    checks['duplicate_references_heading_absent']=r'\section*{References}' not in main
    phrases=['original','earlier','previous implementation','corrected','revision','first implementation',
             'Overleaf archive','EvolveGCN','epidemic threshold','negative dominance','trust score',
             'recovery probability','weakly strongly connected']
    occurrences={p:[source[max(0,m.start()-50):m.end()+100].replace('\n',' ') for m in re.finditer(re.escape(p),source,re.I)] for p in phrases}
    checks['language_occurrences_for_review']=occurrences
    f=pd.read_csv(ROOT/'results/tables/all_forecasting_models_reconciled.csv')
    table=(OUT/'forecasting_results_table.tex').read_text()
    metrics=['macro_F1','balanced_accuracy','MCC','class_0_PR_AUC','ROC_AUC','Brier_score']
    forecast=[]
    for _,r in f.iterrows():
        strings=[]
        for c in metrics:
            if pd.notna(r[c]):
                v=f'{r[c]:.4f}'
                if pd.notna(r[c+'_ci95_low']): v+=r'\newline ['+f'{r[c+"_ci95_low"]:.4f}, {r[c+"_ci95_high"]:.4f}]'
                else: v+=r'\newline (CI unavailable)'
            else:v='NA'
            strings.append(v)
        expected=' & '.join(strings)
        forecast.append({'model':r.model,'cohort':r.cohort,'saved_mean_and_interval_strings_present':expected in table})
    checks['forecast_rows']=len(forecast)
    checks['all_27_forecast_rows_match_canonical']=all(x['saved_mean_and_interval_strings_present'] for x in forecast) and len(forecast)==27
    checks['funding_exact']='No external funding was received for this study.' in main
    (BASE/'qa/manuscript_source_audit.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    (BASE/'qa/forecast_table_audit.json').write_text(json.dumps(forecast,indent=2),encoding='utf-8')
    assert not changes,changes
    assert not checks['undefined_source_references'],checks
    assert not checks['uncited_figures_or_tables'],checks
    assert all(checks[k] for k in ['title_preserved','author_preserved','date_preserved','bibliography_byte_identical',
       'figure_files_present','figure_source_data_present','no_keywords','abstract_length_170_220',
       'duplicate_references_heading_absent','all_27_forecast_rows_match_canonical','funding_exact']),checks
    print(json.dumps({k:v for k,v in checks.items() if k!='language_occurrences_for_review'},indent=2))
    print('Language review:',json.dumps(occurrences,indent=2))
if __name__=='__main__':audit()
