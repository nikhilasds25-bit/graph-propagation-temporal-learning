"""Assemble manuscript assets from saved evidence; never run analysis."""
from pathlib import Path
import hashlib, json, shutil
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'manuscript_integration'
OUT=BASE/'final_manuscript'
FIGS=ROOT/'results/figures/reviewer_corrected'
ASSETS=['population_funnel_separate_units','weekly_activity_split_panels',
        'textblob_zero_spike_visible','degree_strength_loglog_CCDF',
        'original_WCC_bowtie_readable','typed_layer_sizes_readable_log',
        'layer_orientation_matrices','layer_orientation_null_forest',
        'static_vs_temporal_reach','temporal_reach_by_seed_stratum',
        'simulation_amplification_distribution','simulation_secondary_reach_vs_beta',
        'simulation_contested_fraction','simulation_direction_forest',
        'competition_secondary_outcome','sensitivity_amplification_independent_scales',
        'forecasting_reconciled_uncertainty']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def prepare():
    OUT.mkdir(exist_ok=False)
    (OUT/'figures').mkdir()
    (OUT/'supporting_evidence').mkdir()
    (BASE/'qa').mkdir(exist_ok=True)
    paths=[]
    for directory in ['results/tables','results/figures/reviewer_corrected','reproducibility_package']:
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    paths += [BASE/'source_main.tex',BASE/'source_references.bib',BASE/'references.bib']
    protected={str(p.relative_to(ROOT)):sha(p) for p in paths}
    (BASE/'protected_integration_hashes.json').write_text(json.dumps(protected,indent=2),encoding='utf-8')
    shutil.copy2(BASE/'main.tex',BASE/'main_before_integration.tex')
    shutil.copy2(BASE/'main.tex',OUT/'main.tex')
    shutil.copy2(BASE/'references.bib',OUT/'references.bib')
    for stem in ASSETS:
        for suffix in ['.png','.svg','_source.csv']:
            dest=OUT/('supporting_evidence' if suffix=='_source.csv' else 'figures')/(stem+suffix)
            shutil.copy2(FIGS/(stem+suffix),dest)
    for name in ['analysis_populations_table.tex','temporal_reach_table.tex','forecasting_results_table.tex']:
        shutil.copy2(BASE/name,OUT/name)
    # Canonical mean/interval values retained verbatim; change only display names.
    p=OUT/'forecasting_results_table.tex'
    s=p.read_text(encoding='utf-8').replace('Node-only GRU (original GRU)','Original GRU')
    s=s.replace('EvolveGCN style','GCN + GRU head (static graph)').replace('GCN baseline','Monthly GCN')
    s=s.replace('Weekly forecasting and cohort-specific comparison results.','Complete forecasting results with separate weekly cohorts and a monthly comparison task.')
    p.write_text(s,encoding='utf-8')
    p=OUT/'temporal_reach_table.tex'
    p.write_text(p.read_text().replace('layer aware','layer-aware').replace('high out strength','high out-strength'),encoding='utf-8')
    # Display-only terminology change in a copied SVG; all values are preserved.
    p=OUT/'figures/sensitivity_amplification_independent_scales.svg'
    p.write_text(p.read_text(encoding='utf-8').replace('>Recovery<','>Quiescence probability<'),encoding='utf-8')
    for start in range(0,len(ASSETS),6):
        names=ASSETS[start:start+6]
        sheet=Image.new('RGB',(1600,1000),'white'); draw=ImageDraw.Draw(sheet)
        for j,stem in enumerate(names):
            im=Image.open(FIGS/(stem+'.png')).convert('RGB')
            im.thumbnail((775,275))
            x=(j%2)*800; y=(j//2)*333
            sheet.paste(im,(x+(790-im.width)//2,y+30))
            draw.text((x+12,y+6),stem,fill='black')
        sheet.save(BASE/'qa'/f'canonical_figures_{start//6+1}.png')
    print(f'Prepared {len(ASSETS)} figures; protected {len(protected)} existing files.')

if __name__=='__main__': prepare()
