"""Package manuscript-only files and copied derived supporting evidence."""
from pathlib import Path
import json,shutil,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'manuscript_integration'; OUT=BASE/'final_manuscript'
TABLES=['all_forecasting_models_reconciled.csv','matching_sensitivity_comparison.csv',
 'reconstruction_overlap_reconciliation.csv','target_textblob_agreement_recheck.csv',
 'typed_layer_network_statistics.csv','layer_orientation_mixing.csv',
 'layer_orientation_permutation_tests.csv','time_respecting_reachability_summary.csv',
 'secondary_reach_simulation_results.csv','competition_exposure_overlap.csv',
 'direction_simulation_comparison.csv','competition_outcome_definition_comparison.csv',
 'spectral_threshold_audit.csv','forecasting_population_and_information_boundary.csv',
 'stage8_final_paper_results.csv']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def supporting():
    dest=OUT/'supporting_evidence/tables';dest.mkdir(exist_ok=True)
    for name in TABLES:shutil.copy2(ROOT/'results/tables'/name,dest/name)
    for name in ['typed_direction_assumptions.txt','temporal_hop_interpretation_clarification.txt','aligned_GRU_definition.txt','history_graph_findings_addendum.txt']:
        shutil.copy2(ROOT/'results/tables'/name,dest/name)
    (OUT/'README.md').write_text('''# Manuscript typesetting package

Primary source: `main.tex`. Bibliography: `references.bib` (all 22 supplied entries retained).
Upload the entire folder contents to Overleaf and select `main.tex` as the main document.
Use XeLaTeX; pdfLaTeX is also compatible with the supplied packages and figure formats.
With a local standard TeX distribution: XeLaTeX, BibTeX, XeLaTeX, XeLaTeX.

Tables are loaded from five local `.tex` fragments. Referenced figures are in `figures/`.
All figures have an SVG, a 400-DPI PNG, and a saved source CSV in `supporting_evidence/`.
The sensitivity figure also has a PDF, with quiescence terminology typeset from its saved grid.
No raw tweet text, model checkpoint, training output, or analysis environment is included.

`supporting_evidence/tables/` contains copied canonical derived evidence and assumptions.
These tables preserve saved cohort and method identifiers; display names in the manuscript
describe the architecture and task. No estimates are combined across cohorts.
`manuscript_manifest.json` gives SHA-256 hashes and provenance for package inputs.
The project reproducibility package contains the analysis scripts and linked output manifests;
this typesetting ZIP is not an independently executable analysis release or licence grant.
''',encoding='utf-8')
def package():
    files=[OUT/'main.tex',OUT/'references.bib',OUT/'README.md']
    files+=sorted(OUT.glob('*_table.tex'))+[OUT/'evidence_numbers.tex']
    for p in (OUT/'figures').iterdir():
        if p.suffix in ['.png','.svg','.pdf']:files.append(p)
    files+=sorted(p for p in (OUT/'supporting_evidence').rglob('*') if p.is_file())
    manifest=[]
    for p in files:
        rel=p.relative_to(OUT).as_posix()
        source='manuscript integration'
        if rel.startswith('figures/') or rel.startswith('supporting_evidence/'):
            canonical=(ROOT/'results/tables'/p.name) if '/tables/' in rel else ROOT/'results/figures/reviewer_corrected'/p.name
            if canonical.is_file():
                source=str(canonical.relative_to(ROOT)).replace('\\','/')
                if sha(canonical)!=sha(p):source+='; manuscript-only typography/layout (saved values retained)'
            elif rel.startswith('figures/') and (ROOT/'results/figures/reviewer_corrected'/(p.stem+'_source.csv')).is_file():
                source='results/figures/reviewer_corrected/'+p.stem+'_source.csv; manuscript-only vector typesetting'
        manifest.append({'path':rel,'SHA256':sha(p),'source':source})
    m=OUT/'manuscript_manifest.json';m.write_text(json.dumps(manifest,indent=2),encoding='utf-8');files.append(m)
    archive=BASE/'final_manuscript_overleaf.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(p.relative_to(OUT).as_posix() for p in files)
        assert 'main.tex' in z.namelist() and 'references.bib' in z.namelist()
        assert not any(p.endswith(('.aux','.log','.blg','.bbl','.xdv')) for p in z.namelist())
        assert not any('data/raw' in p or 'master_tweets.csv' in p for p in z.namelist())
    print(f'Packaged {len(files)} source/supporting files; ZIP integrity passed; {archive.stat().st_size:,} bytes.')
if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='supporting':supporting()
    else:package()
