"""Predeclared identity/null controls for both reversed graph variants."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import pandas as pd
from weekly_graph_forecasting import ROOT,RUN,execute_job
from build_orientation_baselines import sha256


def main():
    metrics_path=ROOT/'results/tables/graph_direction_ablation_metrics.csv'
    hp_path=ROOT/'results/tables/graph_direction_ablation_hyperparameters.csv'
    if metrics_path.exists() or hp_path.exists():
        raise FileExistsError('Direction control outputs already exist')
    manifest=json.loads((RUN/'manifest.json').read_text())
    assert all(sha256(ROOT/p)==h for p,h in manifest['protected_hashes'].items())
    protocol=pd.read_csv(ROOT/'results/tables/rolling_origin_baseline_protocol.csv')
    rows,params=[],[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(execute_job,origin,name,'reverse',True) for origin in protocol.to_dict('records') for name in ['GraphSAGE','GAT']]
        for i,future in enumerate(as_completed(tasks),1):
            m,h,_=future.result(); rows+=m; params+=h
            print(f'Reversed control jobs completed {i}/30',flush=True)
    rows=pd.DataFrame(rows).sort_values(['Week_Start','Model','Control','Seed'])
    assert len(rows)==600
    assert all(sha256(ROOT/p)==h for p,h in manifest['protected_hashes'].items())
    for path,frame in [(metrics_path,rows),(hp_path,pd.DataFrame(params))]:
        with path.open('x',encoding='utf-8',newline='') as f:
            frame.to_csv(f,index=False,na_rep='NA')


if __name__=='__main__':
    main()
