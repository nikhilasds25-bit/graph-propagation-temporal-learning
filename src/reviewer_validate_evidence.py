"""Independent invariants and adverse-case checks for new reviewer evidence."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import networkx as nx
import reviewer_correction_analysis as a

ROOT=Path(__file__).resolve().parents[1]

def main():
    checks=[]
    def check(name,ok,detail=''):
        checks.append(dict(check=name,passed=bool(ok),detail=detail))
        if not ok: raise AssertionError(name+': '+detail)
    # Equal timestamp cascades and later-before-earlier reversal are adversarial cases.
    adj={'a':([5],['b']),'b':([5],['c']),'c':([4],['d'])}
    reached=a.arrival('a',adj,0)
    check('equal-time closure and backwards-time exclusion',set(reached)=={'b','c'} and reached['c'][0]==5)
    adj={'a':([1,7],['b','c']),'b':([2],['d']),'c':([8],['d'])}
    check('earliest-arrival alternative paths',a.arrival('a',adj,0)['d'][0]==2)
    check('self-loop does not create a descendant','a' not in a.arrival('a',{'a':([0],['a'])},0))
    check('username one-at rule',a.norm(' @@AbC ')=='@abc' and a.norm(' @AbC ')=='abc')
    check('RNG domain separation',not np.array_equal(a.keyed_uniform(np.arange(50),1,0,0),a.keyed_uniform(np.arange(50)+30,0,1,0)))
    n=4; features=(np.ones(n),np.ones(n),np.ones(n),np.ones(n),np.zeros(n))
    # Guaranteed simultaneous adoption with beta=1; one competition resolution.
    r=a.simulator([[(2,1.)],[(2,1.)],[],[]],n,[0],[1],1,1,0,features,False)
    check('competition instrument and no seed subtraction artifact',r['secondary_total']==1 and r['exposed_both']==1 and r['competition_resolution_events']==1)
    z=a.simulator([[],[],[],[]],n,[0],[1],0,0,0,features,False)
    check('zero-adoption outcome NA',np.isnan(z['C0_minus_C1_secondary']) and z['outcome_secondary']=='NA' and z['secondary_total']==0)
    p=ROOT/'results/tables'
    d=pd.read_csv(p/'time_respecting_reachability_by_seed.csv')
    check('temporal reach bounded by static',bool((d.temporal_reachable<=d.static_out_component).all()))
    check('500 seeds or entire smaller population in every temporal stratum',all(len(x)==min(500,int(x.graph_nodes.iloc[0])) for _,x in d.groupby(['layer','direction','stratum'])))
    check('undefined reach ratio for static sink',bool(d.loc[d.static_out_component==0,'temporal_static_ratio'].isna().all()))
    s=pd.read_csv(p/'secondary_reach_simulation_runs.csv')
    check('2000 paired runs per cell',all(len(x)==2000 for _,x in s.groupby(['configuration','direction'])))
    check('simulation identities',bool((s.secondary_0==s.final_reached_0-s.seed_0).all() and (s.secondary_1==s.final_reached_1-s.seed_1).all() and (s.secondary_total==s.secondary_0+s.secondary_1).all()))
    check('simulation population conservation bounds',bool(((s.secondary_total+s.seed_0+s.seed_1)<=s.graph_nodes).all() and (s.secondary_total>=0).all()))
    check('exposure union identity',bool((s.exposed_union==s.exposed_0_only+s.exposed_1_only+s.exposed_both).all()))
    check('secondary NA denominator',bool(s.loc[s.secondary_total==0,'C0_minus_C1_secondary'].isna().all()))
    check('exposure fraction denominator',np.allclose(s.contested_node_fraction,s.exposed_both/(s.graph_nodes-s.seed_0-s.seed_1)))
    check('paired seed allocations',all(x.seed_0.nunique()==1 and x.seed_1.nunique()==1 for _,x in s.groupby(['configuration','run'])))
    q=pd.read_csv(p/'spectral_threshold_audit.csv')
    for key,g in q.groupby(['population','self_loops_retained']):
        rec=g[g.direction=='recorded'].iloc[0]; rev=g[g.direction=='reversed'].iloc[0]
        check('transpose spectrum '+str(key),np.isclose(rec.spectral_radius_weighted,rev.spectral_radius_weighted,rtol=1e-10))
    check('loop removal cannot increase spectral radius',all(x[x.self_loops_retained].spectral_radius_weighted.iloc[0]>=x[~x.self_loops_retained].spectral_radius_weighted.iloc[0]-1e-10 for _,x in q.groupby(['population','direction'])))
    q=pd.read_csv(p/'layer_orientation_permutation_tests.csv')
    check('undefined null cannot produce significance',q.loc[q.undefined_null_draws>0,'p_two_sided'].isna().all())
    check('2000 permutations per null/layer',(q.permutations==2000).all())
    check('Holm cannot reduce p',(q.Holm_p.dropna()>=q.loc[q.Holm_p.notna(),'p_two_sided']-1e-15).all())
    q=pd.read_csv(p/'forecasting_saved_mean_consistency.csv')
    check('saved forecasting means independently match weekly outputs',q.within_1e_10.all())
    layers=pd.read_csv(p/'typed_layer_network_statistics.csv'); events=pd.read_csv(ROOT/'data/processed/reconstructed_temporal_edges.csv')
    q=layers[(layers.direction=='recorded')&(layers.layer!='pooled')]
    check('typed layer event conservation',q.events.sum()==len(events))
    for image in (ROOT/'results/figures/reviewer_corrected').glob('*.png'):
        from PIL import Image
        with Image.open(image) as im:
            check('400-DPI PNG '+image.stem,abs(im.info.get('dpi',(0,))[0]-400)<1)
        check('paired SVG and source CSV '+image.stem,image.with_suffix('.svg').exists() and image.with_name(image.stem+'_source.csv').exists())
    # Maximum reported temporal hops are algorithm witnesses, not a longest-walk statistic.
    a.txt('temporal_hop_interpretation_clarification.txt','The saved max_earliest_arrival_witness_hops counts hops in the actual foremost-path witnesses retained by earliest-arrival relaxation. It is not the maximum possible temporal walk and is not guaranteed to be the globally fewest-hop foremost path: an earlier intermediate arrival can dominate a later shorter path. Reachability and arrival times are exact under the specified non-decreasing timestamp definition; witness depth is descriptive only. This clarification supersedes the phrase shortest witness in the initial temporal report.')
    a.csv('reviewer_evidence_validation.csv',checks)
    print(f'Independent checks passed: {len(checks)}',flush=True)

if __name__=='__main__': main()
