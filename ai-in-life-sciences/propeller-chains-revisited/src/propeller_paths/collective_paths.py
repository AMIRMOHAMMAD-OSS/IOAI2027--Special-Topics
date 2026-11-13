"""Reproduce complete synchronized candidate paths from compact saddle seeds.

This recomputes local constrained minima, follows central stable branches,
refines two constrained connector saddles, verifies their feasible downhill
endpoints, and audits a dense complete A-to-B path. It does not claim a global
minimum over all synchronized curves.
"""
import json,time
from pathlib import Path
import numpy as np
from .collective import endpoint_targets,run_branch
from .constraints import kkt_polish,kkt_state
from .continuation import follow_branch
from .bridges import connect_from_constrained_saddle,proper_same_identity_fit
from .assembly import read_block,assemble_blocks


def same_identity_rmsd(reference,candidate):
    fitted,_=proper_same_identity_fit(reference,candidate)
    return float(np.sqrt(np.mean(np.sum((fitted-np.asarray(reference).reshape(-1))
                                       .reshape(-1,3)**2,axis=1))))


def complete_collective_path(ctx,pattern,out,seed_file,atom_step=.01):
    """One self-contained calculation; no previous run files are required."""
    started=time.monotonic()
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    seeds=json.loads(Path(seed_file).read_text())['patterns'][pattern]
    a_xyz,b_xyz,a,delta=endpoint_targets(ctx,pattern)
    run_branch(ctx,pattern,'forward',out/'forward_nodes',intervals=40,
               gradient_tol=1e-7,constraint_tol=1e-10,audit=False)
    coarse=np.load(out/'forward_nodes/nodes.npz')
    records={k:json.loads((out/'forward_nodes'/f'node_{k:03}'/'result.json').read_text())
             for k in (18,20,22)}
    left=float(seeds['left']['progress']);right=float(seeds['right']['progress'])
    starts=[('outer_left',18,left),('middle_left',20,left),
            ('middle_right',20,right),('outer_right',22,right)]
    for name,k,end in starts:
        rec=records[k];r=follow_branch(ctx,rec['xyz_A'],rec['progress'],end,a,delta,out/name)
        if r['status']!='SMOOTH_KKT_BRANCH_REACHED_REQUESTED_END':
            raise ValueError('Local branch did not reach the connector: '+name)
    branch={name:read_block(out/name) for name,_,_ in starts}
    connectors={}
    for name,t,refs in [('left',left,{'start_minimum':branch['outer_left'][0][-1],
                                    'end_minimum':branch['middle_left'][0][-1]}),
                        ('right',right,{'start_minimum':branch['middle_right'][0][-1],
                                       'end_minimum':branch['outer_right'][0][-1]})]:
        seed=seeds[name]
        x=np.array(seed['integer_coordinates'],dtype=float).reshape(-1)/seed['scale_per_angstrom']
        x,trace,history,status=kkt_polish(ctx,x,a+t*delta,1e-11,1e-8,max_steps=20,atom_cap=.01)
        state=kkt_state(ctx,x,a+t*delta)
        state.update(xyz_A=x.reshape(-1,3).tolist(),progress=t,refinement_status=status,
                     origin='revalidation_of_recorded_constrained_connector_seed')
        (out/(name+'_seed_refinement.json')).write_text(json.dumps(state,indent=2)+'\n')
        if status!='KKT_CONVERGED':raise ValueError('Connector seed did not refine')
        r=connect_from_constrained_saddle(ctx,x,a+t*delta,refs,out/(name+'_connector'),progress=t)
        if r['status']!='CONSTRAINED_CONNECTOR_VERIFIED':
            raise ValueError('Connector failed physical checks: '+name)
        cx,ct=read_block(out/(name+'_connector'),'connector.npz')
        if same_identity_rmsd(refs['start_minimum'],cx[-1])<same_identity_rmsd(refs['start_minimum'],cx[0]):
            cx,ct=cx[::-1],ct[::-1]
        connectors[name]=(cx,ct)
    blocks=[('forward_nodes_0_to_045',*read_block(out/'forward_nodes',low=0,high=.45)),
            ('outer_left',*branch['outer_left']),
            ('left_connector',*connectors['left']),
            ('middle_left_reversed',branch['middle_left'][0][::-1],branch['middle_left'][1][::-1]),
            ('middle_right',*branch['middle_right']),
            ('right_connector',*connectors['right']),
            ('outer_right_reversed',branch['outer_right'][0][::-1],branch['outer_right'][1][::-1]),
            ('forward_nodes_055_to_1',*read_block(out/'forward_nodes',low=.55,high=1))]
    report=assemble_blocks(ctx,blocks,a,delta,out/'complete_path',pattern,atom_step=atom_step)
    if report['status']!='COMPLETE_EXACT_FEASIBLE_COLLECTIVE_PATH':
        raise ValueError('Complete synchronized path failed an acceptance gate')
    report['complete_workflow_elapsed_seconds']=time.monotonic()-started
    (out/'workflow_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
