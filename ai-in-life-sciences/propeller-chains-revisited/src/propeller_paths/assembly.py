"""Assemble fixed-identity branch blocks and explicit feasible connectors."""
import json,time
from pathlib import Path
import numpy as np
from .bridges import proper_same_identity_fit,dense_feasible_polyline

def read_block(folder,name='nodes.npz',low=-np.inf,high=np.inf,reverse=False):
    data=np.load(Path(folder)/name);x=data['coordinates_angstrom'];t=data['progress'];take=(t>=low-1e-10)&(t<=high+1e-10);x=x[take];t=t[take]
    if reverse:x=x[::-1];t=t[::-1]
    return x.reshape(-1,297),t


def assemble_blocks(ctx,blocks,start_angles,winding,out,pattern,atom_step=.01):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);positions=[];progress=[];sources=[];begun=time.perf_counter()
    for name,x,t in blocks:
        x=np.asarray(x).reshape(-1,99,3).copy();t=np.asarray(t)
        if positions:
            fitted,fit=proper_same_identity_fit(positions[-1],x[0]);rot=np.array(fit['rotation']);translation=np.array(fit['translation']);x=x@rot+translation
            gap=float(np.linalg.norm(x[0]-np.asarray(positions[-1]).reshape(-1,3),axis=1).max());difference=float(t[0]-progress[-1])
            if abs(difference)>1e-9 or gap>1e-5:raise ValueError('Unmatched branch join '+name+' gap '+str(gap)+' dt '+str(difference))
        else:fit=None;gap=0.;difference=0.
        sources.append({'source':name,'source_nodes':len(x),'first_progress':float(t[0]),'last_progress':float(t[-1]),'join_maximum_atom_gap_A':gap,'join_progress_difference':difference,'one_fixed_proper_transform':fit})
        positions.extend(x.reshape(-1,297));progress.extend(t)
    positions=np.array(positions);progress=np.array(progress)
    dense,t,e,report=dense_feasible_polyline(ctx,positions,progress,start_angles,winding,atom_step)
    reference=ctx['eg'](ctx['points']['A'])[0];report.update(pattern=pattern,source_blocks=sources,energy_reference_A_kcal_mol=float(reference),finite_checked_height_above_A_kcal_mol=report['maximum_finite_checked_energy_kcal_mol']-reference,elapsed_seconds=time.perf_counter()-begun)
    report['endpoints']={}
    for name,x in [('A',dense[0]),('B',dense[-1])]:
        energy,gradient=ctx['eg'](x);q=ctx['basis'](x);ev=np.linalg.eigvalsh(q.T@ctx['hessian'](x)@q)
        report['endpoints'][name]={'energy_kcal_mol':energy,'rmsd_to_intended_endpoint_A':float(ctx['rmsd'](ctx['points'][name],x)),'max_gradient_atomic_norm':float(np.linalg.norm(gradient.reshape(-1,3),axis=1).max()),'negative_internal_modes':int(np.sum(ev< -1e-4)),'lowest_internal_curvature':float(ev[0])}
    flags={'all_source_joins_match':max(r['join_maximum_atom_gap_A'] for r in sources)<1e-5,'finite_coordinates_and_energies':bool(np.isfinite(dense).all() and np.isfinite(e).all()),'monotonic_complete_progress':bool(abs(t[0])<1e-10 and abs(t[-1]-1)<1e-10 and np.min(np.diff(t))> -1e-10),'exact_node_constraints':report['maximum_node_constraint_rad']<1e-8,'probe_constraints':report['maximum_chord_probe_constraint_rad']<1e-4,'small_atom_gaps':report['maximum_atom_step_A']<.02,'stationary_intended_endpoints':all(r['rmsd_to_intended_endpoint_A']<1e-5 and r['max_gradient_atomic_norm']<1e-6 and r['negative_internal_modes']==0 and r['lowest_internal_curvature']>1e-4 for r in report['endpoints'].values())}
    report.update(checks=flags,status='COMPLETE_EXACT_FEASIBLE_COLLECTIVE_PATH' if all(flags.values()) else 'PATH_REQUIRES_REVIEW',scope='Converged constrained local-minimum branch nodes, explicit constrained saddle/downhill connectors at central switches, and retracted interpolation elsewhere. Every supplied frame and three chord probes are checked. This is an explicit candidate path with a finite checked maximum, not a global collective barrier or a claim that every interpolation point is a constrained minimum.')
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(out/'trajectory.npz',coordinates_angstrom=dense.reshape(-1,99,3),progress=t,energy_kcal_mol=e);np.savez_compressed(out/'assembly_nodes.npz',coordinates_angstrom=positions.reshape(-1,99,3),progress=progress)
    print(pattern,report['status'],'height',report['finite_checked_height_above_A_kcal_mol'],'frames',len(dense),'node_c',report['maximum_node_constraint_rad'],'probe_c',report['maximum_chord_probe_constraint_rad'],'gap',report['maximum_atom_step_A'],'endpoints',report['endpoints'],flush=True)
    return report

