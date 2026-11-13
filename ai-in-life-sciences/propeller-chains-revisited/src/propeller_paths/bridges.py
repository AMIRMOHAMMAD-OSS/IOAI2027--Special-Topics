"""Fixed-identity feasible connectors between constrained conformers.

These helpers construct and diagnose explicit curves, never an energy lower
 envelope assembled from independent optimizations. Energies are bare MMFF94.
"""
import json,time
from pathlib import Path
import numpy as np
from .constraints import torsions_jac,wrapped,kkt_state,kkt_polish


def retract(x,target,tol=2e-12,iterations=25,atom_cap=.05):
    """Local normal Newton retraction onto six angular equalities, radians."""
    x=np.asarray(x).reshape(-1).copy();begin=x.copy()
    for iteration in range(iterations+1):
        theta,j=torsions_jac(x);c=wrapped(theta-target);error=float(np.max(np.abs(c)))
        if error<=tol:return x,{'status':'FEASIBLE','iterations':iteration,'max_constraint_rad':error,'maximum_atom_correction_A':float(np.linalg.norm((x-begin).reshape(-1,3),axis=1).max())}
        if iteration==iterations:break
        direction=-j.T@np.linalg.solve(j@j.T,c);maximum=np.linalg.norm(direction.reshape(-1,3),axis=1).max();scale=min(1.,atom_cap/max(maximum,1e-30));merit=float(c@c)
        accepted=False
        for back in range(16):
            trial=x+scale*.5**back*direction;ct=wrapped(torsions_jac(trial)[0]-target)
            if ct@ct<merit:x=trial;accepted=True;break
        if not accepted:break
    return x,{'status':'RETRACTION_FAILED','iterations':iteration,'max_constraint_rad':error,'maximum_atom_correction_A':float(np.linalg.norm((x-begin).reshape(-1,3),axis=1).max())}


def proper_same_identity_fit(reference,candidate):
    """One rigid endpoint fit, with no permutations or per-frame fitting."""
    a=np.asarray(reference).reshape(-1,3);b=np.asarray(candidate).reshape(-1,3);ac=a-a.mean(0);bc=b-b.mean(0)
    u,s,vh=np.linalg.svd(bc.T@ac);fix=np.eye(3);fix[2,2]=np.linalg.det(u@vh);rotation=u@fix@vh;translation=a.mean(0)-b.mean(0)@rotation
    return (b@rotation+translation).reshape(-1),{'rotation':rotation.tolist(),'translation':translation.tolist(),'determinant':float(np.linalg.det(rotation)),'permutation':'identity'}


def projected_interpolation(ctx,start,end,target,intervals=64,align=True):
    start=np.asarray(start).reshape(-1);end=np.asarray(end).reshape(-1)
    if align:end,fit=proper_same_identity_fit(start,end)
    else:fit={'rotation':np.eye(3).tolist(),'translation':[0.,0.,0.],'determinant':1.,'permutation':'identity'}
    positions=[];records=[]
    for f in np.linspace(0.,1.,intervals+1):
        x,r=retract((1-f)*start+f*end,target)
        if r['status']!='FEASIBLE':raise ValueError('Projection failed at interpolation fraction '+str(f))
        r.update(fraction=float(f),energy_kcal_mol=float(ctx['eg'](x)[0]));positions.append(x);records.append(r)
    return np.array(positions),{'endpoint_fit':fit,'nodes':records,'scope':'Feasible fixed-t interpolation; no energy minimization or optimality claim.'}


def probe_curve(ctx,positions,target,atom_step=.015):
    """Densify with exact retraction; finite nodes and three chord probes."""
    positions=np.asarray(positions).reshape(-1,297);dense=[positions[0].copy()]
    for left,right in zip(positions[:-1],positions[1:]):
        count=max(1,int(np.ceil(np.linalg.norm((right-left).reshape(-1,3),axis=1).max()/atom_step)))
        for f in np.linspace(0.,1.,count+1)[1:]:
            x,r=retract((1-f)*left+f*right,target)
            if r['status']!='FEASIBLE':raise ValueError('Dense projection failed')
            dense.append(x)
    dense=np.array(dense);energies=[];worst_residual=0.;probe_max=-np.inf
    for i,x in enumerate(dense):
        energies.append(ctx['eg'](x)[0]);worst_residual=max(worst_residual,float(np.abs(wrapped(torsions_jac(x)[0]-target)).max()))
        if i:
            for f in (.25,.5,.75):
                p=(1-f)*dense[i-1]+f*x;probe_max=max(probe_max,ctx['eg'](p)[0]);worst_residual=max(worst_residual,float(np.abs(wrapped(torsions_jac(p)[0]-target)).max()))
    return dense,np.asarray(energies),{'frames':len(dense),'maximum_node_energy_kcal_mol':float(max(energies)),'maximum_finite_checked_energy_kcal_mol':float(max(max(energies),probe_max)),
        'maximum_chord_probe_constraint_rad':worst_residual,'maximum_atom_step_A':float(np.linalg.norm(np.diff(dense,axis=0).reshape(-1,99,3),axis=2).max()),
        'scope':'Finite nodes plus quarter, half, and three-quarter chord probes; not an analytic global curve bound.'}


def refine_bridge_candidates(ctx,positions,target,out,attempts=3):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);positions=np.asarray(positions).reshape(-1,297)
    ee=np.array([ctx['eg'](x)[0] for x in positions]);indices=np.argsort(ee[1:-1])[::-1]+1;selected=[]
    for i in indices:
        if all(abs(i-j)>=max(2,len(positions)//8) for j in selected):selected.append(int(i))
        if len(selected)==attempts:break
    records=[]
    for n,i in enumerate(selected):
        begun=time.perf_counter();x,trace,history,status=kkt_polish(ctx,positions[i],target,1e-10,1e-7,max_steps=80,atom_cap=.05);state=kkt_state(ctx,x,target)
        state.update(status=status,seed_frame=i,seed_energy_kcal_mol=float(ee[i]),xyz_A=x.reshape(-1,3).tolist(),history=history,elapsed_seconds=time.perf_counter()-begun)
        (out/f'attempt_{n:02}.json').write_text(json.dumps(state,indent=2)+'\n');np.savez_compressed(out/f'attempt_{n:02}.npz',coordinates_angstrom=np.array(trace).reshape(-1,99,3))
        records.append(state);print('Bridge KKT',n,status,'index',state['negative_tangent_modes'],'E',state['energy_kcal_mol'],'g',state['kkt_gradient_max_atomic_norm'],'sec',state['elapsed_seconds'],flush=True)
    return records


def feasible_descent(ctx,start,target,gradient_tol=1e-7,max_steps=3000,atom_cap=.012):
    """Bare-energy Armijo descent on the exact constraint manifold.

    A positive absolute-Hessian metric accelerates motion; it changes only
    the search direction. Every accepted point is retracted, energy decreasing,
    and recorded. The final KKT/Hessian checks classify the actual endpoint.
    """
    from .constraints import lagrangian_hessian
    x,rr=retract(start,target)
    if rr['status']!='FEASIBLE':raise ValueError('Initial feasible descent retraction failed')
    trace=[x.copy()];history=[];started=time.perf_counter();status='ITERATION_LIMIT';preconditioner=None
    for it in range(max_steps+1):
        e,g=ctx['eg'](x);theta,j=torsions_jac(x);lam=-np.linalg.solve(j@j.T,j@g);gl=g+j.T@lam;gmax=float(np.linalg.norm(gl.reshape(-1,3),axis=1).max())
        history.append({'iteration':it,'energy_kcal_mol':float(e),'max_kkt_gradient':gmax,'max_constraint_rad':float(np.abs(wrapped(theta-target)).max())})
        if gmax<=gradient_tol:status='PROJECTED_GRADIENT_CONVERGED';break
        if it==max_steps:break
        if preconditioner is None or it%40==0:
            hl,_,_=lagrangian_hessian(ctx,x,lam);vals,vecs=np.linalg.eigh(hl);preconditioner=(vecs/np.maximum(np.abs(vals),.03)[None,:])@vecs.T
        mj=preconditioner@j.T;mg=preconditioner@g;direction=-mg+mj@np.linalg.solve(j@mj,j@mg)
        slope=float(g@direction);move=float(np.linalg.norm(direction.reshape(-1,3),axis=1).max())
        if slope>=0 or not np.isfinite(slope):status='NON_DESCENT_DIRECTION';break
        scale=min(1.,atom_cap/max(move,1e-30));accepted=False
        for back in range(25):
            alpha=scale*.5**back;trial,rr=retract(x+alpha*direction,target);et,gt=ctx['eg'](trial)
            if rr['status']=='FEASIBLE' and et<=e+1e-4*alpha*slope:
                x=trial;trace.append(x.copy());accepted=True;break
        if not accepted:status='FEASIBLE_LINESEARCH_STOP';break
    polish_record=None
    if status=='FEASIBLE_LINESEARCH_STOP' and gmax<1e-4:
        y,tail,polish_history,polish_status=kkt_polish(ctx,x,target,1e-11,1e-8,max_steps=8,atom_cap=1e-4)
        shift=float(np.linalg.norm((y-x).reshape(-1,3),axis=1).max())
        if polish_status=='KKT_CONVERGED' and shift<1e-3 and ctx['eg'](y)[0]<=e+5e-10:
            x=y;trace.extend(tail[1:]);status='PROJECTED_DESCENT_WITH_LOCAL_KKT_POLISH'
        polish_record={'status':polish_status,'maximum_atom_shift_A':shift,'history':polish_history}
    report=kkt_state(ctx,x,target);report.update(local_endpoint_polish=polish_record,descent_status=status,descent_iterations=len(trace)-1,xyz_A=x.reshape(-1,3).tolist(),history=history,elapsed_seconds=time.perf_counter()-started)
    return x,np.asarray(trace),report


def connect_from_constrained_saddle(ctx,stationary,target,endpoints,out,progress=None):
    gradient_tol=1e-7
    out=Path(out);out.mkdir(parents=True,exist_ok=False);stationary=np.asarray(stationary).reshape(-1);state=kkt_state(ctx,stationary,target)
    if state['negative_tangent_modes']!=1 or min(abs(v) for v in state['tangent_eigenvalues'])<=1e-4 or state['max_constraint_rad']>1e-9 or state['kkt_gradient_max_atomic_norm']>1e-6:raise ValueError('Seed is not a verified constrained index-one stationary point')
    mode=np.array(state['lowest_tangent_mode']);all_records=[];traces=[]
    for sign in (-1,1):
        x0,r=retract(stationary+sign*.05*mode,target)
        if r['status']!='FEASIBLE' or ctx['eg'](x0)[0]>=state['energy_kcal_mol']:raise ValueError('Mode displacement is not feasible downhill')
        endpoint,trace,record=feasible_descent(ctx,x0,target)
        trace=np.concatenate([stationary[None,:],trace]);record['sign']=sign;record['matches']={}
        for label,ref in endpoints.items():
            fitted,fit=proper_same_identity_fit(ref,endpoint);dist=np.linalg.norm((fitted-np.asarray(ref).reshape(-1)).reshape(-1,3),axis=1)
            record['matches'][label]={'same_identity_rmsd_A':float(np.sqrt(np.mean(dist**2))),'maximum_atom_residual_A':float(dist.max()),'physical_symmetry_rmsd_A':float(ctx['rmsd'](np.asarray(ref).reshape(-1),endpoint))}
        all_records.append(record);traces.append(trace)
        print(out.name,'release',sign,record['descent_status'],'E',record['energy_kcal_mol'],'grad',record['kkt_gradient_max_atomic_norm'],'index',record['negative_tangent_modes'],'matches',record['matches'],flush=True)
        (out/f'release_{sign:+}.json').write_text(json.dumps(record,indent=2)+'\n');np.savez_compressed(out/f'release_{sign:+}.npz',coordinates_angstrom=trace.reshape(-1,99,3))
    curve=np.concatenate([traces[0][::-1],traces[1][1:]]);dense,energy,audit=probe_curve(ctx,curve,target)
    flags={'two_stationary_minima':all(r['kkt_gradient_max_atomic_norm']<=gradient_tol and r['max_constraint_rad']<1e-10 and r['negative_tangent_modes']==0 and r['lowest_tangent_curvature']>1e-4 for r in all_records),
           'two_distinct_requested_minima':len({min(r['matches'],key=lambda key:r['matches'][key]['same_identity_rmsd_A']) for r in all_records})==2 and all(min(v['same_identity_rmsd_A'] for v in r['matches'].values())<1e-5 for r in all_records),
           'release_energy_nonincreasing':all(np.max(np.diff([ctx['eg'](x)[0] for x in trace]))<=5e-10 for trace in traces),
           'finite_peak_within_saddle':audit['maximum_finite_checked_energy_kcal_mol']<=state['energy_kcal_mol']+1e-6,
           'finite_geometry':bool(np.isfinite(dense).all() and np.isfinite(energy).all()),
           'dense_gap_and_constraints':audit['maximum_atom_step_A']<=.016 and audit['maximum_chord_probe_constraint_rad']<1e-4}
    report={'status':'CONSTRAINED_CONNECTOR_VERIFIED' if all(flags.values()) else 'CONNECTOR_REQUIRES_REVIEW','checks':flags,'saddle':state,'releases':all_records,'audit':audit,'progress':progress,'scope':'A constrained index-one connector at fixed six-torsion progress. It is not an unconstrained transition state or elementary activation barrier.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(out/'connector.npz',coordinates_angstrom=dense.reshape(-1,99,3),energy_kcal_mol=energy,progress=np.full(len(dense),float(progress) if progress is not None else np.nan))
    return report


def dense_feasible_polyline(ctx,positions,progress,start_angles,winding,atom_step=.01):
    """Retraction of each finite polyline segment with fixed atom identity.

    Segment target angles vary linearly with their recorded progress; repeated
    progress values therefore describe a real fixed-progress connector.
    """
    positions=np.asarray(positions).reshape(-1,297);progress=np.asarray(progress);dense=[positions[0].copy()];times=[float(progress[0])];intervals=[];energy=[float(ctx['eg'](dense[0])[0])];probe_peak=energy[0];worst_probe=0.;worst_node=float(np.abs(wrapped(torsions_jac(dense[0])[0]-start_angles-progress[0]*winding)).max())
    for i,(left,right) in enumerate(zip(positions[:-1],positions[1:])):
        count=max(1,int(np.ceil(np.linalg.norm((right-left).reshape(-1,3),axis=1).max()/atom_step)));maximum=-np.inf
        for f in np.linspace(0.,1.,count+1)[1:]:
            t=(1-f)*progress[i]+f*progress[i+1];target=start_angles+t*winding;x,rr=retract((1-f)*left+f*right,target)
            if rr['status']!='FEASIBLE':raise ValueError('Dense path retraction failed')
            e=float(ctx['eg'](x)[0]);maximum=max(maximum,e);worst_node=max(worst_node,rr['max_constraint_rad'])
            for fraction in (.25,.5,.75):
                p=(1-fraction)*dense[-1]+fraction*x;pt=(1-fraction)*times[-1]+fraction*t;ep=float(ctx['eg'](p)[0]);probe_peak=max(probe_peak,ep);maximum=max(maximum,ep)
                worst_probe=max(worst_probe,float(np.abs(wrapped(torsions_jac(p)[0]-start_angles-pt*winding)).max()))
            dense.append(x);times.append(float(t));energy.append(e)
        intervals.append({'source_interval':i,'progress_start':float(progress[i]),'progress_end':float(progress[i+1]),'frames':count,'maximum_finite_checked_energy_kcal_mol':maximum})
    dense=np.array(dense);energy=np.array(energy);times=np.array(times);gap=float(np.linalg.norm(np.diff(dense,axis=0).reshape(-1,99,3),axis=2).max())
    report={'frames':len(dense),'maximum_node_energy_kcal_mol':float(energy.max()),'maximum_finite_checked_energy_kcal_mol':float(max(energy.max(),probe_peak)),'maximum_node_constraint_rad':worst_node,'maximum_chord_probe_constraint_rad':worst_probe,'maximum_atom_step_A':gap,'intervals':intervals,'finite_checks_only':True}
    return dense,times,energy,report
