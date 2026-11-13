"""Local KKT predictor/corrector continuation without branch switching."""
import json,time
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from .constraints import torsions_jac,wrapped,lagrangian_hessian,kkt_polish,kkt_state

def follow_branch(ctx,x,t0,tend,a,delta,out,step=.0025,min_step=1e-5,max_steps=160):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);x=np.asarray(x).reshape(-1).copy();t=float(t0);direction=np.sign(tend-t0)
    nodes=[];coords=[];attempts=[];begun=time.perf_counter();status='RUNNING'
    state=kkt_state(ctx,x,a+t*delta)
    def append(record):
        record.update(progress=float(t),profile_derivative=-float(np.dot(record['multipliers_kcal_mol_per_rad'],delta)))
        nodes.append(record);coords.append(x.copy())
        (out/'branch.json').write_text(json.dumps({'status':status,'start':t0,'requested_end':tend,'nodes':nodes,'attempts':attempts,'elapsed_seconds':time.perf_counter()-begun},indent=2)+'\n')
        np.savez_compressed(out/'nodes.npz',coordinates_angstrom=np.array(coords).reshape(-1,99,3),progress=np.array([r['progress'] for r in nodes]),energy_kcal_mol=np.array([r['energy_kcal_mol'] for r in nodes]))
    append(state)
    for iteration in range(max_steps):
        if abs(t-tend)<1e-12:status='SMOOTH_KKT_BRANCH_REACHED_REQUESTED_END';break
        lam=np.array(state['multipliers_kcal_mol_per_rad']);q=ctx['basis'](x);j=torsions_jac(x)[1];cjac=j@q
        hl,_,_=lagrangian_hessian(ctx,x,lam);h=q.T@hl@q
        matrix=np.block([[h,cjac.T],[cjac,np.zeros((6,6))]])
        rate=solve(matrix,np.r_[np.zeros(q.shape[1]),delta],assume_a='sym');velocity=q@rate[:-6]
        max_rate=np.linalg.norm(velocity.reshape(-1,3),axis=1).max()
        trialstep=min(step,abs(tend-t),.04/max(max_rate,1e-30));accepted=False
        while trialstep>=min(min_step,abs(tend-t))-1e-15:
            next_t=t+direction*trialstep;predict=x+direction*trialstep*velocity
            y,trace,history,polish=kkt_polish(ctx,predict,a+next_t*delta,1e-10,1e-7,max_steps=12,atom_cap=.03)
            candidate=kkt_state(ctx,y,a+next_t*delta)
            correction=float(np.linalg.norm((y-predict).reshape(-1,3),axis=1).max());movement=float(np.linalg.norm((y-x).reshape(-1,3),axis=1).max())
            accepted=(candidate['max_constraint_rad']<1e-10 and candidate['kkt_gradient_max_atomic_norm']<1e-7 and candidate['lowest_tangent_curvature']>1e-4 and correction<.02 and movement<.061)
            attempts.append({'from':t,'to':next_t,'step':trialstep,'accepted':bool(accepted),'status':polish,'energy_kcal_mol':candidate['energy_kcal_mol'],'constraint_rad':candidate['max_constraint_rad'],'kkt_gradient':candidate['kkt_gradient_max_atomic_norm'],'lowest_tangent_curvature':candidate['lowest_tangent_curvature'],'predictor_correction_A':correction,'atom_step_A':movement})
            if accepted:
                x=y;t=next_t;state=candidate;state['maximum_atom_step_A']=movement;state['predictor_correction_A']=correction;append(state);break
            trialstep*=.5
        if not accepted:status='CONTINUATION_OBSTRUCTION';break
        if len(nodes)%5==0:print(out.name,'node',len(nodes),'t',t,'E',state['energy_kcal_mol'],'curv',state['lowest_tangent_curvature'],'step',trialstep,'sec',time.perf_counter()-begun,flush=True)
    else:status='ACCEPTED_NODE_BUDGET_REACHED'
    report={'status':status,'start':t0,'requested_end':tend,'nodes':nodes,'attempts':attempts,'elapsed_seconds':time.perf_counter()-begun,
        'scope':'Local predictor/corrector continuation with no minimizer restart or branch splicing; an obstruction requires interpretation, not automatic classification as a fold.'}
    (out/'branch.json').write_text(json.dumps(report,indent=2)+'\n');print(out.name,status,'last_t',t,'curv',state['lowest_tangent_curvature'],flush=True)
    return report

