"""Bare-MMFF equality-constrained collective minima and independent diagnostics.

Angles and equality residuals are radians, coordinates angstrom, energies
kcal/mol. The multiplier convention is L(x,lambda)=E(x)+lambda.T*c(x).
Acceptance uses actual residuals, KKT stationarity, and constrained curvature.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize

TORSIONS=np.array([[2,3,24,6],[3,24,6,7],[8,9,25,12],
                   [9,25,12,13],[14,15,26,18],[15,26,18,19]])
SUPPORT=np.array(sorted({3*int(a)+d for row in TORSIONS for a in row for d in range(3)}))

def wrapped(value):return np.arctan2(np.sin(value),np.cos(value))

def torsions_jac(x):
    """Analytic chain-rule derivatives of six atan2 torsions; no degree factors."""
    p=np.asarray(x).reshape(-1,3)[TORSIONS];ident=np.eye(12).reshape(4,3,12)
    u=p[:,0]-p[:,1];b=p[:,2]-p[:,1];w0=p[:,3]-p[:,2]
    du=np.broadcast_to(ident[0]-ident[1],(6,3,12));db=np.broadcast_to(ident[2]-ident[1],(6,3,12));dw0=np.broadcast_to(ident[3]-ident[2],(6,3,12))
    length=np.linalg.norm(b,axis=1);e=b/length[:,None]
    de=(db-e[:,:,None]*np.einsum('bi,bij->bj',e,db)[:,None,:])/length[:,None,None]
    uv=np.sum(u*e,axis=1);wv=np.sum(w0*e,axis=1)
    v=u-uv[:,None]*e;w=w0-wv[:,None]*e
    dv=du-de*uv[:,None,None]-e[:,:,None]*(np.einsum('bij,bi->bj',du,e)+np.einsum('bi,bij->bj',u,de))[:,None,:]
    dw=dw0-de*wv[:,None,None]-e[:,:,None]*(np.einsum('bij,bi->bj',dw0,e)+np.einsum('bi,bij->bj',w0,de))[:,None,:]
    cross=np.cross(e,v)
    dcross=np.cross(de.transpose(0,2,1),v[:,None,:]).transpose(0,2,1)+np.cross(e[:,None,:],dv.transpose(0,2,1)).transpose(0,2,1)
    cosine=np.sum(v*w,axis=1);sine=np.sum(cross*w,axis=1)
    dc=np.einsum('bij,bi->bj',dv,w)+np.einsum('bi,bij->bj',v,dw)
    ds=np.einsum('bij,bi->bj',dcross,w)+np.einsum('bi,bij->bj',cross,dw)
    denominator=cosine*cosine+sine*sine
    if np.min(length)<1e-8 or np.min(denominator)<1e-12:raise ValueError('Singular torsion geometry')
    local=(cosine[:,None]*ds-sine[:,None]*dc)/denominator[:,None]
    jac=np.zeros((6,len(np.asarray(x).reshape(-1))))
    for k,atoms in enumerate(TORSIONS):jac[k,(3*atoms[:,None]+np.arange(3)).reshape(-1)]=local[k]
    return np.arctan2(sine,cosine),jac

def lagrangian_hessian(ctx,x,multipliers,step=1e-4):
    """Constraint curvature differentiates the analytic J at fixed multipliers."""
    x=np.asarray(x).reshape(-1);correction=np.zeros((len(x),len(x)))
    for j in SUPPORT:
        d=np.zeros_like(x);d[j]=step
        correction[:,j]=(torsions_jac(x+d)[1].T@multipliers-torsions_jac(x-d)[1].T@multipliers)/(2*step)
    correction=(correction+correction.T)/2
    if step==1e-4:
        energy_hessian=ctx['hessian'](x)
    else:
        energy_hessian=np.empty((len(x),len(x)))
        for j in range(len(x)):
            d=np.zeros_like(x);d[j]=step
            energy_hessian[:,j]=(ctx['eg'](x+d)[1]-ctx['eg'](x-d)[1])/(2*step)
        energy_hessian=(energy_hessian+energy_hessian.T)/2
    return energy_hessian+correction,energy_hessian,correction

def kkt_state(ctx,x,target,curvature=True,hessian_step=1e-4):
    x=np.asarray(x).reshape(-1);energy,g=ctx['eg'](x);theta,j=torsions_jac(x)
    c=wrapped(theta-target);q=ctx['basis'](x);a=j@q
    _,singular,vt=np.linalg.svd(a,full_matrices=True)
    rank=int(np.sum(singular>1e-9));multipliers=np.linalg.lstsq(a.T,-q.T@g,rcond=None)[0]
    lag_grad=g+j.T@multipliers;z=q@vt[rank:].T
    projected=z@(z.T@g)
    report={'energy_kcal_mol':float(energy),'constraint_residual_rad':c.tolist(),'max_constraint_rad':float(np.abs(c).max()),
        'max_constraint_deg':float(np.degrees(np.abs(c).max())),'multipliers_kcal_mol_per_rad':multipliers.tolist(),
        'kkt_gradient_max_atomic_norm':float(np.linalg.norm(lag_grad.reshape(-1,3),axis=1).max()),
        'projected_gradient_max_atomic_norm':float(np.linalg.norm(projected.reshape(-1,3),axis=1).max()),
        'constraint_rank':rank,'constraint_jacobian_singular_values':singular.tolist(),'internal_dimension':q.shape[1],'tangent_dimension':z.shape[1]}
    if curvature:
        h,he,hc=lagrangian_hessian(ctx,x,multipliers,step=hessian_step)
        tangent=(z.T@h@z);tangent=(tangent+tangent.T)/2;values,vectors=np.linalg.eigh(tangent)
        report.update(hessian_step_A=hessian_step,energy_hessian_step_A=hessian_step,constraint_hessian_step_A=hessian_step,lowest_tangent_curvature=float(values[0]),negative_tangent_modes=int(np.sum(values < -1e-4)),
             tangent_eigenvalues=values.tolist(),lowest_tangent_mode=(z@vectors[:,0]).tolist(),
             constraint_hessian_frobenius_norm=float(np.linalg.norm(hc)),energy_hessian_frobenius_norm=float(np.linalg.norm(he)))
    return report

def kkt_polish(ctx,x,target,constraint_tol=1e-9,gradient_tol=1e-5,max_steps=15,atom_cap=.05):
    """Bounded Newton steps on stationarity+six exact equalities, after SLSQP."""
    x=np.asarray(x).reshape(-1).copy();trace=[x.copy()];history=[]
    for it in range(max_steps+1):
        energy,g=ctx['eg'](x);theta,j=torsions_jac(x);c=wrapped(theta-target);q=ctx['basis'](x);a=j@q
        lam=np.linalg.lstsq(a.T,-q.T@g,rcond=None)[0];lag=g+j.T@lam
        gmax=float(np.linalg.norm(lag.reshape(-1,3),axis=1).max());cmax=float(np.abs(c).max())
        history.append({'iteration':it,'energy_kcal_mol':energy,'max_constraint_rad':cmax,'kkt_gradient_max':gmax})
        if cmax<=constraint_tol and gmax<=gradient_tol:return x,trace,history,'KKT_CONVERGED'
        if it==max_steps:break
        h,_,_=lagrangian_hessian(ctx,x,lam);internal=q.T@h@q
        matrix=np.block([[internal,a.T],[a,np.zeros((6,6))]])
        try:step=solve(matrix,-np.r_[q.T@lag,c],assume_a='sym')
        except Exception as error:return x,trace,history,'KKT_LINEAR_SOLVE_FAILED: '+str(error)
        dx=q@step[:-6];dl=step[-6:];maximum=np.linalg.norm(dx.reshape(-1,3),axis=1).max()
        scale=min(1.,atom_cap/max(maximum,1e-30));merit=float(lag@lag+1e4*(c@c));accepted=False
        for back in range(16):
            alpha=scale*.5**back;trial=x+alpha*dx;et,gt=ctx['eg'](trial);th,jt=torsions_jac(trial);ct=wrapped(th-target)
            lt=gt+jt.T@(lam+alpha*dl);mt=float(lt@lt+1e4*(ct@ct))
            if mt<merit*(1-1e-4*alpha):x=trial;trace.append(x.copy());accepted=True;break
        if not accepted:return x,trace,history,'KKT_LINESEARCH_STOP'
    return x,trace,history,'KKT_STEP_LIMIT'

def minimize_fixed_progress(ctx,seed,target_rad,options=None):
    options={} if options is None else dict(options);started=time.perf_counter();seed=np.asarray(seed).reshape(-1).copy();target=np.asarray(target_rad)
    constraint_tol=options.get('constraint_tol',1e-9);gradient_tol=options.get('gradient_tol',1e-5)
    q=ctx['basis'](seed);h=ctx['hessian'](seed);values,vectors=np.linalg.eigh(q.T@h@q)
    # An invertible fixed change of coordinates removes the large stiffness
    # spread without introducing a restraint potential or changing constraints.
    chart=q@vectors/np.sqrt(np.clip(np.abs(values),.2,1e5))[None,:]
    trace=[seed.copy()];counts={'energy_gradient':0,'constraints':0};cache={}
    def evaluate(y):
        if 'y' not in cache or not np.array_equal(y,cache['y']):
            x=seed+chart@y;e,g=ctx['eg'](x);theta,j=torsions_jac(x)
            cache.update(y=np.array(y).copy(),x=x,e=e,g=g,c=wrapped(theta-target),j=j)
        return cache
    def objective(y):
        counts['energy_gradient']+=1;r=evaluate(y);return r['e'],chart.T@r['g']
    def constraint(y):counts['constraints']+=1;return evaluate(y)['c']
    def jacobian(y):return evaluate(y)['j']@chart
    def callback(y):trace.append((seed+chart@y).copy())
    opt=minimize(objective,np.zeros(chart.shape[1]),jac=True,method='SLSQP',callback=callback,
        constraints={'type':'eq','fun':constraint,'jac':jacobian},
        options={'ftol':options.get('slsqp_ftol',1e-11),'maxiter':options.get('slsqp_maxiter',600),'disp':False})
    x=seed+chart@opt.x
    if np.linalg.norm(trace[-1]-x)>1e-14:trace.append(x.copy())
    before=kkt_state(ctx,x,target,curvature=False)
    x,polish_trace,polish_history,polish_status=kkt_polish(ctx,x,target,constraint_tol,gradient_tol,max_steps=options.get('kkt_steps',15))
    trace.extend(polish_trace[1:]);result=kkt_state(ctx,x,target,hessian_step=options.get('hessian_step',1e-4))
    flags={'angular_constraints':result['max_constraint_rad']<=constraint_tol,'kkt_stationarity':result['kkt_gradient_max_atomic_norm']<=gradient_tol,
       'projected_stationarity':result['projected_gradient_max_atomic_norm']<=gradient_tol,'six_independent_constraints':result['constraint_rank']==6,
       'tangent_dimension_285':result['tangent_dimension']==285,'positive_tangent_curvature':result['lowest_tangent_curvature']>1e-4,
       'finite':bool(np.isfinite(x).all() and np.isfinite(result['tangent_eigenvalues']).all())}
    result.update(status='CONSTRAINED_MINIMUM_VERIFIED' if all(flags.values()) else 'CONSTRAINED_CANDIDATE_REQUIRES_REVIEW',checks=flags,
        xyz_A=x.reshape(-1,3).tolist(),target_torsions_rad=target.tolist(),target_torsions_deg=np.degrees(target).tolist(),
        constraint_tolerance_rad=constraint_tol,gradient_tolerance=gradient_tol,
        slsqp={'success':bool(opt.success),'message':str(opt.message),'iterations':int(opt.nit),'evaluations':counts,'ftol':options.get('slsqp_ftol',1e-11)},
        before_polish=before,kkt_polish_status=polish_status,kkt_polish_history=polish_history,elapsed_seconds=time.perf_counter()-started)
    return result,np.asarray(trace)

def save_case(folder,result,trace):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    (folder/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(folder/'optimizer_trace.npz',coordinates_angstrom=np.asarray(trace).reshape(-1,99,3))

def check_torsion_jacobian(x):
    theta,j=torsions_jac(x);h=1e-6;numerical=np.zeros_like(j)
    for k in SUPPORT:
        dx=np.zeros(len(np.asarray(x).reshape(-1)));dx[k]=h
        numerical[:,k]=wrapped(torsions_jac(x+dx)[0]-torsions_jac(x-dx)[0])/(2*h)
    return {'step_A':h,'maximum_absolute_error_rad_per_A':float(np.abs(j-numerical).max()),'maximum_jacobian_entry':float(np.abs(j).max())}
