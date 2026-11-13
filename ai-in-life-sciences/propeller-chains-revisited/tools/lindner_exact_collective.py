#!/usr/bin/env python3
"""Run exact-constraint collective nodes using the Mac's existing checked files.

Paste this entire file into nano, save it, activate neural_opt, and run:
    python -u lindner_exact_collective.py
This is a node-convergence study. Connections between nodes are a separate test.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

COMPLETION_MARKER = "# END_LINDNER_EXACT_COLLECTIVE"
if __name__ == "__main__" and not Path(__file__).read_text().rstrip().endswith(COMPLETION_MARKER):
    raise SystemExit("Incomplete file. Paste the entire script, including its final END marker, then save again.")
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
try:
    import numpy as np
    from scipy.linalg import solve
    from scipy.optimize import minimize
except ImportError as error:
    raise SystemExit("Activate your existing neural_opt environment, then run this file again. " + str(error))

TRUSTED_CENTRAL_SOURCE_SHA256 = "f89d8f862bcba85024fcabe889d15903fa2b67a69cdcc2d88d68700663996b4b"
EMBEDDED_SOLVER_SHA256 = "ab36f7c1c75fb01e5d1059e6555b692d68fe76c3d97285bc2a0ca87f3c1ae258"
DEFAULT_COMPARISON = Path("/Volumes/LaCie/riemannian_propellers/lindner_readable.72dGzI2Z/comparison_results_1794512270493666000")

def require(condition, message):
    if not condition: raise RuntimeError(message)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path); temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

# BEGIN_EMBEDDED_CONSTRAINT_SOLVER
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
# END_EMBEDDED_CONSTRAINT_SOLVER

def load_checked_mac(comparison):
    import ast, scipy
    from rdkit import rdBase
    require(comparison.is_dir(), "The existing Mac comparison folder was not found: " + str(comparison))
    base = comparison.parent
    saved = json.loads((comparison / "comparison.json").read_text())
    expected_provenance = saved["input_provenance"]
    source=(base/'revalidate.py').read_bytes();digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    require(hashlib.sha256(source).hexdigest()==TRUSTED_CENTRAL_SOURCE_SHA256,'The original revalidate.py is missing or changed.')
    refs=base/'refs.json';data=json.loads(refs.read_text());central={};terminal=[]
    for p in base.glob('results*/validation.json'):
        try:r=json.loads(p.read_text())
        except (OSError,ValueError):continue
        if r.get('status')=='REVALIDATION_PASSED':central[digest(p)]=(p,r)
    for p in base.glob('terminal_results_*/validation.json'):
        try:r=json.loads(p.read_text())
        except (OSError,ValueError):continue
        if r.get('status')=='TERMINAL_REVALIDATION_PASSED' and digest(p)==expected_provenance['terminal_sha256']:terminal.append((p.stat().st_mtime_ns,p,r))
    require(bool(terminal),'No passed terminal validation matching this comparison was found.')
    _,terminal_path,old=max(terminal,key=lambda v:v[0]);parent=old.get('source_validation_sha256')
    require(parent in central,'The terminal results have no matching passed central source.')
    central_path,prior=central[parent];require(prior['reference_sha256']==digest(refs),'refs.json changed since validation.')
    for report,edges in ((prior,{'TS1':{'C','M'},'TS2':{'M','D'}}),(old,{'TS_AC':{'A','C'},'TS_MB':{'M','B'}})):
        for label,expected in edges.items():
            branches=report.get('connections',{}).get(label,{})
            require(len(branches)==2 and {b.get('matched_state') for b in branches.values()}==expected,'Saved connectivity differs for '+label)
    records=old['stationary_points']
    for k in ('C','TS1','M','TS2','D'):require(records[k]==prior['stationary_points'][k],'Central/terminal provenance differs for '+k)
    tree=ast.parse(source);body=next(n for n in tree.body if isinstance(n,ast.Try)).body
    imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    guard=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='require')
    setup=body[:next(i for i,n in enumerate(body) if isinstance(n,ast.FunctionDef))]
    names={'eg','maxg','basis','hessian','refine','rmsd'}
    defs=[n for n in body if isinstance(n,ast.FunctionDef) and n.name in names]
    matcher=[n for n in body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'heavy','hm','maps','hydrogens'} for t in n.targets)]
    namespace={'data':data};module=ast.Module(body=imports+[guard]+setup+defs+matcher,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(base/'revalidate.py'),'exec'),namespace)
    ctx={k:namespace[k] for k in names|{'molecule','properties','heavy','maps','hydrogens'}}
    require(0<len(ctx['maps'])<100000,'Graph symmetry enumeration failed.')
    ctx.update(base=base,data=data,records=records,points={k:np.asarray(v['xyz_A']).reshape(297) for k,v in records.items()})
    checked={}
    for k in ('A','C','M','B','D','TS1','TS2','TS_AC','TS_MB'):
        x=ctx['points'][k];e,g=ctx['eg'](x)
        require(np.isfinite(x).all() and abs(e-records[k]['energy_kcal_mol'])<.001 and ctx['maxg'](g)<1e-4,'Current force check failed for '+k)
        require(records[k]['negative_internal_modes']==int(k.startswith('TS')),'Saved index differs for '+k)
        checked[k]={'energy_kcal_mol':e,'gradient_max':ctx['maxg'](g)}
    probe=ctx['points']['C']+np.random.default_rng(20261112).normal(0,.001,297)
    d=np.random.default_rng(1).normal(size=297);d/=np.linalg.norm(d)
    error=abs(float(ctx['eg'](probe)[1]@d)-(ctx['eg'](probe+1e-5*d)[0]-ctx['eg'](probe-1e-5*d)[0])/2e-5)
    require(error<1e-4,'Analytical gradient check failed.')
    a,b=ctx['points']['A'],ctx['points']['B'];mirror=(a.reshape(99,3)*[-1,1,1]).ravel()
    require(ctx['rmsd'](a,b)>.02 and ctx['rmsd'](mirror,b)<.002,'A/B enantiomer check failed.')
    ctx['provenance']={'central_validation':str(central_path),'central_sha256':parent,'terminal_validation':str(terminal_path),
        'terminal_sha256':digest(terminal_path),'reference_sha256':digest(refs),'source_sha256':hashlib.sha256(source).hexdigest(),
        'force_checks':checked,'directional_gradient_error':error,
        'software':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'rdkit':rdBase.rdkitVersion}}
    for key in ("central_sha256", "terminal_sha256", "reference_sha256", "source_sha256"):
        require(ctx["provenance"][key] == expected_provenance[key], "Saved comparison input provenance changed: " + key)
    ctx["provenance"]["loader"] = "embedded_checked_central_loader"
    return ctx

def load_targets(ctx, comparison, pattern):
    folder = comparison / ("collective_" + pattern)
    source, summary = folder / "trajectory.npz", folder / "summary.json"
    require(source.is_file() and summary.is_file(), "Missing saved collective files for " + pattern)
    old = json.loads(summary.read_text())
    require(old.get("pattern") == pattern and old.get("status") == "SYNCHRONIZED_PATH_NUMERICALLY_CHECKED",
            "Saved collective provenance is incomplete for " + pattern)
    with np.load(source, allow_pickle=False) as data:
        endpoints = data["coordinates_angstrom"][[0, -1]].reshape(2, 297).copy()
    a_xyz, b_xyz = endpoints
    require(np.isfinite(endpoints).all(), "Nonfinite saved endpoints")
    for label, point in (("A", a_xyz), ("B", b_xyz)):
        require(ctx["rmsd"](ctx["points"][label], point) < 0.002,
                "Saved collective endpoint does not properly match " + label)
        require(abs(ctx["eg"](point)[0] - ctx["eg"](ctx["points"][label])[0]) < 0.001,
                "Saved endpoint energy differs for " + label)
    a = np.radians(np.asarray(old["angles_start_deg"]))
    delta = np.radians(np.asarray(old["winding_delta_deg"]))
    require(a.shape == delta.shape == (6,) and np.isfinite(a).all() and np.isfinite(delta).all(),
            "Invalid saved torsion targets")
    require(np.max(np.abs(wrapped(torsions_jac(a_xyz)[0] - a))) < 1e-7 and
            np.max(np.abs(wrapped(torsions_jac(b_xyz)[0] - a - delta))) < 1e-7,
            "Saved torsion winding and exact endpoints disagree")
    return a_xyz, b_xyz, a, delta, {"trajectory_sha256": digest(source), "summary_sha256": digest(summary)}

def run_branch(ctx, pattern, direction, target_data, out, intervals):
    out.mkdir(); a_xyz, b_xyz, a, delta, hashes = target_data
    values = np.linspace(0.0, 1.0, intervals + 1)
    if direction == "reverse": values = values[::-1]
    x = (a_xyz if direction == "forward" else b_xyz).copy()
    energy0 = ctx["eg"](ctx["points"]["A"])[0]; rows, coordinates = [], []
    report = {"status": "RUNNING", "pattern": pattern, "direction": direction, "nodes": rows,
              "A_energy_kcal_mol": energy0, "initial_torsions_rad": a.tolist(), "torsion_delta_rad": delta.tolist(),
              "input_hashes": hashes, "constraint_tolerance_rad": 1e-10, "kkt_gradient_tolerance": 1e-7,
              "scope": "Constrained node sequence only; continuity and connector energies remain separate tests."}
    started = time.monotonic()
    def checkpoint():
        report["elapsed_seconds"] = time.monotonic() - started
        if rows: report["maximum_node_energy_above_A_kcal_mol"] = max(r["energy_kcal_mol"] for r in rows) - energy0
        write_json(out / "branch.json", report)
        np.savez_compressed(out / "nodes.npz", coordinates_angstrom=np.asarray(coordinates).reshape(-1, 99, 3),
                            progress=np.array([r["progress"] for r in rows]),
                            energy_kcal_mol=np.array([r["energy_kcal_mol"] for r in rows]))
    checkpoint()
    try:
        for k, progress in enumerate(values):
            target = a + progress * delta; seed = x.copy()
            if k:
                theta, jac = torsions_jac(x); q = ctx["basis"](x)
                seed += q @ np.linalg.lstsq(jac @ q, -wrapped(theta - target), rcond=None)[0]
            result, trace = minimize_fixed_progress(ctx, seed, target,
                {"constraint_tol": 1e-10, "gradient_tol": 1e-7})
            result.update(pattern=pattern, progress=float(progress), direction=direction)
            save_case(out / f"node_{k:03d}", result, trace)
            candidate = np.asarray(result["xyz_A"]).reshape(297)
            row = {key: result[key] for key in ("status", "energy_kcal_mol", "max_constraint_rad",
                   "kkt_gradient_max_atomic_norm", "projected_gradient_max_atomic_norm",
                   "lowest_tangent_curvature", "negative_tangent_modes", "elapsed_seconds")}
            row.update(progress=float(progress), directory=f"node_{k:03d}",
                       maximum_fixed_identity_atom_step_A=float(ctx["maxg"](candidate - x)))
            rows.append(row); coordinates.append(candidate.reshape(99, 3)); checkpoint()
            print(f"{pattern} {direction} node {k+1}/{intervals+1}: E-A={row['energy_kcal_mol']-energy0:.6f}; "
                  f"KKT={row['kkt_gradient_max_atomic_norm']:.2g}; {row['status']}", flush=True)
            require(result["status"] == "CONSTRAINED_MINIMUM_VERIFIED",
                    f"Node {k} failed one or more numerical acceptance checks; its full diagnostics were saved.")
            x = candidate
        expected = "B" if direction == "forward" else "A"
        report["endpoint_expected_state"] = expected
        report["endpoint_proper_rmsd_A"] = float(ctx["rmsd"](ctx["points"][expected], x))
        require(report["endpoint_proper_rmsd_A"] < 0.002, "The last constrained node is in a different basin.")
        report["status"] = "COMPLETE_CONSTRAINED_NODE_SEQUENCE"
    except Exception as error:
        report.update(status="PARTIAL_OR_UNRESOLVED_NODE_SEQUENCE", error=str(error))
        print(pattern + " " + direction + ": " + str(error), flush=True)
    finally:
        checkpoint()
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--out", type=Path, help="New output folder; must not already exist")
    parser.add_argument("--intervals", type=int, default=40, help="40 gives 41 nodes per branch")
    parser.add_argument("--pattern", choices=("both", "I_III", "II_IV"), default="both")
    args = parser.parse_args(); require(args.intervals >= 4, "Use at least four intervals.")
    comparison = args.comparison.expanduser().resolve(); ctx = load_checked_mac(comparison)
    patterns = ("I_III", "II_IV") if args.pattern == "both" else (args.pattern,)
    targets = {p: load_targets(ctx, comparison, p) for p in patterns}
    probe = ctx["points"]["C"] + np.random.default_rng(20261113).normal(0, 0.001, 297)
    jacobian_check = check_torsion_jacobian(probe)
    require(jacobian_check["maximum_absolute_error_rad_per_A"] < 1e-5, "Torsion derivative check failed.")
    out = args.out.expanduser().resolve() if args.out else comparison / ("exact_collective_" + str(time.time_ns()))
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes(); (out / "exact_collective_source.py").write_bytes(source)
    report = {"status": "RUNNING", "output": str(out), "intervals": args.intervals, "branches": [],
              "source_sha256": hashlib.sha256(source).hexdigest(), "embedded_solver_sha256": EMBEDDED_SOLVER_SHA256,
              "trusted_central_source_sha256": TRUSTED_CENTRAL_SOURCE_SHA256, "prior_provenance": ctx["provenance"],
              "torsion_jacobian_check": jacobian_check,
              "scope": "Exact-constraint node convergence. No continuous path or minimized collective barrier is claimed."}
    write_json(out / "summary.json", report); started = time.monotonic()
    print("Existing Mac inputs verified. New results: " + str(out), flush=True)
    print("This checks constrained nodes; connections between nodes are a separate test.", flush=True)
    try:
        for pattern in patterns:
            for direction in ("forward", "reverse"):
                branch = run_branch(ctx, pattern, direction, targets[pattern],
                                    out / (pattern + "_" + direction), args.intervals)
                report["branches"].append(branch); write_json(out / "summary.json", report)
        complete = all(b["status"] == "COMPLETE_CONSTRAINED_NODE_SEQUENCE" for b in report["branches"])
        report["status"] = "COMPLETE_EXACT_CONSTRAINED_NODE_SEQUENCES" if complete else "NODE_SEQUENCES_REQUIRE_REVIEW"
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED_WITH_SAVED_NODE_RESULTS"
    finally:
        report["elapsed_seconds"] = time.monotonic() - started; write_json(out / "summary.json", report)
    print(report["status"] + "\nResults: " + str(out), flush=True)
    print("Node peaks are not certified continuous-path barriers. Review branch connectors separately.", flush=True)
    return 0 if report["status"] == "COMPLETE_EXACT_CONSTRAINED_NODE_SEQUENCES" else 2

if __name__ == "__main__":
    try: raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        raise SystemExit("Stopped: " + str(error) + "\nExisting results were preserved.")
# END_LINDNER_EXACT_COLLECTIVE
