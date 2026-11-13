#!/usr/bin/env python3
"""Repair the supplied exact-collective script, then run it in this environment."""
import ast, hashlib, json, subprocess, sys, time
from pathlib import Path

EXPECTED_AST_SHA256 = "d5bd08e99940e021411be6b4cf4557064a6da048cb498aca5c66808aa9ae0a0b"
CORE_SHA256 = "f89d8f862bcba85024fcabe889d15903fa2b67a69cdcc2d88d68700663996b4b"
NEW_LOADER = r'''
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
'''

def canonical(value):
    if isinstance(value, ast.AST):
        return [type(value).__name__, [[name, canonical(item)]
                for name, item in ast.iter_fields(value)
                if not (name == "type_params" and item == [])]]
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value

def fingerprint(tree):
    data = json.dumps(canonical(tree), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()

class FixLoader(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == "load_checked_mac":
            return ast.parse(NEW_LOADER).body[0]
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == "TRUSTED_RUNNER_SHA256"
               for t in node.targets):
            node.value = ast.Constant(CORE_SHA256)
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "TRUSTED_RUNNER_SHA256":
            node.id = "TRUSTED_CENTRAL_SOURCE_SHA256"
        return node

    def visit_Constant(self, node):
        if node.value == "trusted_runner_sha256":
            node.value = "trusted_central_source_sha256"
        return node

    def visit_Import(self, node):
        node.names = [item for item in node.names if item.name != "importlib.util"]
        return node if node.names else None

def make_fixed(source, destination):
    text = source.read_text()
    tree = ast.parse(text, filename=str(source))
    if fingerprint(tree) != EXPECTED_AST_SHA256:
        raise RuntimeError("The exact-collective script is a different version or incomplete. Nothing was run: " + str(source))
    patched = ast.fix_missing_locations(FixLoader().visit(tree))
    output = ast.unparse(patched) + "\n# END_LINDNER_EXACT_COLLECTIVE\n"
    compile(output, str(destination), "exec")
    with destination.open("x") as handle:
        handle.write(output)
    return destination

def main():
    if not Path(__file__).read_text().rstrip().endswith("# END_LINDNER_COLLECTIVE_REPAIR"):
        raise RuntimeError("The repair file was not completely copied.")
    arguments = sys.argv[1:]
    source = Path.home() / "lindner_exact_collective2.py"
    if arguments[:1] == ["--source"]:
        if len(arguments) < 2:
            raise RuntimeError("--source needs the existing script path.")
        source = Path(arguments[1]).expanduser().resolve()
        arguments = arguments[2:]
    if not source.is_file():
        raise RuntimeError("Existing exact-collective script not found: " + str(source))
    destination = source.with_name("lindner_exact_collective_fixed_" + str(time.time_ns()) + ".py")
    make_fixed(source, destination)
    print("Fixed loader written: " + str(destination), flush=True)
    print("Starting the exact-constraint node calculation.", flush=True)
    return subprocess.call([sys.executable, "-u", str(destination), *arguments])

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, SyntaxError) as error:
        raise SystemExit("Stopped: " + str(error) + "\nExisting scripts and results were preserved.")
# END_LINDNER_COLLECTIVE_REPAIR
