#!/usr/bin/env python3
"""Complete two synchronized candidate paths using existing accepted Mac nodes.

This reconstructs recorded constrained connectors and checks explicit finite paths.
The result is a comparison of selected MMFF94 candidates, not global optimality.
"""
import argparse, ast, hashlib, json, os, sys, time
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
from scipy.linalg import solve

DEFAULT_NODES = Path("/Volumes/LaCie/riemannian_propellers/lindner_readable.72dGzI2Z/comparison_results_1794512270493666000/exact_collective_1794525495802870000")
SOURCE_AST_SHA256 = "ad22b180051417094d650fa8d5adf62ecb0da333cdaa0fb3118462952b2d9141"

def require(condition, message):
    if not condition: raise RuntimeError(message)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, data):
    path = Path(path); temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def canonical(value):
    if isinstance(value, ast.AST):
        return [type(value).__name__, [[name, canonical(item)]
                for name, item in ast.iter_fields(value)
                if not (name == "type_params" and item == [])]]
    if isinstance(value, list): return [canonical(item) for item in value]
    return value

def load_existing(nodes):
    report = read_json(nodes / "summary.json")
    require(report.get("status") == "COMPLETE_EXACT_CONSTRAINED_NODE_SEQUENCES" and
            report.get("intervals") == 40, "This continuation needs the completed 41-node study.")
    source = nodes / "exact_collective_source.py"; text = source.read_text()
    require(digest(source) == report["source_sha256"], "The node-run source differs from its recorded hash.")
    tree = ast.parse(text, filename=str(source))
    stamp = hashlib.sha256(json.dumps(canonical(tree), ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    require(stamp == SOURCE_AST_SHA256, "This node solver is a different executable version; no calculation was started.")
    namespace = {"__name__": "checked_lindner_node_source", "__file__": str(source)}
    exec(compile(tree, str(source), "exec"), namespace)
    for name in ("wrapped", "torsions_jac", "lagrangian_hessian", "kkt_state", "kkt_polish"):
        globals()[name] = namespace[name]
    ctx = namespace["load_checked_mac"](nodes.parent)
    for key in ("central_sha256", "terminal_sha256", "reference_sha256", "source_sha256"):
        require(ctx["provenance"][key] == report["prior_provenance"][key], "Original input provenance changed: " + key)
    expected = {(p, d) for p in ("I_III", "II_IV") for d in ("forward", "reverse")}
    require(len(report["branches"]) == 4 and {(b["pattern"], b["direction"]) for b in report["branches"]} == expected,
            "The saved study does not contain all four expected branches.")
    hashes = {"summary.json": digest(nodes / "summary.json"), "exact_collective_source.py": digest(source),
              "../comparison.json": digest(nodes.parent / "comparison.json")}
    for branch in report["branches"]:
        name = branch["pattern"] + "_" + branch["direction"]
        path = nodes / name / "branch.json"; current = read_json(path)
        require(current == branch and current["status"] == "COMPLETE_CONSTRAINED_NODE_SEQUENCE" and
                len(current["nodes"]) == 41 and all(r["status"] == "CONSTRAINED_MINIMUM_VERIFIED" for r in current["nodes"]),
                "Saved node-branch report is incomplete or changed: " + name)
        grid = np.linspace(0., 1., 41)
        if current["direction"] == "reverse": grid = grid[::-1]
        require(np.max(np.abs(np.array([r["progress"] for r in current["nodes"]]) - grid)) < 1e-12,
                "Saved branch progress grid differs: " + name)
        hashes[name + "/branch.json"] = digest(path)
    return ctx, report, namespace, hashes

def checked_forward(ctx, nodes, pattern, target_data, hashes):
    folder = nodes / (pattern + "_forward"); branch = read_json(folder / "branch.json")
    a_xyz, b_xyz, a, delta, target_hashes = target_data
    require(branch["input_hashes"] == target_hashes, "Collective target inputs changed: " + pattern)
    require(np.max(np.abs(np.array(branch["initial_torsions_rad"]) - a)) < 1e-12 and
            np.max(np.abs(np.array(branch["torsion_delta_rad"]) - delta)) < 1e-12, "Saved branch winding changed.")
    path = folder / "nodes.npz"; hashes[str(path.relative_to(nodes))] = digest(path)
    with np.load(path, allow_pickle=False) as data:
        xyz = data["coordinates_angstrom"].copy(); progress = data["progress"].copy(); energies = data["energy_kcal_mol"].copy()
    require(xyz.shape == (41, 99, 3) and progress.shape == energies.shape == (41,) and
            np.isfinite(xyz).all() and np.isfinite(energies).all() and
            np.max(np.abs(progress - np.linspace(0., 1., 41))) < 1e-12, "Invalid saved node arrays: " + pattern)
    records = {}; checks = []
    for k, (x, t, stored_energy) in enumerate(zip(xyz.reshape(41, 297), progress, energies)):
        record_path = folder / ("node_%03d" % k) / "result.json"; record = read_json(record_path)
        hashes[str(record_path.relative_to(nodes))] = digest(record_path)
        require(record["status"] == "CONSTRAINED_MINIMUM_VERIFIED" and all(record["checks"].values()) and
                record["negative_tangent_modes"] == 0 and record["lowest_tangent_curvature"] > 1e-4,
                "Saved constrained-minimum checks failed: " + str(k))
        require(abs(record["progress"] - t) < 1e-12 and record["pattern"] == pattern and
                np.max(np.abs(np.array(record["xyz_A"]).reshape(-1) - x)) < 1e-12 and
                abs(record["energy_kcal_mol"] - stored_energy) < 1e-9, "Saved node records disagree: " + str(k))
        row = branch["nodes"][k]
        require(row["directory"] == "node_%03d" % k and abs(row["progress"] - t) < 1e-12 and
                all(row[key] == record[key] for key in ("status", "energy_kcal_mol", "max_constraint_rad",
                    "kkt_gradient_max_atomic_norm", "projected_gradient_max_atomic_norm",
                    "lowest_tangent_curvature", "negative_tangent_modes")), "Branch row and node record disagree: " + str(k))
        require(np.array(record["target_torsions_rad"]).shape == (6,) and
                np.max(np.abs(np.array(record["target_torsions_rad"]) - a - t * delta)) < 1e-12,
                "Saved node torsion target differs: " + str(k))
        energy, gradient = ctx["eg"](x); theta, jac = torsions_jac(x); residual = wrapped(theta - a - t * delta)
        require(np.linalg.matrix_rank(jac, tol=1e-9) == 6, "Constraint rank changed.")
        lam = -np.linalg.solve(jac @ jac.T, jac @ gradient)
        gmax = float(np.linalg.norm((gradient + jac.T @ lam).reshape(-1, 3), axis=1).max())
        require(abs(energy - stored_energy) < 1e-6 and np.max(np.abs(residual)) < 1e-10 and gmax < 1e-7,
                "Current node force or constraint check failed: " + pattern + " " + str(k))
        check = {"node": k, "energy_kcal_mol": float(energy), "max_constraint_rad": float(np.max(np.abs(residual))), "kkt_gradient": gmax}
        if k in (18, 20, 22):
            state = kkt_state(ctx, x, a + t * delta)
            require(state["constraint_rank"] == 6 and state["tangent_dimension"] == 285 and
                    state["negative_tangent_modes"] == 0 and state["lowest_tangent_curvature"] > 1e-4,
                    "A branch starting node is not a constrained minimum.")
            check["fresh_curvature"] = state["lowest_tangent_curvature"]
        records[k] = record; checks.append(check)
    print(pattern, "saved 41-node sequence checked; starting complete-path construction.", flush=True)
    return xyz.reshape(41, 297), progress, records, checks



# Quantized reference connector seeds: integer coordinates / 100000 = angstrom.
# They are starting guesses; stationarity, index, endpoints, and paths are rechecked.
CONNECTOR_SEEDS = {
    'I_III': {
        'left': (0.4675, [
            -929561, -27413, -75255, -863413, -148732, -98142, -732562, -167043, -48560, -670973, -63300, 27633, -739573, 60142, 48083,
            -870345, 76435, -2303, -400148, -47698, 28475, -381663, 7019, -101144, -249799, 40713, -146764, -134491, 24187, -62950,
            -154652, -21598, 69640, -284313, -65814, 111134, 136178, 35996, -57257, 202219, -89873, -62423, 334306, -103185, -12118,
            398319, 7416, 50334, 332945, 133656, 54272, 200763, 146944, 4256, 668681, 14146, 48603, 687254, 50018, -88349,
            818690, 71665, -137693, 929003, 57893, -52663, 914344, 22772, 81499, 784942, -713, 132708, -534871, -85733, 92625,
            1919, 54248, -127505, 533627, -10487, 118214, -935174, -255385, -176649, -661236, -298404, -71993, -675262, 171588, 127649,
            -949164, 203088, 18444, -498885, 28783, -194440, -235361, 94899, -288463, -42117, -25377, 170685, -298046, -129062, 249130,
            133289, -210125, -123766, 405232, -236773, -22501, 402796, 254095, 114179, 130088, 280712, 14302, 571840, 66564, -184254,
            846652, 110120, -280959, 1038757, 11020, 165577, 769045, -38545, 278854, -1030200, -13746, -114609, 1028974, 75260, -92239,
            -539252, -37525, 191254, -529709, -192786, 115810, 8038, -5022, -219857, -1272, 158015, -162936, 531062, 51589, 208765,
            535731, -112709, 158070, -934410, -350332, -122147, -1040191, -229240, -193610, -887965, -268466, -274527, -676699, -364866, 13563,
            -553938, -283711, -87539, -696454, -349021, -162318, -691704, 155179, 234590, -567846, 178140, 108121, -715183, 269832, 100902,
            -904947, 284977, -39165, -1053100, 191745, -14242, -951908, 229726, 124599, -566281, 104356, -153781, -549922, -65655, -213723,
            -472266, 65557, -293628, -269062, 19615, -360552, -134062, 121605, -318399, -293907, 186884, -299169, 21415, -112832, 155064,
            -77323, -28058, 274171, 15130, 67195, 168138, -307721, -50962, 325183, -383039, -196930, 257968, -211877, -192762, 271845,
            162280, -220172, -228799, 24394, -203795, -117622, 158495, -302124, -70046, 378465, -289433, -114644, 379899, -299284, 63655,
            513877, -225772, -27939, 511573, 247771, 105622, 375929, 264156, 219769, 376483, 346086, 60998, 154728, 342628, -72478,
            21468, 269707, 20823, 157631, 333968, 105872, 602606, 91382, -286094, 507511, 149003, -153049, 516424, -26892, -193909,
            811906, 31673, -348950, 953847, 123338, -299357, 798111, 205227, -305101, 1128472, 39356, 109445, 1032855, 77758, 252159,
            1052479, -92337, 198909, 722804, 43880, 334063, 863784, -61266, 328277, 708412, -129019, 289233,
        ]),
        'right': (0.525, [
            -928098, -48406, -83877, -847529, -162053, -98074, -717835, -162685, -42495, -671452, -48890, 29988, -755349, 65888, 43074,
            -884698, 64558, -13402, -402462, -19679, 41113, -382780, 29237, -90465, -250345, 39539, -144155, -134152, 15644, -63389,
            -153614, -28054, 70245, -286489, -44036, 122027, 137106, 28568, -62489, 209451, -93782, -69335, 341365, -101420, -17419,
            398547, 10688, 48862, 326117, 132911, 55907, 194140, 140489, 4251, 668729, 20034, 49017, 687460, 61078, -86437,
            819116, 71529, -138706, 929585, 43950, -57321, 914805, 5273, 75854, 785056, -5826, 130280, -538210, -55899, 104009,
            1555, 42215, -131106, 533396, -2742, 118711, -902906, -279788, -173953, -630642, -285480, -56823, -709523, 186756, 121790,
            -978750, 181593, -1298, -499095, 71684, -177459, -235377, 81946, -289812, -36193, -59856, 160349, -301655, -89739, 266621,
            147815, -215706, -135093, 419465, -230668, -30502, 388272, 254940, 120901, 115593, 269374, 18168, 572001, 94060, -177945,
            847142, 112323, -281307, 1039422, -22856, 155637, 769021, -49215, 274850, -1027742, -48332, -127859, 1029773, 52769, -99114,
            -549136, 6019, 193861, -530093, -158368, 142802, 6553, -22873, -219244, -2131, 143432, -173744, 530507, 62837, 206607,
            535207, -103395, 162645, -893436, -371498, -114915, -1009425, -267030, -196103, -850308, -291467, -269225, -642038, -349655, 31082,
            -525257, -258802, -68774, -655607, -343695, -145981, -732319, 172769, 227908, -602214, 204487, 109956, -757554, 278826, 87450,
            -941528, 266195, -59924, -1078837, 156955, -38381, -989991, 211220, 103494, -567377, 135489, -121588, -549958, -15356, -219499,
            -470012, 135961, -260918, -309252, 31024, -352681, -139911, 54803, -334976, -247276, 190428, -298305, 26872, -136181, 114777,
            -63137, -103358, 256709, 19643, 30959, 183783, -244362, -23956, 332925, -403236, -86270, 305953, -267281, -193231, 276615,
            175500, -219066, -240896, 38809, -216810, -126622, 179911, -308243, -86238, 398966, -280118, -125993, 393885, -298199, 51706,
            527534, -214493, -30710, 497166, 256119, 110903, 361942, 258270, 227047, 355447, 347442, 72470, 139387, 336752, -64696,
            7644, 252376, 20338, 137318, 319363, 113100, 602993, 133867, -274855, 510253, 172930, -134787, 513808, 4450, -200218,
            797465, 44112, -351058, 954206, 108815, -304302, 813677, 215076, -298756, 1129687, -14755, 94084, 1049241, 49200, 237451,
            1037045, -124738, 195633, 704200, 20095, 329292, 863118, -50520, 330338, 727474, -150378, 279232,
        ]),
    },
    'II_IV': {
        'left': (0.46, [
            -931289, -9524, -16760, -906454, -146521, -24826, -773171, -192688, -19897, -663431, -99638, -9822, -692488, 39733, 860,
            -827504, 83534, -4256, -393644, -82540, -9036, -336167, -38580, -131498, -208339, 23213, -132048, -140902, 48611, -9355,
            -198216, 4214, 113042, -326178, -57186, 113601, 129970, 65427, -12225, 159718, -72597, -26464, 296135, -117211, -24885,
            405233, -25670, -7208, 374534, 110759, 15685, 239468, 156674, 3950, 674983, -2392, -12280, 728193, 55695, -131229,
            854005, 119482, -127552, 923676, 127699, -6353, 873199, 71984, 111800, 747553, 7726, 110251, -523155, -162978, -10258,
            -10672, 127995, -7699, 546772, -84812, -19346, -1024438, -239173, -38278, -746186, -341732, -29779, -584618, 144109, 17033,
            -866251, 229197, 3828, -409251, -60117, -262538, -146176, 65151, -263833, -124457, 24647, 243873, -389218, -97869, 245321,
            49776, -174941, -45314, 322437, -266170, -43399, 481281, 211286, 52920, 213263, 306496, 13830, 653095, 45464, -262155,
            916528, 180752, -250121, 955718, 83939, 237242, 693682, -54335, 237317, -1034203, 25952, -20251, 1020324, 177878, -4159,
            -518604, -232648, 74341, -516889, -229241, -97649, -16954, 198477, -91502, -15298, 192849, 80908, 555362, -162974, 57104,
            551160, -139513, -114531, -1032376, -303632, 49836, -1118645, -183907, -46827, -1014958, -300317, -128594, -680892, -374967, 51468,
            -700354, -365151, -126377, -836425, -402707, -21172, -524328, 123462, 105545, -523530, 150538, -73187, -623397, 244914, 33497,
            -818018, 286098, -76306, -974279, 243236, -7864, -838635, 270551, 101346, -393565, 23792, -331081, -517649, -65236, -249879,
            -374759, -152637, -309697, -36893, 64525, -260282, -180499, 165418, -291052, -171239, -4811, -344186, -141835, -58524, 312898,
            -156673, 118144, 290702, -15989, 27020, 230739, -498475, -94888, 241407, -356962, -198788, 272635, -362955, -28426, 325737,
            84623, -276787, -63010, -8959, -151415, -134109, -12112, -181305, 44349, 270187, -323102, 34275, 426754, -296630, -35479,
            288924, -297644, -142819, 546778, 171077, 130043, 441113, 300921, 100955, 537227, 243248, -35271, 121947, 338426, -36673,
            207327, 336498, 118911, 291875, 363503, -36856, 678553, 126721, -330792, 545040, 53429, -247407, 676043, -49834, -310813,
            919482, 108266, -332114, 1019866, 211918, -231418, 860173, 269349, -281014, 1055697, 123440, 216179, 907329, 152042, 307955,
            969401, -14212, 283769, 584835, -45101, 243631, 731752, -4537, 326941, 721383, -160118, 241651,
        ]),
        'right': (0.525, [
            -931120, -8062, -23343, -906920, -145064, -33025, -773621, -191385, -34829, -663384, -98876, -24592, -691693, 40732, -14816,
            -826747, 84761, -14431, -393675, -84558, -10324, -324206, -38392, -125514, -196174, 21459, -112178, -139975, 42052, 16854,
            -210377, -2640, 132244, -338460, -62399, 118849, 129997, 63539, 23547, 160695, -73816, 6787, 295275, -114356, -20732,
            404588, -21575, -15069, 374891, 115109, 9061, 238876, 156894, 27877, 674885, -2192, -29447, 731959, 53693, -147697,
            858233, 116364, -141188, 924045, 126386, -17957, 869083, 73875, 99665, 742935, 10840, 95290, -523474, -163049, -25388,
            -11608, 123327, 31520, 544794, -81435, -36592, -1025577, -237431, -41619, -747187, -340508, -45153, -583029, 145023, -4449,
            -864944, 230451, -4364, -384778, -55607, -263413, -120795, 64894, -236303, -150812, 16062, 270437, -414739, -104635, 242843,
            54224, -180850, 16899, 320909, -261512, -51178, 484445, 219499, 13691, 212664, 305034, 52207, 660952, 41771, -280768,
            925285, 174675, -262807, 947279, 88160, 227593, 683383, -46921, 221798, -1034024, 27609, -22551, 1021068, 175696, -13619,
            -522016, -240446, 52481, -514628, -220887, -118240, -17806, 203082, -43842, -18670, 176794, 127038, 555474, -165860, 32894,
            544210, -128504, -135632, -1023724, -309757, 40542, -1120204, -182746, -34027, -1026030, -289935, -137670, -702743, -377331, 47844,
            -680859, -362223, -129421, -837393, -399550, -62725, -525142, 131157, 87001, -519771, 143357, -93309, -620800, 247372, 1201,
            -824755, 286369, -89465, -973637, 244187, -5473, -828253, 273093, 89549, -358688, 27733, -329400, -494083, -55412, -261082,
            -350170, -149363, -307957, -12553, 65761, -221424, -153239, 164917, -266568, -136537, -5069, -319016, -177118, -66777, 336973,
            -186000, 110124, 313866, -41464, 16129, 268891, -522951, -104015, 227446, -383690, -205008, 273398, -398497, -34776, 325534,
            94716, -280563, 36056, -4394, -186272, -75106, -8792, -164243, 104108, 325452, -318516, 42153, 412677, -280241, -107062,
            242436, -302022, -116007, 551029, 201010, 98149, 449535, 321964, 27165, 538156, 222175, -81178, 109138, 331233, 73882,
            269163, 338952, 139743, 241217, 362791, -36366, 689839, 121100, -350290, 552581, 51761, -269536, 683903, -54906, -326624,
            930141, 100602, -343277, 1028281, 205202, -241366, 870855, 263198, -297169, 1048500, 125704, 209036, 897398, 158807, 294679,
            957908, -8851, 277209, 574629, -35452, 223951, 719275, 4336, 311496, 708767, -153075, 229724,
        ]),
    },
}


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

def feasible_descent(ctx,start,target,gradient_tol=1e-7,max_steps=3000,atom_cap=.012):
    """Bare-energy Armijo descent on the exact constraint manifold.

    A positive absolute-Hessian metric accelerates motion; it changes only
    the search direction. Every accepted point is retracted, energy decreasing,
    and recorded. The final KKT/Hessian checks classify the actual endpoint.
    """
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

def same_identity_rmsd(reference,candidate):
    fitted,_=proper_same_identity_fit(reference,candidate)
    return float(np.sqrt(np.mean(np.sum((fitted-np.asarray(reference).reshape(-1))
                                       .reshape(-1,3)**2,axis=1))))

def construct_path(ctx, nodes, pattern, target_data, checked, out, atom_step):
    started = time.monotonic(); out.mkdir(parents=True, exist_ok=False)
    xyz, progress, records, input_checks = checked
    a_xyz, b_xyz, a, delta, hashes = target_data
    left = CONNECTOR_SEEDS[pattern]["left"][0]; right = CONNECTOR_SEEDS[pattern]["right"][0]
    starts = [("outer_left", 18, left), ("middle_left", 20, left),
              ("middle_right", 20, right), ("outer_right", 22, right)]
    for name, k, end in starts:
        rec = records[k]
        result = follow_branch(ctx, rec["xyz_A"], rec["progress"], end, a, delta, out / name)
        require(result["status"] == "SMOOTH_KKT_BRANCH_REACHED_REQUESTED_END", "Local branch did not reach connector: " + name)
    branch = {name: read_block(out / name) for name, k, end in starts}; connectors = {}
    for side, t, refs in [("left", left, {"start_minimum": branch["outer_left"][0][-1], "end_minimum": branch["middle_left"][0][-1]}),
                          ("right", right, {"start_minimum": branch["middle_right"][0][-1], "end_minimum": branch["outer_right"][0][-1]})]:
        x = np.array(CONNECTOR_SEEDS[pattern][side][1], dtype=float) / 100000.
        require(x.shape == (297,) and np.max(np.abs(wrapped(torsions_jac(x)[0] - a - t * delta))) < .001,
                "The connector seed and this pattern's torsion targets disagree.")
        x, trace, history, status = kkt_polish(ctx, x, a + t * delta, 1e-11, 1e-8, max_steps=20, atom_cap=.01)
        state = kkt_state(ctx, x, a + t * delta)
        state.update(xyz_A=x.reshape(-1, 3).tolist(), progress=t, refinement_status=status,
                     origin="revalidation_of_recorded_constrained_connector_seed", history=history)
        write_json(out / (side + "_seed_refinement.json"), state)
        require(status == "KKT_CONVERGED", "The connector seed did not refine: " + side)
        result = connect_from_constrained_saddle(ctx, x, a + t * delta, refs, out / (side + "_connector"), progress=t)
        require(result["status"] == "CONSTRAINED_CONNECTOR_VERIFIED", "Connector failed numerical checks: " + side)
        cx, ct = read_block(out / (side + "_connector"), "connector.npz")
        if same_identity_rmsd(refs["start_minimum"], cx[-1]) < same_identity_rmsd(refs["start_minimum"], cx[0]):
            cx, ct = cx[::-1], ct[::-1]
        connectors[side] = (cx, ct)
    blocks = [("saved_forward_nodes_0_to_045", xyz[:19], progress[:19]),
              ("outer_left", *branch["outer_left"]), ("left_connector", *connectors["left"]),
              ("middle_left_reversed", branch["middle_left"][0][::-1], branch["middle_left"][1][::-1]),
              ("middle_right", *branch["middle_right"]), ("right_connector", *connectors["right"]),
              ("outer_right_reversed", branch["outer_right"][0][::-1], branch["outer_right"][1][::-1]),
              ("saved_forward_nodes_055_to_1", xyz[22:], progress[22:])]
    result = assemble_blocks(ctx, blocks, a, delta, out / "complete_path", pattern, atom_step=atom_step)
    require(result["status"] == "COMPLETE_EXACT_FEASIBLE_COLLECTIVE_PATH", "Complete path failed a numerical acceptance check.")
    result["complete_workflow_elapsed_seconds"] = time.monotonic() - started
    write_json(out / "workflow_summary.json", result)
    return result

def saved_sequential_height(ctx, comparison):
    data = read_json(comparison / "comparison.json")["sequential"]
    require(data["status"] == "LOWER_SEQUENTIAL_ROUTE_REVALIDATED", "The saved sequential route did not pass.")
    require(data["states"] == ["A", "C", "P", "Q", "D", "M", "B"], "Saved sequential states differ.")
    require(data["saddles"] == ["TS_AC", "TS_CP", "TS_PQ", "TS_QD", "TS_DM", "TS_MB"], "Saved sequential saddle list differs.")
    energies = []
    for label in data["saddles"]:
        record = data["edges"][label]["saddle"]; energy, gradient = ctx["eg"](np.array(record["xyz_A"]).reshape(-1))
        require(abs(energy - record["energy_kcal_mol"]) < 1e-6 and ctx["maxg"](gradient) < 1e-5 and
                record["negative_internal_modes"] == 1, "Saved sequential saddle force check failed: " + label)
        energies.append(energy)
    reference = ctx["eg"](ctx["points"]["A"])[0]; height = max(energies) - reference
    require(abs(height - data["highest_saddle_above_A_kcal_mol"]) < 1e-6 and
            data["path_check"]["status"] == "VERIFIED_DISCRETE_PATH", "Saved sequential height or path status differs.")
    path_height = data["maximum_sampled_path_energy_above_A_kcal_mol"]
    require(abs(data["A_energy_kcal_mol"] - reference) < 1e-6 and np.isfinite(path_height) and
            abs(path_height - (data["path_check"]["maximum_sampled_energy_kcal_mol"] - reference)) < 1e-6 and
            path_height >= height - 1e-6, "Saved sequential finite-path maximum is inconsistent.")
    return {"current_checked_saddle_height_above_A_kcal_mol": float(height),
            "saved_finite_checked_path_height_above_A_kcal_mol": float(path_height),
            "scope": "Original accepted sequential path record reused; its six saddle energies and forces rechecked in this run."}

def compact_summary(report):
    out = Path(report["output"]); paths = report["paths"]
    heights = {p: r["finite_checked_height_above_A_kcal_mol"] for p, r in paths.items()}
    baseline = report["sequential_baseline"]["saved_finite_checked_path_height_above_A_kcal_mol"]
    connectors = {}; checks = {}
    for pattern, path in paths.items():
        connectors[pattern] = {}
        for side in ("left", "right"):
            connector = read_json(out / pattern / (side + "_connector") / "summary.json")
            require(connector["status"] == "CONSTRAINED_CONNECTOR_VERIFIED" and all(connector["checks"].values()),
                    "A saved connector report did not pass: " + pattern + " " + side)
            connectors[pattern][side] = connector["status"]
        checks[pattern] = {"all_path_checks_passed": all(path["checks"].values()), "frames": path["frames"],
            "maximum_endpoint_proper_rmsd_A": max(v["rmsd_to_intended_endpoint_A"] for v in path["endpoints"].values()),
            "maximum_node_constraint_rad": path["maximum_node_constraint_rad"],
            "maximum_chord_probe_constraint_rad": path["maximum_chord_probe_constraint_rad"],
            "maximum_atom_step_A": path["maximum_atom_step_A"]}
    return {"status": report["status"], "sequential_baseline": report["sequential_baseline"],
        "synchronized_finite_checked_heights_above_A_kcal_mol": heights,
        "candidate_height_differences_from_saved_sequential_path_kcal_mol": {p: v - baseline for p, v in heights.items()},
        "verified_constrained_connector_count": sum(len(v) for v in connectors.values()),
        "constrained_connectors": connectors, "path_checks": checks, "output": str(out),
        "elapsed_seconds": report["elapsed_seconds"], "scope": report["scope"]}

def main():
    require(Path(__file__).read_text().rstrip().endswith("# END_LINDNER_COMPLETE_PATHS"), "The file was not completely copied.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--out", type=Path, help="New output folder")
    parser.add_argument("--pattern", choices=("both", "I_III", "II_IV"), default="both")
    parser.add_argument("--atom-step", type=float, default=.01)
    args = parser.parse_args(); require(0 < args.atom_step <= .01, "Use an atom step above zero and at most 0.01 angstrom.")
    nodes = args.nodes.expanduser().resolve(); ctx, previous, source, hashes = load_existing(nodes)
    patterns = ("I_III", "II_IV") if args.pattern == "both" else (args.pattern,)
    targets = {p: source["load_targets"](ctx, nodes.parent, p) for p in patterns}
    checked = {p: checked_forward(ctx, nodes, p, targets[p], hashes) for p in patterns}
    sequential = saved_sequential_height(ctx, nodes.parent)
    out = args.out.expanduser().resolve() if args.out else nodes / ("complete_paths_" + str(time.time_ns()))
    out.mkdir(parents=True, exist_ok=False); (out / "complete_paths_source.py").write_bytes(Path(__file__).read_bytes())
    report = {"status": "RUNNING", "output": str(out), "node_source": str(nodes), "input_sha256": hashes,
              "source_sha256": digest(Path(__file__)), "prior_provenance": ctx["provenance"], "paths": {},
              "sequential_baseline": sequential,
              "scope": "Revalidation of selected seeded MMFF94 candidates with finite path checks. No global optimum or experimental rate claim."}
    write_json(out / "input_node_checks.json", {p: checked[p][3] for p in patterns})
    write_json(out / "summary.json", report); begun = time.monotonic()
    print("Existing accepted nodes verified. New complete-path results: " + str(out), flush=True)
    try:
        for pattern in patterns:
            report["paths"][pattern] = construct_path(ctx, nodes, pattern, targets[pattern], checked[pattern], out / pattern, args.atom_step)
            write_json(out / "summary.json", report)
        report["status"] = "COMPLETE_COLLECTIVE_CANDIDATE_PATHS_VERIFIED"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="STOPPED_WITH_SAVED_DIAGNOSTICS", error=str(error) or type(error).__name__)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - begun; write_json(out / "summary.json", report)
    summary = compact_summary(report); write_json(out / "final_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("Finite numerical path checks passed. The selected candidates are not a global-optimality proof.", flush=True)
    return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        raise SystemExit("Stopped: " + str(error) + "\nExisting node results were preserved.")
    except (KeyError, TypeError, IndexError) as error:
        raise SystemExit("Stopped: A saved input has an unexpected schema: " + str(error) + "\nExisting node results were preserved.")
# END_LINDNER_COMPLETE_PATHS
