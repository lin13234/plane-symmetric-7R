import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco
import cad_map as C
from assembly_fk import planar, fk
from spatial_cont import G, eta, qfull, solve
m, d = C.m, C.d
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
def contacts(q):
    viol = C.set_q(q); worst = 0.0; pairs = set()
    for i in range(d.ncon):
        c = d.contact[i]
        if floor in (c.geom1, c.geom2): continue
        if c.dist < -1e-3:
            b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]); b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2])
            pairs.add(tuple(sorted((b1, b2))))
        worst = min(worst, c.dist)
    return viol, worst, pairs
def ranges(xs_deg, free):
    segs, start = [], None
    for i, f in enumerate(free):
        if f and start is None: start = i
        if (not f or i == len(free) - 1) and start is not None:
            end = i if f else i - 1; segs.append([float(xs_deg[start]), float(xs_deg[end])]); start = None
    if len(segs) > 1 and segs[0][0] == xs_deg[0] and segs[-1][1] == xs_deg[-1]:
        segs = [[segs[-1][0], segs[0][1] + 360.0]] + segs[1:-1]   # wrap-around merge
    return segs
out = {"xml": C.XML, "map": "qpos[joint2,joint3,joint4,joint5,joint6,joint7,joint1]=[q2,q3,-q4,q5,q6,-q7,q1]",
       "threshold_m": 1e-3, "step_deg": 2.0, "planar": {}, "spatial": {}}
ts = np.arange(0, 360, 2.0)
for side, sname in ((0, "G"), (1, "H")):
    for e in (1, -1):
        rec = [contacts(planar(np.deg2rad(t), e, side)[0]) for t in ts]
        free = np.array([r[1] > -1e-3 for r in rec])
        out["planar"][f"P_{'0' if side == 0 else 'pi'}_{'+' if e > 0 else '-'}"] = {
            "free_fraction": float(free.mean()), "free_q2_deg": ranges(ts, free),
            "worst_pen_mm": float(min(r[1] for r in rec) * 1000), "max_weld_violation": float(max(r[0] for r in rec)),
            "pairs": sorted({p for r in rec for p in r[2]})}
# spatial cycles: continuation in q3 from G_k over a full period
for k in (1, 2, 3, 4):
    q0, _ = planar(G[k], eta[k], 0)
    x = np.array([q0[0], q0[3], q0[4]]); qs = [qfull(x, 0.0)]; maxres = 0.0
    q3s = np.deg2rad(np.arange(0.5, 360.01, 0.5))
    for i, q3 in enumerate(q3s):
        x, err = solve(x + (np.array([0, 0, 0.05]) if i == 0 else 0), q3); maxres = max(maxres, err); qs.append(qfull(x, q3))
    qs = np.array(qs)
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    ret = float(np.abs(wrap(qs[-1] - qs[0])).max())
    qH, _ = planar(G[k], eta[k], 1)
    dH = float(np.abs(wrap(qs[360] - qH)).max())  # q3 = 180 deg
    sel = qs[::4]; q3deg = np.degrees(np.r_[0.0, q3s])[::4]
    rec = [contacts(q) for q in sel]
    free = np.array([r[1] > -1e-3 for r in rec])
    out["spatial"][f"S{k}"] = {"closure_res_max": maxres, "return_to_Gk": ret, "dist_to_Hk_at_q3_180": dH,
        "free_fraction": float(free.mean()), "free_q3_deg": ranges(q3deg, free),
        "worst_pen_mm": float(min(r[1] for r in rec) * 1000), "max_weld_violation": float(max(r[0] for r in rec)),
        "pairs": sorted({p for r in rec for p in r[2]})}
    print(k, out["spatial"][f"S{k}"]["free_fraction"], out["spatial"][f"S{k}"]["worst_pen_mm"], ret, dH, maxres)
for kk, v in out["planar"].items(): print(kk, v["free_fraction"], v["free_q2_deg"], v["worst_pen_mm"], v["pairs"])
for kk, v in out["spatial"].items(): print(kk, v["free_q3_deg"], v["pairs"])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "interference_audit.json"), "w"), indent=1)
