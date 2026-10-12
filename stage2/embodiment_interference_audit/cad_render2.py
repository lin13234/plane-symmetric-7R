import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, PIL.Image as Im
import cad_map as C
from assembly_fk import planar
m, d = C.m, C.d
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cad_frames2")
os.makedirs(OUTDIR, exist_ok=True)
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
m.geom_rgba[floor] = [1, 1, 1, 0]; m.geom_matid[floor] = -1
m.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]; m.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
W = H = 1000
m.vis.global_.offwidth = W; m.vis.global_.offheight = H
r = mujoco.Renderer(m, H, W)
G = {1: 0.0, 2: np.pi + 2*np.arctan((5-np.sqrt(5))/10), 3: 2*np.arctan(5/4), 4: 2*np.pi - 2*np.arctan((5-np.sqrt(5))/2)}
cases = {"pos_G1": (1, G[1]), "pos_mid": (1, np.deg2rad(110)), "pos_G2": (1, G[2]),
         "neg_G3": (-1, G[3]), "neg_mid": (-1, np.deg2rad(180)), "neg_G4": (-1, G[4])}
jid = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{k}") for k in range(1, 8)}
def project(c, p):
    fwd = np.array(c.forward); up = np.array(c.up); pos = np.array(c.pos)
    right = np.cross(fwd, up); v = p - pos; z = v @ fwd
    x = v @ right * c.frustum_near / z; y = v @ up * c.frustum_near / z
    hw = (c.frustum_top - c.frustum_bottom) / 2
    return float((x / hw + 1) / 2 * W), float((1 - (y - c.frustum_bottom) / (c.frustum_top - c.frustum_bottom)) * H)
meta = {}
# fixed camera: computed once from the G1 configuration so all panels share the same view
q0, _ = planar(0.0, 1, 0); C.set_q(q0)
f = -d.xaxis[jid[1]].copy(); center0 = np.mean([d.xanchor[jid[k]] for k in range(1, 8)], axis=0)
for name, (eta, t) in cases.items():
    q, _ = planar(t, eta, 0)
    viol = C.set_q(q)
    pairs = set()
    for i in range(d.ncon):
        g1, g2 = d.contact[i].geom1, d.contact[i].geom2
        if floor in (g1, g2): continue
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g1]); b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g2])
        pairs.add((b1, b2, round(float(d.contact[i].dist), 4)))
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center0; cam.distance = 0.85
    cam.elevation = float(np.degrees(np.arcsin(np.clip(f[2], -1, 1)))); cam.azimuth = float(np.degrees(np.arctan2(f[1], f[0])))
    r.update_scene(d, camera=cam)
    img = r.render().copy()
    mask = img.sum(axis=2) == 0
    img[mask] = [222, 228, 238]
    sc = r.scene.camera[0]
    lab = {f"J{k}": project(sc, d.xanchor[jid[k]].copy()) for k in range(1, 8)}
    fn = os.path.join(OUTDIR, f"{name}.png"); Im.fromarray(img).save(fn)
    meta[name] = {"eta": eta, "q2_deg": float(np.degrees(t)), "weld_violation": float(viol), "file": fn, "labels": lab,
                  "link_contacts": sorted(pairs)}
    print(name, "viol %.1e" % viol, "contacts", sorted(pairs)[:6])
json.dump(meta, open(os.path.join(OUTDIR, "meta.json"), "w"), indent=1)
