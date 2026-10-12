import numpy as np, io, sys, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assembly_fk import planar, fk
from spatial_cont import G, eta, qfull, solve
wrap = lambda a: (np.asarray(a) + np.pi) % (2*np.pi) - np.pi
out = {}
for k in (1, 2, 3, 4):
    t0 = G[k]; h = 1e-6
    tp = wrap(planar(t0 + h, eta[k], 0)[0] - planar(t0 - h, eta[k], 0)[0]) / (2*h)
    # spatial tangent: continuation a small step from G_k on both sides of q3=0
    q0 = planar(t0, eta[k], 0)[0]; x = np.array([q0[0], q0[3], q0[4]])
    xs, _ = solve(x + np.array([0, 0, 0.05]), 1e-3)
    xs2, _ = solve(xs, 2e-3)
    ts = (qfull(xs2, 2e-3) - qfull(xs, 1e-3)) / 1e-3
    c = abs(tp @ ts) / np.linalg.norm(tp) / np.linalg.norm(ts)
    ang = np.degrees(np.arccos(min(1, c)))
    # same at H_k (H side planar vs spatial near q3=pi)
    qH = planar(t0, eta[k], 1)[0]
    tpH = wrap(planar(t0 + h, eta[k], 1)[0] - planar(t0 - h, eta[k], 1)[0]) / (2*h)
    xh = np.array([qH[0], qH[3], qH[4]])
    a1, _ = solve(xh + np.array([0, 0, 0.05]), np.pi + 1e-3); a2, _ = solve(a1, np.pi + 2e-3)
    tsH = (qfull(a2, np.pi + 2e-3) - qfull(a1, np.pi + 1e-3)) / 1e-3
    cH = abs(tpH @ tsH) / np.linalg.norm(tpH) / np.linalg.norm(tsH)
    angH = np.degrees(np.arccos(min(1, cH)))
    # check the spatial solution near H is actually near H (not another branch)
    dH = float(np.abs(wrap(qfull(a1, np.pi + 1e-3) - qH)).max())
    out[f"G{k}"] = ang; out[f"H{k}"] = angH
    print(f"k={k}: angle(planar, spatial) at G = {ang:.2f} deg, at H = {angH:.2f} deg, |q-H| after 1 mrad = {dH:.2e}")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "interface_tangent_angles.json"), "w"), indent=1)
