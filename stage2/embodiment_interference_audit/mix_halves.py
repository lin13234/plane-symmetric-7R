import numpy as np
from assembly_fk import fk, planar
# plane-symmetry halves: side A = {J2,J3,J4}, side B = {J1,J7,J6}, J5 shared
A = [1, 2, 3]; B = [0, 6, 5]
def mixed(t1, t2, side=0):
    qp, _ = planar(t1, 1, side); qm, _ = planar(t2, -1, side)
    q = qp.copy(); q[B] = qm[B]; return q
def res(q):
    T, _, _ = fk(q); return np.abs(T - np.eye(4)).max()
ts = np.deg2rad(np.arange(0, 360, 1.0))
best = (9, None)
for t1 in ts:
    for t2 in ts:
        e = res(mixed(t1, t2))
        if e < best[0]: best = (e, (np.degrees(t1), np.degrees(t2)))
print("min closure error mixing (+ side A, - side B) over all q2 pairs:", best)
same = [res(mixed(t, t)) for t in ts]
print("same q2: min/max error", min(same), max(same))
