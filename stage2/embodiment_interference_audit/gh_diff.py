import numpy as np
from assembly_fk import fk, planar, MDH, Rx, Rz, Tx, Tz
t = 1.0
for eta in (1, -1):
    qa, _ = planar(t, eta, 0); qb, _ = planar(t, eta, 1)
    _, pa, aa = fk(qa); _, pb, ab = fk(qb)
    print("eta", eta, "max point diff", np.abs(pa - pb).max())
    print(" axis dots", np.round(np.sum(aa * ab, 1), 3))
    # body frames
    def frames(q):
        T = np.eye(4); out = []
        for (al, a, d, th0), qi in zip(MDH, q):
            T = T @ Rx(al) @ Tx(a) @ Rz(qi + th0) @ Tz(d); out.append(T.copy())
        return out
    fa, fb = frames(qa), frames(qb)
    for i in range(7):
        print("  body", i + 1, "R diff", np.round(np.abs(fa[i][:3, :3] - fb[i][:3, :3]).max(), 3))
