import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from assembly_fk import folded
from interference_audit import contacts, ranges  # reuses the same model, mapping and 1-mm threshold
res = {}
ts = np.arange(0, 360, 2.0)
for a in (0, 1):
    for e in (1, -1):
        rec = [contacts(folded(np.deg2rad(t), a, e)) for t in ts]
        free = np.array([r[1] > -1e-3 for r in rec])
        res[f"F_{a}_{'+' if e > 0 else '-'}"] = {"free_fraction": float(free.mean()), "free_t_deg": ranges(ts, free),
            "worst_pen_mm": float(min(r[1] for r in rec) * 1000), "max_weld_violation": float(max(r[0] for r in rec)),
            "pairs": sorted({p for r in rec for p in r[2]})}
        print(a, e, res[f"F_{a}_{'+' if e > 0 else '-'}"])
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "interference_audit_folded.json"), "w"), indent=1)
