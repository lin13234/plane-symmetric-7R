from pathlib import Path
import os, sys, io, numpy as np, mujoco
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))
from assembly_fk import planar, folded
XML = str(Path(__file__).resolve().parents[2] / "model" / "zhengti_stage2_mesh_collision.xml")
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
print("joints", names, "qposadr", list(m.jnt_qposadr))
print("neq", m.neq, [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i) for i in range(m.neq)])
PAPER_TO_SLOT = [6, 0, 1, 2, 3, 4, 5]
SIGN = np.array([1, 1, -1, 1, 1, -1, 1.0])
def set_q(q):
    mujoco.mj_resetData(m, d)
    slots = np.zeros(7)
    for i in range(7):
        slots[PAPER_TO_SLOT[i]] = SIGN[PAPER_TO_SLOT[i]] * q[i]
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{k}")] for k in (2, 3, 4, 5, 6, 7, 1)]
    for s, a in enumerate(adr):
        d.qpos[a] = slots[s]
    mujoco.mj_forward(m, d)
    eq = [i for i in range(d.nefc) if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
    return np.abs(d.efc_pos[eq]).max() if eq else None
if __name__ == "__main__":
    for t in (0.0, 1.0, 2.5):
        for eta in (1, -1):
            for side in (0, 1):
                q, _ = planar(t, eta, side)
                print("P t=%.1f eta=%+d side=%d  weld violation=%.2e" % (t, eta, side, set_q(q)))
    print("F", set_q(folded(1.0, 0, 1)))
    print("zero", set_q(np.zeros(7)))
