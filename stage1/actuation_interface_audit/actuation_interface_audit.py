"""驱动投影审计（回应审稿意见 M2）。纯运动学计算，不运行 MuJoCo、不训练。

问题：四个位置伺服 J1,J2,J3,J7 驱动一个单环 7R 机构。正则弧段上瞬时自由度为 1，分岔界面 G_k,H_k 处为 2。
  (1) 正则弧段：一维切向量在驱动坐标上的投影是否始终非零（即驱动坐标能否参数化运动，不存在驱动奇异）？
  (2) 界面处：二维核空间投影到驱动坐标后秩是否为 2（四个驱动能否区分平面与空间两个方向）？
  (3) 冗余度：位置约束数 4 减去瞬时自由度，正则处 3、界面处 2。
另外从已审计的 Stage-I 回放（只读）比较 A3 记录中"状态之间灰区"与"确认状态"内的闭环残差，
检查焊缝顺应是否在过渡区段承担了更多的不一致。
输出：actuation_interface_audit.json
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from assembly_fk import fk, planar
from spatial_cont import G, eta, qfull, solve

A = [0, 1, 2, 6]          # J1, J2, J3, J7（解析编号，0 起）
wrap = lambda x: (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def resid(q):
    T, _, _ = fk(q)
    return (T - np.eye(4))[:3, :].ravel()


def jac(q, h=1e-7):
    return np.array([(resid(q + e) - resid(q - e)) / (2 * h) for e in np.eye(7) * h]).T


def kernel(q, dim):
    U, s, Vt = np.linalg.svd(jac(q))
    return Vt[-dim:].T, s


def act_fraction(t):
    t = t / np.linalg.norm(t)
    return float(np.linalg.norm(t[A]))


out = {"actuated_joints": "J1,J2,J3,J7", "interfaces": {}, "regular_arcs": {}}
# (2) 界面
for k in (1, 2, 3, 4):
    for side, name in ((0, f"G{k}"), (1, f"H{k}")):
        q, _ = planar(G[k], eta[k], side)
        K, s = kernel(q, 2)
        sa = np.linalg.svd(K[A], compute_uv=False)
        sp = np.linalg.svd(K[[3, 4, 5]], compute_uv=False)
        out["interfaces"][name] = {"closure_singular_values_smallest3": s[-3:].tolist(),
                                   "actuated_projection_singular_values": sa.tolist(),
                                   "actuated_projection_rank": int((sa > 1e-6).sum()),
                                   "passive_projection_singular_values": sp.tolist()}
# (1) 正则弧段：平面环
for e_ in (1, -1):
    for side in (0, 1):
        fr = []
        for t in np.linspace(0, 2 * np.pi, 721)[:-1]:
            if min(abs(wrap(t - G[k])) for k in G if eta[k] == e_) < 1e-3:
                continue
            q, _ = planar(t, e_, side)
            K, s = kernel(q, 1)
            fr.append(act_fraction(K[:, 0]))
        out["regular_arcs"][f"P_{'0' if side == 0 else 'pi'},{'+' if e_ > 0 else '-'}"] = {
            "min_actuated_fraction": min(fr), "n_samples": len(fr)}
# 空间环：从 G_k 沿 q3 延拓一整圈
for k in (1, 2, 3, 4):
    q0, _ = planar(G[k], eta[k], 0)
    x = np.array([q0[0], q0[3], q0[4]])
    fr, first = [], True
    for q3 in np.linspace(0, 2 * np.pi, 721)[1:-1]:
        x, err = solve(x + (np.array([0, 0, .05]) if first else 0), q3); first = False
        if abs(wrap(q3 - np.pi)) < 1e-3:
            continue
        K, s = kernel(qfull(x, q3), 1)
        fr.append(act_fraction(K[:, 0]))
    out["regular_arcs"][f"B_S{k}"] = {"min_actuated_fraction": min(fr), "n_samples": len(fr), "last_closure_residual": float(err)}

# Stage-I A3 回放：灰区与确认状态内的闭环残差（只读已审计数据）
try:
    import pandas as pd
    d = pd.read_csv(os.environ.get("AUDITED_REPLAYS_CSV", "audited_replays.csv.gz"),
                    usecols=["condition", "assembly", "seed", "confirmed_branch", "raw_label", "closure_orientation_rad", "closure_position_m"])
    d = d[d.condition == "A3"]
    conf = d.raw_label.eq(d.confirmed_branch)
    rows = {}
    for lab, m in (("confirmed_state", conf), ("between_states_grey", ~conf)):
        rows[lab] = {"n_samples": int(m.sum()),
                     "orientation_residual_rad_median": float(d.loc[m, "closure_orientation_rad"].median()),
                     "orientation_residual_rad_p95": float(d.loc[m, "closure_orientation_rad"].quantile(.95)),
                     "position_residual_m_median": float(d.loc[m, "closure_position_m"].median())}
    out["stageI_A3_closure_residuals"] = rows
except Exception as ex:  # 数据不可用时不影响运动学部分
    out["stageI_A3_closure_residuals"] = f"unavailable: {ex}"

json.dump(out, open(os.path.join(HERE, "actuation_interface_audit.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
