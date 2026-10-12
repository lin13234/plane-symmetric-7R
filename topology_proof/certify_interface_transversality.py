"""精确核验八个 G/H 接口处的局部结构（横截性）。

对每个接口：
1. 按 certify_gateway_rank.py 的同一方式构造精确 6x7 旋量矩阵 J；
2. 断言非对角块恒为零（行 {2,4,6} x 列 {3,5,7}、行 {1,3,5} x 列 {1,2,4,6}，1-based）；
3. 平面切向 v_P：q3=q5=q7 分量为 0、q2 分量为 1，其余由活动块零空间精确求得；
4. 空间切向 v_S：被动块（列 3,5,7）零空间的精确基向量，与候选式 e3 - 2 cos z e5 - e7 比较；
5. 断言 J v_P = J v_S = 0，二者支撑不相交（因而线性无关），且 rank J = 5（核为二维）。
全部为 sympy 精确有理/根式运算，不含浮点。
"""
from pathlib import Path
import json
import sympy as s
from exact_model import PARAMS, matrix

OUT = Path(__file__).resolve().parent
zs = [s.Integer(0), (5 - s.sqrt(5)) / 10, s.Rational(5, 4), (5 + s.sqrt(5)) / 10]
records = []
ok_all = True
for a in (0, 1):
    for kk, Z in enumerate(zs, 1):
        cz = s.simplify((1 - Z * Z) / (1 + Z * Z)); sz = s.simplify(2 * Z / (1 + Z * Z))
        sign = 1 if kk in (1, 3) else -1
        cy = sign * cz; sy = sign * sz
        if a:
            cz, sz = -cz, sz
        sig = (-1) ** a
        trig = [(cy, sy), (cy, sy), (sig, 0), (cz, sz), (1, 0), (cz, -sz), (sig, 0)]
        T = s.eye(4); cols = []
        for row, (ct, st) in zip(PARAMS, trig):
            ca, sa, aa, dd, off = row
            F = T * matrix(ca, sa, aa, 0, 1, 0)
            w = F[:3, 2]; p = F[:3, 3]; cols.append(w.col_join(p.cross(w)))
            if off == s.pi:
                ct, st = -ct, -st
            T = (T * matrix(ca, sa, aa, dd, ct, st)).applyfunc(s.simplify)
        J = s.Matrix.hstack(*cols).applyfunc(s.simplify)
        assert T == s.eye(4)
        off1 = J.extract([1, 3, 5], [2, 4, 6]).applyfunc(s.simplify)
        off2 = J.extract([0, 2, 4], [0, 1, 3, 5]).applyfunc(s.simplify)
        offdiag_zero = off1.is_zero_matrix and off2.is_zero_matrix
        rank = J.rank(simplify=True)
        # 平面切向
        A = J.extract([1, 3, 5], [0, 1, 3, 5])
        nsA = A.nullspace(simplify=True)
        vP = None
        if len(nsA) == 1:
            n = nsA[0]
            n = (n / n[1]).applyfunc(s.radsimp)  # q2 分量归一
            vP = s.zeros(7, 1)
            for i, c in zip([0, 1, 3, 5], n):
                vP[i] = s.simplify(c)
        # 空间切向
        B = J.extract([0, 2, 4], [2, 4, 6])
        nsB = B.nullspace(simplify=True)
        vS = None; cand_match = None
        if len(nsB) == 1:
            n = nsB[0]
            n = (n / n[0]).applyfunc(s.radsimp)  # q3 分量归一
            vS = s.zeros(7, 1)
            for i, c in zip([2, 4, 6], n):
                vS[i] = s.simplify(c)
            # 候选式中的 cos z 取 q4 槽位的实际余弦（含 H 侧符号变换）
            cand = s.zeros(7, 1); cand[2] = 1; cand[4] = -2 * cz; cand[6] = -1
            cand_match = bool(s.simplify(vS - cand).is_zero_matrix)
        kerP = vP is not None and (J * vP).applyfunc(s.simplify).is_zero_matrix
        kerS = vS is not None and (J * vS).applyfunc(s.simplify).is_zero_matrix
        disjoint = vP is not None and vS is not None and all(
            s.simplify(vP[i] * vS[i]) == 0 for i in range(7))
        indep = vP is not None and vS is not None and s.Matrix.hstack(vP, vS).rank(simplify=True) == 2
        good = bool(offdiag_zero and rank == 5 and kerP and kerS and disjoint and indep)
        ok_all &= good
        records.append({
            "state": ("H" if a else "G") + str(kk),
            "offdiagonal_blocks_zero": bool(offdiag_zero),
            "rank_J": int(rank),
            "v_P": [str(x) for x in vP] if vP is not None else None,
            "v_S": [str(x) for x in vS] if vS is not None else None,
            "v_S_equals_e3_minus_2cosz_e5_minus_e7": cand_match,
            "J_vP_zero": bool(kerP), "J_vS_zero": bool(kerS),
            "disjoint_supports": bool(disjoint), "independent": bool(indep),
            "passed": good,
        })
(OUT / "interface_transversality_exact_certificates.json").write_text(
    json.dumps(records, indent=2, ensure_ascii=False), encoding="utf8")
print(json.dumps(records, indent=2, ensure_ascii=False))
if not ok_all:
    raise SystemExit("接口横截性精确核验失败")
