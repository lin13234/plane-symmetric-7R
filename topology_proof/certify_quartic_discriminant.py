"""从当前冻结 MDH 参数从头推导约化四次式 P(Z;c) 并精确计算其判别式。

不读取旧文件中的 P：
1. 在约化坐标 q=[y,y,x,z,u,-z,-x] 下，用 exact_model.PARAMS 构造 A_1..A_7，
   计算 E_b=B1B2B3B4-B7^{-1}B6^{-1}B5^{-1}（B=(A2..A7,A1)），并核验补充材料
   S6 中列出的非零块元素 g1,g2,h3,h4 等（模单位圆关系）；
2. (g2,h3) 关于 (cos y,sin y) 线性，行列式 N；(g1,h4) 关于 (cos u,sin u) 线性，
   行列式 D^2；Cramer 解后施加 cos^2+sin^2=1 得到消元式 N^2-D^2；
3. 用 Z=tan(z/2) 有理化，P:=-(1+Z^2)^2 (N^2-D^2)，与补充材料 Eq.(P) 比较；
4. 精确计算 disc_Z P，与 -102400 c^4 R(c^2) 比较，并核验文中引用的特殊值。
全部为 sympy 精确有理运算。失败则非零退出。
"""
from pathlib import Path
import json
import sympy as s
from exact_model import PARAMS, matrix

OUT = Path(__file__).resolve().parent
cx, sx, cy, sy, cz, sz, cu, su = s.symbols('c_x s_x c_y s_y c_z s_z c_u s_u', real=True)
Z, c, w = s.symbols('Z c w', real=True)
checks = {}

# ---------- 1. 约化闭环块 E_b ----------
trig = [(cy, sy), (cy, sy), (cx, sx), (cz, sz), (cu, su), (cz, -sz), (cx, -sx)]
A = []
for row, (ct, st) in zip(PARAMS, trig):
    ca, sa, a, d, off = row
    if off == s.pi:
        ct, st = -ct, -st
    A.append(matrix(ca, sa, a, d, ct, st))


def hinv(M):
    R = M[:3, :3]; p = M[:3, 3]
    return (R.T).row_join(-R.T * p).col_join(s.Matrix([[0, 0, 0, 1]]))


B = A[1:] + A[:1]
Eb = (B[0] * B[1] * B[2] * B[3] - hinv(B[6]) * hinv(B[5]) * hinv(B[4])).applyfunc(s.expand)
ideal = s.groebner([cx**2 + sx**2 - 1, cy**2 + sy**2 - 1, cz**2 + sz**2 - 1, cu**2 + su**2 - 1],
                   sx, sy, sz, su, cx, cy, cz, cu, order='grevlex', domain=s.QQ)


def zero(expr):
    return s.expand(ideal.reduce(s.expand(expr))[1]) == 0


Kb = sz * sy + cz * cy * cx; Hb = sz * cy - cz * sy * cx
g1 = sx * cz * (1 + cu) + cx * su
g2 = cx * cy * sz - cz * sy
h3 = (4 - 5 * cx * cz) * cy + 5 * (1 - sz) * sy + 1
h4 = sx * cz * su + cx * (1 - cu)
claimed = {(0, 0): (1 - cu) * Kb + sx * su * cy, (0, 1): su * Kb + sx * cy * (1 + cu),
           (0, 2): -2 * g2, (0, 3): h3,
           (1, 0): (1 + cu) * Hb + sx * sy * su, (1, 1): sx * sy * (cu - 1) - su * Hb,
           (2, 0): -g1, (2, 1): h4}
for i in range(4):
    for j in range(4):
        checks[f'Eb_{i+1}{j+1}_matches_S6'] = zero(Eb[i, j] - claimed.get((i, j), 0))
checks['cz_Eb11_identity'] = zero(cz * Eb[0, 0] - (cy * h4 - sz * (1 - cu) * g2))
checks['cz_Eb12_identity'] = zero(cz * Eb[0, 1] - (cy * g1 - sz * su * g2))
checks['cz_Eb21_identity'] = zero(cz * Eb[1, 0] - (sx * cy * sz * g1 + (cx * cz * (1 + cu) - sx * su) * g2))
checks['cz_Eb22_identity'] = zero(cz * Eb[1, 1] - (-sx * cy * sz * h4 + (-cx * cz * su - sx * (cu - 1)) * g2))

# ---------- 2. 线性消元 ----------
My = s.Matrix([[g2.coeff(cy), g2.coeff(sy)], [h3.coeff(cy), h3.coeff(sy)]]).applyfunc(s.expand)
ry = s.Matrix([-(g2 - g2.coeff(cy) * cy - g2.coeff(sy) * sy), -(h3 - h3.coeff(cy) * cy - h3.coeff(sy) * sy)]).applyfunc(s.expand)
g1e, h4e = s.expand(g1), s.expand(h4)
Mu = s.Matrix([[g1e.coeff(cu), g1e.coeff(su)], [h4e.coeff(cu), h4e.coeff(su)]])
ru = s.Matrix([-(s.expand(g1) - s.expand(g1).coeff(cu) * cu - s.expand(g1).coeff(su) * su),
               -(s.expand(h4) - s.expand(h4).coeff(cu) * cu - s.expand(h4).coeff(su) * su)]).applyfunc(s.expand)
assert all(v.free_symbols.isdisjoint({cy, sy}) for v in list(My) + list(ry))
assert all(v.free_symbols.isdisjoint({cu, su}) for v in list(Mu) + list(ru))
h, t = cz, sz
N = 4 * h + 5 * cx * (t - 1)
D2 = h**2 + cx**2 * t**2
circ = {cz**2: 1 - sz**2}
detY = My.det(); detU = Mu.det()
checks['det_y_system_equals_N_mod_circle'] = s.expand((detY - N).subs(cz**2, 1 - sz**2)) == 0
checks['det_u_system_equals_D2_mod_circle'] = zero(detU - (cx**2 + sx**2 * cz**2)) and zero(cx**2 + sx**2 * cz**2 - D2)
adjY = My.adjugate() * ry  # N*(cos y, sin y)
checks['cramer_N_cos_y_eq_minus_h'] = zero(adjY[0] - (-h))
checks['cramer_N_sin_y_eq_minus_ct'] = zero(adjY[1] - (-cx * t))
adjU = Mu.adjugate() * ru  # D^2*(cos u, sin u)
checks['cramer_u_unit_norm_identity'] = zero(adjU[0]**2 + adjU[1]**2 - detU**2)
# 消元式：(N cos y)^2+(N sin y)^2 = N^2  =>  h^2+c^2 t^2 - N^2 = 0
elim = s.expand(adjY[0]**2 + adjY[1]**2 - detY**2)
checks['eliminant_equals_D2_minus_N2'] = zero(elim - (D2 - N**2))

# ---------- 3. Weierstrass 有理化 ----------
hZ = (1 - Z**2) / (1 + Z**2); tZ = 2 * Z / (1 + Z**2)
Pderived = s.factor(s.cancel(-(1 + Z**2)**2 * (N**2 - D2).subs({cz: hZ, sz: tZ, cx: c})))
Pderived = s.expand(Pderived)
Ptext = s.expand(-(5 * Z**2 - 12 * Z + 5) * (5 * Z**2 - 8 * Z + 5) * c**2
                 - 40 * (Z - 1)**3 * (Z + 1) * c - 15 * (Z - 1)**2 * (Z + 1)**2)
checks['P_derived_equals_text_EqP'] = s.expand(Pderived - Ptext) == 0
checks['P_identity_minus_(1+Z2)^2_(N2-D2)'] = s.expand(
    s.cancel(Ptext + (1 + Z**2)**2 * (N**2 - D2).subs({cz: hZ, sz: tZ, cx: c}))) == 0

# ---------- 4. 判别式与特殊值 ----------
PP = s.Poly(Ptext, Z)
disc = s.factor(s.discriminant(PP, Z))
R = 2475 * w**4 - 4166 * w**3 + 3681 * w**2 + 5130 * w - 15525
checks['discriminant_equals_-102400_c4_R(c2)'] = s.expand(disc + 102400 * c**4 * R.subs(w, c**2)) == 0
checks['leading_coeff_k4'] = s.expand(PP.LC() + 5 * (c + 1) * (5 * c + 3)) == 0
checks['P(1,c)=4c^2'] = s.expand(Ptext.subs(Z, 1) - 4 * c**2) == 0
checks['P(-1,c)=-396c^2'] = s.expand(Ptext.subs(Z, -1) + 396 * c**2) == 0
checks['P(Z,1)'] = s.expand(Ptext.subs(c, 1) + 4 * Z * (4 * Z - 5) * (5 * Z**2 - 5 * Z + 1)) == 0
checks['P(Z,-1)'] = s.expand(Ptext.subs(c, -1) - 4 * (5 * Z - 4) * (Z**2 - 5 * Z + 5)) == 0
checks['P(Z,0)'] = s.expand(Ptext.subs(c, 0) + 15 * (Z - 1)**2 * (Z + 1)**2) == 0
checks['cubic_at_c=-3/5'] = s.expand(Ptext.subs(c, -s.Rational(3, 5))
                                     - (-12 * Z**3 - s.Rational(564, 25) * Z**2 + 84 * Z - 48)) == 0
P0 = s.factor(Ptext.subs(Z, 0))
checks['P(0,c)<0_on_[-1,0]'] = s.expand(P0 + 5 * (5 * c - 3) * (c - 1)) == 0  # -5(5c-3)(c-1)<0 for c<=0
# R(w) 在 [0,1] 上的严格负性：精确 Sturm 计数为 0 个根且 R(0)<0
checks['R_no_root_in_[0,1]'] = s.Poly(R, w).count_roots(0, 1) == 0
checks['R(0)<0'] = R.subs(w, 0) < 0
checks['R_crude_bound_-4239'] = 2475 + 3681 + 5130 - 15525 == -4239

result = {'source': 'derived from exact_model.PARAMS; P not read from any previous file',
          'reduced_vector': '[y,y,x,z,u,-z,-x]',
          'N': str(N), 'D2': str(D2),
          'P_derived': str(s.collect(Pderived, c)),
          'P_derived_factor_in_c': str(s.collect(s.factor_terms(Pderived), c)),
          'discriminant_factored': str(disc),
          'R(w)': str(R),
          'P(0,c)': str(P0),
          'checks': {k: bool(v) for k, v in checks.items()},
          'check_count': len(checks)}
result['passed'] = all(result['checks'].values())
(OUT / 'quartic_discriminant_exact_certificates.json').write_text(
    json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
print(json.dumps(result, indent=2, ensure_ascii=False))
if not result['passed']:
    raise SystemExit('四次式/判别式精确核验失败')
