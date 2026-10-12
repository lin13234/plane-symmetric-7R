"""s=sin q3=0 时完整闭环旋转第二行的精确形式（q7 取一般值）。

使用 build_full7r_algebra.py 生成的无约束原始多项式（不施加协调关系），
代入 (C3,S3)=(a,0)、(C5,S5)=(b,0)，a,b in {+1,-1}，其余关节角（含 q7）保持
一般单位圆坐标，精确核验
    T21 = a*b*sin q7,   T22 = a*b*cos q7,   T23 = 0.
因而 T21=0,T22=1 当且仅当 sin q7=0, cos q7=a*b=cos q3*cos q5，
即 q7 = q3+q5 (mod 2pi)。另记录 sin q5 一般时的 T21,T22 形式供参考。
失败则非零退出。
"""
from pathlib import Path
import json
import sympy as s

OUT = Path(__file__).resolve().parent
data = json.loads((OUT / 'full7r_unrestricted_polynomials.json').read_text(encoding='utf8'))
L = {name: s.Symbol(name) for name in data['coordinates']}
e = {r['id']: s.sympify(r['polynomial'], locals=L) for r in data['equations']}
C = [L[f'C{i}'] for i in range(1, 8)]; S = [L[f'S{i}'] for i in range(1, 8)]
T21 = e['loop_21']; T22 = e['loop_22'] + 1; T23 = e['loop_23']
circ = [C[i]**2 + S[i]**2 - 1 for i in (0, 1, 3, 4, 5, 6)]
ideal = s.groebner(circ, *S[:2], S[3], S[4], S[5], S[6], *C[:2], C[3], C[4], C[5], C[6],
                   order='grevlex', domain=s.QQ)


def red(expr):
    return s.expand(ideal.reduce(s.expand(expr))[1])


checks = {}; forms = {}
for a in (1, -1):
    for b in (1, -1):
        sub = {C[2]: a, S[2]: 0, C[4]: b, S[4]: 0}
        r21, r22, r23 = (red(X.subs(sub)) for X in (T21, T22, T23))
        key = f'cos_q3={a},cos_q5={b}'
        checks[key + ':T21=ab*sin_q7'] = s.expand(r21 - a * b * S[6]) == 0
        checks[key + ':T22=ab*cos_q7'] = s.expand(r22 - a * b * C[6]) == 0
        checks[key + ':T23=0'] = r23 == 0
        forms[key] = {'T21': str(r21), 'T22': str(r22), 'T23': str(r23)}
# 参考：sin q3=0 但 q5 一般
general = {}
for a in (1, -1):
    sub = {C[2]: a, S[2]: 0}
    general[f'cos_q3={a}'] = {'T21': str(s.factor(red(T21.subs(sub)))),
                              'T22': str(s.factor(red(T22.subs(sub))))}
result = {'source': 'full7r_unrestricted_polynomials.json (no coordination imposed)',
          'statement': 'sin q3=sin q5=0 => (T21,T22,T23)=cos q3*cos q5*(sin q7,cos q7,0) for every q1,q2,q4,q6,q7; '
                       'closure T21=0,T22=1 <=> q7 = q3+q5 mod 2pi',
          'forms_sin_q5_zero': forms, 'forms_general_q5': general,
          'checks': {k: bool(v) for k, v in checks.items()}, 'check_count': len(checks)}
result['passed'] = all(result['checks'].values())
(OUT / 's0_rotation_components_exact_certificates.json').write_text(
    json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
print(json.dumps(result, indent=2, ensure_ascii=False))
if not result['passed']:
    raise SystemExit('s=0 旋转分量精确核验失败')
