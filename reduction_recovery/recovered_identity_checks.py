"""仅核查本次补回的低阶代数恒等式，不重新求解已有运动分支。

输入采用历史分块文件已经保存的矩阵元素。此脚本不调用延拓、
优化器、数值 Jacobian/SVD、训练或全局消元，不改动任何源文件。
"""
from pathlib import Path
import json
import sympy as s

ROOT = Path(__file__).resolve().parent
cx, sx, cy, sy, cz, sz, cu, su = s.symbols('cx sx cy sy cz sz cu su', real=True)
g1 = sx*cz*(1+cu)+cx*su
g2 = cx*cy*sz-sy*cz
h3 = (4-5*cx*cz)*cy+5*(1-sz)*sy+1
h4 = sx*cz*su+cx*(1-cu)
e11 = (1-cu)*(sz*sy+cz*cy*cx)+sx*su*cy
e12 = su*(sz*sy+cz*cy*cx)+sx*cy*(1+cu)
e21 = (1+cu)*(sz*cy-cz*sy*cx)+sy*sx*su
e22 = sy*sx*(cu-1)-su*(sz*cy-cz*sy*cx)
circle = s.groebner([cx**2+sx**2-1, cy**2+sy**2-1,
                    cz**2+sz**2-1, cu**2+su**2-1],
                   sx,sy,sz,su,cx,cy,cz,cu)

def circle_zero(expr):
    """在四个单位圆恒等式的商环中验证多项式为零。"""
    return s.expand(circle.reduce(s.expand(expr))[1]) == 0

checks = {}
checks['cz_E11'] = circle_zero(cz*e11-cy*h4+sz*(1-cu)*g2)
checks['cz_E12'] = circle_zero(cz*e12-cy*g1+sz*su*g2)
checks['cz_E21'] = circle_zero(cz*e21-sx*cy*sz*g1-(cx*cz*(1+cu)-sx*su)*g2)
checks['cz_E22'] = circle_zero(cz*e22+sx*cy*sz*h4-(-cx*cz*su-sx*(cu-1))*g2)

# 两个含 u 的方程共同约束半角向量；不丢掉 cos(u/2)=0 支。
C,S,b,c = s.symbols('C S b c', real=True)
checks['u_factor_g1'] = s.expand(b*(2*C**2)+c*(2*S*C)-2*C*(b*C+c*S)) == 0
checks['u_factor_h4'] = s.expand(b*(2*S*C)+c*(2*S**2)-2*S*(b*C+c*S)) == 0

# 消去 y 的平方关系；保留有符号重建，避免 tan(y) 的 pi 歧义。
N=4*cz+5*cx*(sz-1)
D2=cz**2+cx**2*sz**2
checks['N2_minus_D2'] = circle_zero(N**2-D2-(15*cz**2+40*cx*cz*(sz-1)+cx**2*(25*(sz-1)**2-sz**2)))
Z,X=s.symbols('Z X', real=True)
P=-(5*Z**2-12*Z+5)*(5*Z**2-8*Z+5)*c**2-40*(Z-1)**3*(Z+1)*c-15*(Z-1)**2*(Z+1)**2
Q=(4*(1-Z**2)-5*c*(Z-1)**2)**2-(1-Z**2)**2-4*c**2*Z**2
checks['quartic_bridge_Q_equals_minus_P'] = s.expand(Q+P)==0
PX=5*(X**2-4)*Z**4+5*(X**2-1)*(X**2-9)*Z**3-(29*X**4-88*X**2+29)*Z**2+5*(X**2-1)*(9*X**2-1)*Z-5*X**2*(4*X**2-1)
checks['X_chart_factor_four'] = s.cancel((1+X**2)**2*P.subs(c,(1-X**2)/(1+X**2))-4*PX)==0
# 比较已存的大有理恢复式与由有符号圆向量得到的紧凑恢复式。
# 仅对两个已知表达式做多项式余式检查，不重新做机构学消元。
A=X**4*Z**4-5*X**4*Z**3+4*X**4*Z**2+5*X**4*Z-4*X**4-3*X**2*Z**4+4*X**2*Z**2-3*X**2-4*Z**4+5*Z**3+4*Z**2-5*Z+1
RY=Z*(X**2-1)*(Z-1)*(X**2*Z-4*X**2-4*Z+1)/A
RYsmall=-2*(1-X**2)*Z/(3*(1-Z**2)*(1+X**2)-5*(1-X**2)*(Z-1)**2)
diff_num=s.cancel(RY-RYsmall).as_numer_denom()[0]
checks['saved_Y_recovery_modulo_quartic'] = s.rem(diff_num,PX,Z)==0
checks['reciprocal_quartic'] = s.cancel(Z**4*P.subs({Z:1/Z,c:-c},simultaneous=True)-P)==0
checks['leading_coefficient'] = s.expand(s.Poly(P,Z).coeff_monomial(Z**4)+5*(c+1)*(5*c+3))==0
checks['c_zero_double_roots'] = s.expand(P.subs(c,0)+15*(Z-1)**2*(Z+1)**2)==0
checks['c_one_gate_factors'] = s.expand(P.subs(c,1)+4*Z*(4*Z-5)*(5*Z**2-5*Z+1))==0
checks['c_minus_one_gate_factors'] = s.expand(P.subs(c,-1)-4*(5*Z-4)*(Z**2-5*Z+5))==0

result={'scope':'newly restored polynomial identities only; no branch recomputation',
        'checks':checks,'passed':all(checks.values()),'number_of_checks':len(checks),
        'sympy_version':s.__version__}
(ROOT/'recovered_identity_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
if not result['passed']:
    raise SystemExit('恢复恒等式核查失败；禁止把失败项称为已证明。')
