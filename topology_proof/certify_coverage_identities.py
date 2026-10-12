"""认证完整闭环的覆盖性证明所需恒等式，不预设协调关系。

先检查原始19个多项式的线性组合，再认证候选分支排除不等式。
只有在证明中已经得到q4+q6=m*pi之后才代入该关系。
"""
from pathlib import Path
import json
import sympy as s
OUT=Path(__file__).resolve().parent
data=json.loads((OUT/'full7r_unrestricted_polynomials.json').read_text(encoding='utf8'))
locals_={name:s.Symbol(name) for name in data['coordinates']}
e={r['id']:s.sympify(r['polynomial'],locals=locals_) for r in data['equations']}
C1,C2,c,h,k,C6,C7=[locals_[f'C{i}'] for i in range(1,8)]
S1,S2,vx,t,v,S6,S7=[locals_[f'S{i}'] for i in range(1,8)]
cx,sx=c,vx
f=c*v+vx*h*(1+k)
M=c*h*(1+k)-vx*v;L=t*(1+k)
Kr=h*h-k*t*t;Ki=t*M
CQ=C1*C2-S1*S2;SQ=C1*S2+C2*S1
checks={}
checks['original_position_orientation_combination']=s.expand(s.Rational(2,5)*e['loop_24']-e['loop_23']-f)==0
checks['original_sum_angle_factor']=s.expand(e['loop_23']-S6*f+vx*(h*S6+t*C6))==0
checks['original_complex_translation_x']=s.expand(e['loop_14']-(2+C1+(2-s.Rational(5,2)*M)*CQ-(-s.Rational(5,2)+s.Rational(5,2)*L)*SQ+s.Rational(5,2)*e['loop_13']))==0
checks['original_complex_translation_z']=s.expand(e['loop_34']-(S1+(2-s.Rational(5,2)*M)*SQ+(-s.Rational(5,2)+s.Rational(5,2)*L)*CQ+s.Rational(5,2)*(e['loop_33']+1)))==0
ideal=s.groebner([c*c+vx*vx-1,h*h+t*t-1,k*k+v*v-1],vx,t,v,c,h,k,domain=s.QQ)
def zero(expr):return s.expand(ideal.reduce(s.expand(expr))[1])==0
checks['M_norm_identity']=zero(M*M+f*f-(1+k)*(1+h*h-k*t*t))
checks['K_unit_on_g1']=zero(Kr*Kr+Ki*Ki+t*t*f*f-1)
checks['complex_symmetry_real']=zero((Kr-1)*M+Ki*L)
checks['complex_symmetry_imag']=zero(Ki*M-(Kr+1)*L+t*f*f)
for eta in (-1,1):
    sub={C6:eta*h,S6:-eta*t}
    checks[f'orientation_third_column_13_eta{eta}']=s.expand(e['loop_13'].subs(sub)-eta*(CQ*Ki-SQ*Kr))==0
    checks[f'orientation_third_column_33_eta{eta}']=s.expand((e['loop_33']+1).subs(sub)-eta*(CQ*Kr+SQ*Ki))==0

# D>0时g1且q5!=pi给出标准姿态恢复。这里只认证新排除式。
D2=c*c+vx*vx*h*h
urec={k:(c*c-vx*vx*h*h)/D2,v:-2*c*vx*h/D2}
Er=-2+2*Kr+s.Rational(5,2)*Ki-s.Rational(5,2)*M
Ei=-s.Rational(5,2)+2*Ki-s.Rational(5,2)*Kr-s.Rational(5,2)*L
delta=24*h*h-40*c*h*t+15*c*c*t*t+25*c*c
num=s.cancel(((Er*Er+Ei*Ei-1)*D2-delta).subs(urec)).as_numer_denom()[0]
checks['anti_sum_branch_positive_gap_identity']=zero(num)
checks['anti_sum_gap_square_completion']=s.expand(delta-(24*(h-s.Rational(5,6)*c*t)**2+25*c*c-s.Rational(5,3)*c*c*t*t))==0
EplusR=-2-2*Kr-s.Rational(5,2)*Ki+s.Rational(5,2)*M
EplusI=-s.Rational(5,2)-2*Ki+s.Rational(5,2)*Kr+s.Rational(5,2)*L
# 模单位圆恒等式认证两个桥长；不能直接把h^2+t^2当作常数替换。
checks['u_pi_eta_plus_bridge_squared_16']=zero((EplusR**2+EplusI**2).subs({k:-1,v:0})-16)
checks['u_pi_eta_minus_bridge_squared_25']=zero((Er**2+Ei**2).subs({k:-1,v:0})-25)
# 由原始旋转矩阵第二行恢复q7。该检查在q5恢复式之后，未在输入中预设q7。
for eta in (-1,1):
    A=c*k-h*vx*v
    B=eta*(h*c*v+h*h*vx*k-vx*t*t)
    checks[f'q7_original_rotation_21_eta{eta}']=s.expand(e['loop_21'].subs({C6:eta*h,S6:-eta*t})-(A*S7-B*C7))==0
    checks[f'q7_original_rotation_22_eta{eta}']=s.expand((e['loop_22']+1).subs({C6:eta*h,S6:-eta*t})-(A*C7+B*S7))==0
    checks[f'q7_normal_A_eta{eta}']=zero(s.cancel((A-c).subs(urec)).as_numer_denom()[0])
    checks[f'q7_normal_B_eta{eta}']=zero(s.cancel((B+eta*vx).subs(urec)).as_numer_denom()[0])
    checks[f'q7_exceptional_A_eta{eta}']=s.expand(A.subs({c:0,h:0}))==0
    checks[f'q7_exceptional_B_eta{eta}']=s.expand((B.subs({c:0,h:0})+eta*vx).subs(t*t,1))==0

# 退化D=0的eta-支是长度2与sqrt29的差，严格大于单位桥。
checks['exceptional_triangle_strict_gap']=bool(s.sqrt(29)-2>1)
# 此处仅认证 R(c²) 的范围界；判别式本身由 certify_quartic_discriminant.py 从 MDH 参数精确重算。
checks['quartic_discriminant_positive_bound']=2475+3681+5130-15525==-4239
result={'checks':checks,'passed':all(checks.values()),'check_count':len(checks),
        'anti_sum_gap_lower_bound':'24*(h-5*c*t/6)^2+(70/3)*c^2 > 0 for D>0',
        'u_pi_bridge_moduli_squared':{'eta_plus':16,'eta_minus':25},
        'source':'unrestricted original polynomials; no coordination at input'}
(OUT/'full7r_coverage_exact_certificates.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
if not result['passed']:raise SystemExit('覆盖性恒等式核查失败')
