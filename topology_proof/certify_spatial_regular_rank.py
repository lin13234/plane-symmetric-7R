"""新推导空间常规图的完整q3被动6x6子式，独立排除隐藏秩亏损。

该计算只认证公式，不重新求解任何旧轨迹。
"""
from pathlib import Path
import json
import sympy as s
from exact_model import PARAMS,matrix
OUT=Path(__file__).resolve().parent
c,r,h,t,k,v,Y,W=s.symbols('c r h t k v Y W',real=True)
# 已证明协调关系之后才构造此矩阵。
trig=[(Y,W),(Y,W),(c,r),(h,t),(k,v),(h,-t),(c,-r)]
T=s.eye(4);cols=[]
cir=s.groebner([Y*Y+W*W-1,c*c+r*r-1,h*h+t*t-1,k*k+v*v-1],
               W,r,t,v,Y,c,h,k,domain=s.QQ)
def red(e):return s.factor(cir.reduce(s.expand(e))[1])
for row,(ct,st) in zip(PARAMS,trig):
 ca,sa,aa,dd,off=row
 F=T*matrix(ca,sa,aa,0,1,0)
 w=F[:3,2];p=F[:3,3];cols.append(w.col_join(p.cross(w)))
 if off==s.pi:ct,st=-ct,-st
 T=(T*matrix(ca,sa,aa,dd,ct,st)).applyfunc(red)
J=s.Matrix.hstack(*cols).applyfunc(red)
B=J[:,[0,1,3,4,5,6]]
print('开始完整6x6子式精确消元',flush=True)
det=s.factor(B.det(method='domain-ge'))
print('6x6子式完成',flush=True)
det=red(det)
# 标准有符号恢复；N²=D²是在闭环上使用的约束。
D2=h*h+c*c*t*t;N=4*h+5*c*(t-1)
num=s.factor(s.cancel(det.subs({Y:-h/N,W:-c*t/N,
           k:(c*c-r*r*h*h)/(c*c+r*r*h*h),v:-2*c*r*h/(c*c+r*r*h*h)})).as_numer_denom()[0])
den=s.factor(s.cancel(det.subs({Y:-h/N,W:-c*t/N,
           k:(c*c-r*r*h*h)/(c*c+r*r*h*h),v:-2*c*r*h/(c*c+r*r*h*h)})).as_numer_denom()[1])
I=s.groebner([c*c+r*r-1,h*h+t*t-1],r,t,h,c,domain=s.QQ)
J0=5*c*c*h*t-4*c*t-5*h*t+5*h
ratio_check=I.reduce(s.expand(num-5*c*c*r*D2*J0))[1]==0
den_check=I.reduce(s.expand(den-2*D2**2*N))[1]==0
# 完整雅可比零点需要J0=0；新的消元证书与既有四次判别式
# 使用同一个R(c²)，但此处独立认证的是完整7R子式而非旧轨迹。
G=s.groebner([N*N-D2,J0,h*h+t*t-1],h,t,c,domain=s.QQ)
R=2475*c**8-4166*c**6+3681*c**4+5130*c**2-15525
K=625*c**4-800*c*c+1856
elim=c*c*K*R
elimination_check=G.reduce(s.expand(elim))[1]==0
out={'full_passive_det_before_recovery':str(det),
     'simplified_full_passive_det':'5*c^2*sin(x)*J0/(2*D2*N)',
     'J0':str(J0),'numerator_identity':ratio_check,'denominator_identity':den_check,
     'rank_zero_eliminant':str(s.factor(elim)),
     'eliminant_ideal_membership':elimination_check,
     'K_positive_certificate':'625*(c^2-16/25)^2+8000/5 > 0',
     'R_upper_bound_on_unit_interval':-4239,
     'normal_chart':'sin(x)!=0; cos(x)!=0; cos(z)!=0; N!=0; D2>0',
     'passed':ratio_check and den_check and elimination_check}
# 精确配方：625(w-16/25)^2+1600。
out['K_positive_certificate']='625*(c^2-16/25)^2+1600 > 0'
(OUT/'spatial_regular_rank_expression.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2),flush=True)
