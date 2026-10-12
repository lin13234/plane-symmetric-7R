"""精确认证q5=pi折叠平面周期并提取其完整秩亏损方程。

仅计算新发现的候选，不复算任何既有721点轨迹。用代数数与
cos(t),sin(t)构造完整MDH、物理轴及6x7旋量矩阵。
"""
from pathlib import Path
import json
import sympy as s
from exact_model import PARAMS,matrix
OUT=Path(__file__).resolve().parent
C,S=s.symbols('C S',real=True)
def red(v):
    """先在代数数域化简，再利用C²+S²=1整理。"""
    v=s.cancel(s.expand(v))
    return s.rem(v,s.Poly(C**2+S**2-1,S),S).as_expr().expand().simplify()

allrecords=[]
for a in (0,1):
 for eps in (-1,1):
    c1=(-4+10*eps*s.sqrt(10))/41
    s1=(-5-8*eps*s.sqrt(10))/41
    # (2-5i/2)exp(iQ)=-(2+5i/2)-exp(iq1)。
    cQ=s.simplify((-2-c1)*s.Rational(8,41)-(-s.Rational(5,2)-s1)*s.Rational(10,41))
    sQ=s.simplify((-2-c1)*s.Rational(10,41)+(-s.Rational(5,2)-s1)*s.Rational(8,41))
    c2=s.simplify(cQ*c1+sQ*s1);s2=s.simplify(sQ*c1-cQ*s1)
    sigma=(-1)**a
    trig=[(c1,s1),(c2,s2),(sigma,0),(C,S),(-1,0),
          (cQ*C+sigma*sQ*S,sigma*sQ*C-cQ*S),(-sigma,0)]
    T=s.eye(4);cols=[]
    for row,(ct,st) in zip(PARAMS,trig):
      ca,sa,aa,dd,off=row
      F=T*matrix(ca,sa,aa,0,1,0)
      omega=F[:3,2];point=F[:3,3]
      cols.append(omega.col_join(point.cross(omega)))
      if off==s.pi:ct,st=-ct,-st
      T=T*matrix(ca,sa,aa,dd,ct,st)
    E=(T-s.eye(4)).applyfunc(red)
    screw=s.Matrix.hstack(*cols).applyfunc(red)
    B=screw.extract([0,2,4],[2,4,6])
    detB=s.factor(red(B.det()))
    Ap=screw.extract([1,3,5],[0,1,3])
    detA=s.factor(red(Ap.det()))
    rec={'a':a,'epsilon':eps,'cos_q1':str(c1),'sin_q1':str(s1),
         'cos_Q':str(cQ),'sin_Q':str(sQ),'cos_q2':str(c2),'sin_q2':str(s2),
         'full_loop_identity':E==s.zeros(4),'active_3x3_minor':str(detA),
         'passive_block_determinant':str(detB),
         'screw_matrix':[[str(v) for v in screw.row(i)] for i in range(6)]}
    allrecords.append(rec)
(OUT/'folded_planar_exact_certificates.json').write_text(json.dumps(allrecords,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps([{k:v for k,v in r.items() if k!='screw_matrix'} for r in allrecords],ensure_ascii=False,indent=2))
if not all(r['full_loop_identity'] for r in allrecords):raise SystemExit('折叠周期闭环证书失败')
