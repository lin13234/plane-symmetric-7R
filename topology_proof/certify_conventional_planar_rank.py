"""从原始物理轴独立求q5=0平面周期的全秩亏损条件。

先仅施加平面切片及旋转闭环，再由平移闭环消元。没有用旧SVD
结果认定整个周期的秩，也没有重新计算任何冻结轨迹。
"""
from pathlib import Path
import json
import sympy as s
from exact_model import PARAMS,matrix
OUT=Path(__file__).resolve().parent
C1,S1,CQ,SQ,H,V=s.symbols('C1 S1 CQ SQ H V',real=True)
circle=s.groebner([C1*C1+S1*S1-1,CQ*CQ+SQ*SQ-1,H*H+V*V-1],
                  S1,SQ,V,C1,CQ,H,domain=s.QQ)
def red(e):return s.factor(circle.reduce(s.expand(e))[1])
records=[]
for a in (0,1):
 sigma=(-1)**a
 trig=[(C1,S1),(CQ*C1+SQ*S1,SQ*C1-CQ*S1),(sigma,0),
       (H,V),(1,0),(H*CQ+sigma*V*SQ,V*CQ-sigma*H*SQ),(sigma,0)]
 T=s.eye(4);cols=[]
 for row,(ct,st) in zip(PARAMS,trig):
  ca,sa,aa,dd,off=row
  F=T*matrix(ca,sa,aa,0,1,0)
  w=F[:3,2];p=F[:3,3];cols.append(w.col_join(p.cross(w)))
  if off==s.pi:ct,st=-ct,-st
  T=(T*matrix(ca,sa,aa,dd,ct,st)).applyfunc(red)
 J=s.Matrix.hstack(*cols).applyfunc(red)
 B=J.extract([0,2,4],[2,4,6])
 A=J.extract([1,3,5],[0,1,3,5])
 det=red(B.det())
 closure=[red(T[i,3]) for i in range(3)]
 rec={'a':a,'passive_determinant_before_translation':str(det),
      'translation_equations':list(map(str,closure)),
      'active_3x3_minors':{},'passive_block':[[str(v) for v in B.row(i)] for i in range(3)]}
 from itertools import combinations
 for cc in combinations(range(4),3):rec['active_3x3_minors'][str(cc)]=str(red(A[:,cc].det()))
 records.append(rec)
(OUT/'conventional_planar_rank_expressions.json').write_text(json.dumps(records,indent=2),encoding='utf8')
print(json.dumps(records,indent=2))
