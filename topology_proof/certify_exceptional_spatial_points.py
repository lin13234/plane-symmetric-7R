"""认证约化退化图的8个具名(±q3)状态在完整7R中正则。

使用半角图之外的单位圆坐标，完整变换与6阶旋量子式均精确。
不计算或改写任何已存在721点轨迹。
"""
from pathlib import Path
import json
import sympy as s
from exact_model import PARAMS,matrix
OUT=Path(__file__).resolve().parent
records=[]
for sx in (-1,1):
 for sz in (-1,1):
  for eps in (-1,1):
    if sz==1:cy=-s.Rational(1,4);sy=eps*s.sqrt(15)/4
    else:cy=(-4+10*eps*s.sqrt(115))/116;sy=(-10-4*eps*s.sqrt(115))/116
    cu=s.simplify(sy*sy-cy*cy);su=s.simplify(-2*sx*sz*sy*cy)
    trig=[(cy,sy),(cy,sy),(0,sx),(0,sz),(cu,su),(0,-sz),(0,-sx)]
    T=s.eye(4);cols=[]
    for row,(ct,st) in zip(PARAMS,trig):
        ca,sa,a,d,off=row
        F=T*matrix(ca,sa,a,0,1,0);omega=F[:3,2];p=F[:3,3]
        cols.append(omega.col_join(p.cross(omega)))
        if off==s.pi:ct,st=-ct,-st
        T=T*matrix(ca,sa,a,d,ct,st)
    E=(T-s.eye(4)).applyfunc(s.simplify)
    S=s.Matrix.hstack(*cols).applyfunc(s.simplify)
    determinant=s.factor(s.simplify(S[:,[0,1,3,4,5,6]].det()))
    records.append({'sin_x':sx,'sin_z':sz,'epsilon':eps,'cos_y':str(cy),'sin_y':str(sy),
                    'cos_u':str(cu),'sin_u':str(su),'complete_loop_identity':E==s.zeros(4),
                    'passive_6x6_screw_minor':str(determinant),'rank6':determinant!=0})
(OUT/'exceptional_spatial_exact_certificates.json').write_text(json.dumps(records,indent=2),encoding='utf8')
print(json.dumps(records,indent=2))
if not all(r['complete_loop_identity'] and r['rank6'] for r in records):raise SystemExit('退化点证书失败')
