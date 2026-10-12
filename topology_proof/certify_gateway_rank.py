"""认证八个平面/空间交点的完整闭环及完整6x7旋量秩。

各角以代数三角值保存，避免浮点接近点冒充交点。六阶子式由
平面互补块det=0全部消失，再逐点寻找非零五阶子式。
"""
from pathlib import Path
from itertools import combinations
import json
import sympy as s
from exact_model import PARAMS,matrix
OUT=Path(__file__).resolve().parent
zs=[s.Integer(0),(5-s.sqrt(5))/10,s.Rational(5,4),(5+s.sqrt(5))/10]
records=[]
for a in (0,1):
 for kk,Z in enumerate(zs,1):
  cz=s.simplify((1-Z*Z)/(1+Z*Z));sz=s.simplify(2*Z/(1+Z*Z))
  sign=1 if kk in (1,3) else -1
  cy=sign*cz;sy=sign*sz
  if a:cz,sz=-cz,sz
  sig=(-1)**a
  trig=[(cy,sy),(cy,sy),(sig,0),(cz,sz),(1,0),(cz,-sz),(sig,0)]
  T=s.eye(4);cols=[]
  for row,(ct,st) in zip(PARAMS,trig):
   ca,sa,aa,dd,off=row
   F=T*matrix(ca,sa,aa,0,1,0)
   w=F[:3,2];p=F[:3,3];cols.append(w.col_join(p.cross(w)))
   if off==s.pi:ct,st=-ct,-st
   T=(T*matrix(ca,sa,aa,dd,ct,st)).applyfunc(s.simplify)
  J=s.Matrix.hstack(*cols).applyfunc(s.simplify)
  B=J.extract([0,2,4],[2,4,6])
  # 活动平面块秩3，加被动块秩2即可完整秩5。
  A=J.extract([1,3,5],[0,1,3,5])
  active=None;passive=None
  for cc in combinations(range(4),3):
   det=s.simplify(A[:,cc].det())
   if det!=0:active={'cols':[ [0,1,3,5][i]+1 for i in cc],'det':str(det)};break
  for rr in combinations(range(3),2):
   for cc in combinations(range(3),2):
    det=s.simplify(B.extract(rr,cc).det())
    if det!=0:passive={'rows':[[0,2,4][i]+1 for i in rr],
                         'cols':[[2,4,6][i]+1 for i in cc],'det':str(det)};break
   if passive:break
  rec={'state':('H' if a else 'G')+str(kk),'cos_y':str(cy),'sin_y':str(sy),
       'cos_z':str(cz),'sin_z':str(sz),'full_loop_identity':T==s.eye(4),
       'det_passive_block':str(s.simplify(B.det())),
       'nonzero_active_minor':active,'nonzero_passive_minor':passive,
       'full_rank':5 if active and passive and s.simplify(B.det())==0 else None}
  records.append(rec)
(OUT/'gateway_rank_exact_certificates.json').write_text(json.dumps(records,indent=2),encoding='utf8')
print(json.dumps(records,indent=2))
if not all(r['full_loop_identity'] and r['full_rank']==5 for r in records):
 raise SystemExit('交点闭环/秩认证失败')
