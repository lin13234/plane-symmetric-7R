"""建立不施加空间协调关系的完整七角实代数闭环。

每个角采用(C_i,S_i)单位圆坐标，共14个变量；固定π偏置只改变
对应三角对符号。该表示覆盖所有关节角，包括tan(q_i/2)无穷远图。
"""
from pathlib import Path
import json,hashlib
import sympy as s
from exact_model import PARAMS,matrix
OUT=Path(__file__).resolve().parent
cs=s.symbols('C1:8');ss=s.symbols('S1:8')
T=s.eye(4)
for i,row in enumerate(PARAMS):
    ca,sa,a,d,off=row;ct,st=cs[i],ss[i]
    if off==s.pi:ct,st=-ct,-st
    T=T*matrix(ca,sa,a,d,ct,st)
eq=[]
for i in range(3):
 for j in range(4):
    v=s.expand(T[i,j]-(1 if i==j else 0))
    eq.append({'id':f'loop_{i+1}{j+1}','polynomial':str(v),'terms':len(s.Poly(v,*cs,*ss).terms())})
for i in range(7):eq.append({'id':f'circle_{i+1}','polynomial':str(cs[i]**2+ss[i]**2-1),'terms':3})
out={'model':'current-J1-J7; l=1; no coordination imposed',
     'coordinates':list(map(str,cs+ss)),'equations':eq,'total_equations':len(eq),
     'coverage':'all real torus angles, including individual pi angles',
     'coordination_imposed':False}
(OUT/'full7r_unrestricted_polynomials.json').write_text(json.dumps(out,indent=2),encoding='utf8')
(OUT/'full7r_unrestricted_polynomials.txt').write_text('\n\n'.join(e['id']+' = '+e['polynomial']+' = 0' for e in eq),encoding='utf8')
# 源材料只读散列，冻结轨迹不执行。
prior=Path('source_registry.json')
records=json.loads(prior.read_text(encoding='utf8'))
chosen=[r for r in records if r['source_id'] in ['S01','S20','D01','D02','D03','D04']]
checks=[]
for r in chosen:
    p=Path(r['path']);h=hashlib.sha256(p.read_bytes()).hexdigest()
    checks.append({'path':str(p),'sha256_before':r['sha256'],'sha256_current':h,'unchanged':h==r['sha256']})
(OUT/'protected_sources.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
if not all(r['unchanged'] for r in checks):raise SystemExit('冻结输入发生变化')
print(json.dumps({'equations':len(eq),'loop_term_counts':[e['terms'] for e in eq[:12]],
                  'protected_sources_unchanged':True},indent=2))
