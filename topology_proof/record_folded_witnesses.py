"""保存新折叠周期的七角反例与连续参数路径，不重跑既有轨迹。"""
from pathlib import Path
import csv,json
import numpy as np
from exact_model import numeric_loop
OUT=Path(__file__).resolve().parent
records=[];path=[]
for a in (0,1):
 for eps in (-1,1):
    theta=np.arctan2((-5-8*eps*np.sqrt(10))/41,(-4+10*eps*np.sqrt(10))/41)
    for t in (0.,0.4,np.pi/2):
        q=np.array([theta,theta,a*np.pi,t,np.pi,(-1)**a*2*theta-t,((a+1)%2)*np.pi])
        E=numeric_loop(q)-np.eye(4);E_mm=E.copy();E_mm[:3,3]*=40
        records.append({'a':a,'epsilon':eps,'t':float(t),'q':q.tolist(),
              'normalized_closure_frobenius':float(np.linalg.norm(E)),
              'l40mm_closure_frobenius':float(np.linalg.norm(E_mm)),
              'exact_status':'full loop identity and rank6 certified for every t'})
        if a==0 and eps==1 and t==0:
            for tau in np.linspace(0,2*np.pi,65):
                Q=np.array([theta,theta,0,tau,np.pi,2*theta-tau,np.pi])
                err=np.linalg.norm(numeric_loop(Q)-np.eye(4))
                path.append([float(tau),*Q.tolist(),float(err)])
(OUT/'folded_cycle_witnesses.json').write_text(json.dumps(records,indent=2),encoding='utf8')
with (OUT/'new_folded_cycle_path.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['t',*(f'q{i}' for i in range(1,8)),'normalized_closure_frobenius']);w.writerows(path)
print(json.dumps({'witness':records[6],'path_rows':len(path),
                  'path_max_residual':max(r[-1] for r in path)},indent=2))
