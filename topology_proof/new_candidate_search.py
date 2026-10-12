"""在完整七角闭环中做确定性候选发现，不称为穷尽性证明。

固定新的q3=7/10，使用Halton初值和完整12分量闭环优化。
只为发现反例或为区间认证提供候选盒；不读取旧721点轨迹。
"""
from pathlib import Path
import json,csv
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc
from exact_model import PARAMS,numeric_loop
OUT=Path(__file__).resolve().parent
PASSIVE=[0,1,3,4,5,6]
ROWS=[(float(ca),float(sa),float(a),float(d),int(off!=0)) for ca,sa,a,d,off in PARAMS]
def evaluate(p,x):
    q=np.zeros(7);q[PASSIVE]=p;q[2]=x
    A=[];DA=[]
    for theta,(ca,sa,a,d,off) in zip(q,ROWS):
        c,st=np.cos(theta),np.sin(theta)
        if off:c,st=-c,-st
        A.append(np.array([[c,-st,0,a],[st*ca,c*ca,-sa,-d*sa],[st*sa,c*sa,ca,d*ca],[0,0,0,1.]]))
        DA.append(np.array([[-st,-c,0,0],[c*ca,-st*ca,0,0],[c*sa,-st*sa,0,0],[0,0,0,0.]]))
    pre=[np.eye(4)]
    for v in A:pre.append(pre[-1]@v)
    suf=[np.eye(4)]*8
    for i in reversed(range(7)):suf[i]=A[i]@suf[i+1]
    E=pre[-1][:3,:]-np.eye(4)[:3,:]
    J=np.column_stack([(pre[i]@DA[i]@suf[i+1])[:3,:].ravel() for i in PASSIVE])
    return E.ravel(),J,q
def wrap(a):return np.arctan2(np.sin(a),np.cos(a))

x=0.7
seeds=(qmc.Halton(6,scramble=False).random(192)*2-1)*np.pi
roots=[];success=0
for index,seed in enumerate(seeds):
    sol=least_squares(lambda v:evaluate(v,x)[0],seed,jac=lambda v:evaluate(v,x)[1],
                      xtol=2e-13,ftol=2e-13,gtol=2e-13,max_nfev=300)
    f,j,q=evaluate(sol.x,x);error=np.linalg.norm(f)
    if error>1e-9:continue
    success+=1;q=wrap(q)
    if any(np.linalg.norm(wrap(q-r['q']))<1e-5 for r in roots):continue
    relation=wrap([q[0]-q[1],q[3]+q[5],q[2]+q[6]])
    roots.append({'q':q.tolist(),'normalized_closure_frobenius':float(error),
                  'coordination_errors':relation.tolist(),'seed_index':index,
                  'min_passive_singular_value':float(np.linalg.svd(j,compute_uv=False)[-1]),
                  'evidence_class':'SAMPLED EVIDENCE; candidate only until interval certificate'})
    print(f'new root {len(roots)}: residual={error:.2e}, coordination={relation}',flush=True)
result={'input_q3':x,'deterministic_starts':len(seeds),'converged_starts':success,
        'distinct_discovered_roots':len(roots),'roots':roots,
        'is_exhaustive':False,'global_exhaustion_claim':'UNPROVED'}
(OUT/'new_full7r_candidates.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
