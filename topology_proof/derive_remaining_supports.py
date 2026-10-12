"""完整平面速度支持的其余最大候选：{1,2,5},{3,6},{4,7}。

相邻实际轴正交，J1/J2同向。闭合时J7固定为基座z轴，故所有
平面角速度支持落在六环独立集的五种最大支持内。本脚本检查
另外三种，避免只证明指定四杆活动组的局部结果。
"""
from pathlib import Path
import itertools
import sympy as s
from exact_model import q,axes_prefix,symbolic_loop
OUT=Path(__file__).resolve().parent
p,w=axes_prefix(q)
lines=[]
for i,j in [(0,4),(2,5),(3,6)]:
    dot=s.trigsimp(w[i].dot(w[j]))
    lines.append(f'J{i+1} dot J{j+1} = {dot}')
    if (i,j)==(2,5):subidx=(3,4)
    elif (i,j)==(3,6):subidx=(4,5)
    else:subidx=(2,3)
    for a,b in itertools.product((-1,1),repeat=2):
        subs={q[subidx[0]]:a*s.pi/2,q[subidx[1]]:b*s.pi/2}
        dist=(p[j]-p[i]).cross(w[i]).subs(subs).applyfunc(lambda v:s.trigsimp(s.expand_trig(v)))
        norm=s.trigsimp(dist.dot(dist))
        lines.append(f'quarter locks {subidx}: {a},{b}, squared line distance={norm}')
# {1,2,5}锁定的四个角只有离散quarter候选，输出每类剩余闭环。
for a,b,c,d in itertools.product((-1,1),repeat=4):
    angles=list(q)
    for index,value in zip((2,3,5,6),(a,b,c,d)):angles[index]=value*s.pi/2
    T=symbolic_loop(angles).applyfunc(lambda v:s.trigsimp(s.expand_trig(v)))
    lines.extend([f'支持125: q3,q4,q6,q7={a},{b},{c},{d} * pi/2',str(T),''])
(OUT/'remaining_planar_supports.txt').write_text('\n'.join(lines),encoding='utf8')
print('\n'.join(lines))
