"""先推导真实活动轴的平行条件，再枚举八个0/pi候选切片。

八个角切片是轴条件的推论，不预先作为全部平面解的定义。
保存完整闭环矩阵各项；不调用已有轨迹求解器。
"""
from pathlib import Path
import json,itertools
import sympy as s
from exact_model import q,axes_prefix,symbolic_loop
OUT=Path(__file__).resolve().parent
points,axes=axes_prefix(q)
normal=axes[0]
cross4=normal.cross(axes[3]).applyfunc(s.trigsimp)
dot4=s.trigsimp(normal.dot(axes[3]))
records={'axis1':str(normal),'axis2':str(axes[1]),'axis1_cross_axis4':[str(v) for v in cross4],
         'axis1_dot_axis4':str(dot4),'axis6_after_q3_binary':{}}
for a in (0,1):
    cross6=normal.cross(axes[5]).subs(q[2],a*s.pi).applyfunc(s.trigsimp)
    records['axis6_after_q3_binary'][str(a)]=[str(v) for v in cross6]
records['slices']=[]
dump=['完整MDH平面轴条件及八切片（精确）','']
for a,b,c in itertools.product((0,1),repeat=3):
    angles=list(q);angles[2]=a*s.pi;angles[4]=b*s.pi;angles[6]=c*s.pi
    T=symbolic_loop(angles).applyfunc(lambda v:s.trigsimp(s.expand_trig(v)))
    p,ax=axes_prefix(angles)
    active=[0,1,3,5]
    # 公共法向取J1；同轴线的不同点可投影到固定平面。
    axial_heights=[s.trigsimp((p[i]-p[0]).dot(normal)) for i in active]
    entry={'a_q3':a,'b_q5':b,'c_q7':c,'loop':[[str(v) for v in T.row(i)] for i in range(4)],
           'active_axis_heights':list(map(str,axial_heights))}
    records['slices'].append(entry)
    dump.extend([f'切片 q3={a}pi q5={b}pi q7={c}pi',str(T),f'轴上点相对高度={axial_heights}',''])
(OUT/'planar_axes_and_slices.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
(OUT/'planar_axes_and_slices.txt').write_text('\n'.join(dump),encoding='utf8')
print(json.dumps({k:v for k,v in records.items() if k!='slices'},ensure_ascii=False,indent=2))
print('\n'.join(dump))
