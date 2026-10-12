"""精确检查另一最大平面活动组{J3,J5,J7}，以及折叠四杆解。

通过全闭环排除“换一组活动轴即可在任意q3下产生平面运动”的
遗漏，而不是把q3=0/pi作为定义。
"""
from pathlib import Path
import sympy as s
from exact_model import q,symbolic_loop
OUT=Path(__file__).resolve().parent
lines=[]
for a in (0,1):
    for b in (0,1):
        v=list(q);v[3]=a*s.pi;v[5]=b*s.pi
        T=symbolic_loop(v).applyfunc(lambda v:s.trigsimp(s.expand_trig(v)))
        lines.extend([f'其他活动组：q4={a}pi,q6={b}pi',str(T),''])
(OUT/'other_planar_support.txt').write_text('\n'.join(lines),encoding='utf8')
print('\n'.join(lines))
