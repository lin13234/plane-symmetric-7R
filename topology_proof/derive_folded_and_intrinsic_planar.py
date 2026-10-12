"""核查q5=pi折叠切片，以及J3/J5/J7活动组的真实轴条件。

不预设折叠切片应当排除。先保存精确变换，再从闭环判断是否
存在固定q3和旋转q3的有限平面运动。
"""
from pathlib import Path
import sympy as s
from exact_model import q,symbolic_loop,axes_prefix
OUT=Path(__file__).resolve().parent
p,w=axes_prefix(q)
lines=[]
for pair in [(2,4),(4,6)]:
    dot=s.trigsimp(w[pair[0]].dot(w[pair[1]]))
    lines.append(f'J{pair[0]+1} dot J{pair[1]+1} = {dot}')
for a in (0,1):
    for b in (0,1):
        angles=list(q);angles[3]=a*s.pi;angles[4]=s.pi;angles[5]=b*s.pi
        T=symbolic_loop(angles).applyfunc(lambda v:s.trigsimp(s.expand_trig(v)))
        points,axes=axes_prefix(angles)
        coincidence=(points[6]-points[2]).cross(axes[2]).applyfunc(s.trigsimp)
        lines.extend([f'q4={a}pi q5=pi q6={b}pi',str(T),f'J3/J7共线矩={coincidence}',''])
(OUT/'folded_intrinsic_slices.txt').write_text('\n'.join(lines),encoding='utf8')
print('\n'.join(lines))
