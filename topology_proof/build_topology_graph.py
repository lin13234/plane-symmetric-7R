"""输出七关节空间的incidence graph，而非仅(q2,q3)投影。

平面圆和空间圆分别在两个交点处分成两条弧；折叠圆单独表示。
JSON保留多重边，防止常规简单图把两条弧压成一条而误计循环。
"""
from pathlib import Path
import json
OUT=Path(__file__).resolve().parent
nodes=[];edges=[];circles=[]
for eta,ks in [('+',(1,2)),('-',(3,4))]:
 for k in ks:
  for side in 'GH':nodes.append({'id':f'{side}{k}','component':f'K{eta}','rank':5})
 for a,side in [(0,'G'),(1,'H')]:
  cid=f'P{a}{eta}';circles.append({'id':cid,'type':'planar','nodes':[f'{side}{k}' for k in ks]})
  for arc in (0,1):edges.append({'id':f'{cid}_arc{arc}','from':f'{side}{ks[0]}','to':f'{side}{ks[1]}','circle':cid,'type':'planar'})
 for k in ks:
  cid=f'S{k}';circles.append({'id':cid,'type':'spatial','nodes':[f'G{k}',f'H{k}']})
  for arc in (0,1):edges.append({'id':f'{cid}_arc{arc}','from':f'G{k}','to':f'H{k}','circle':cid,'type':'spatial'})
for a in (0,1):
 for eps in (-1,1):
  cid=f'F{a}{eps:+d}';nodes.append({'id':cid,'component':cid,'rank':6,'artificial_graph_vertex':True})
  edges.append({'id':cid,'from':cid,'to':cid,'circle':cid,'type':'folded_planar'})
  circles.append({'id':cid,'type':'folded_planar','nodes':[]})
graph={'evidence':'EXACT PROOF','space':'T^7; no collision or joint limits',
       'nodes':nodes,'edges':edges,'analytic_circles':circles,
       'counts':{'connected_components':6,'planar_circles':8,'spatial_circles':4,
                 'vertices':len(nodes),'edges':len(edges),'first_betti_number':len(edges)-len(nodes)+6}}
(OUT/'topology_graph.json').write_text(json.dumps(graph,indent=2),encoding='utf8')
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="740" viewBox="0 0 1240 740">',
 '<rect width="1240" height="740" fill="#fafbfc"/>',
 '<style>text{font-family:Arial,sans-serif;fill:#182433}.title{font-size:25px;font-weight:700}.label{font-size:19px}.note{font-size:16px}</style>',
 '<text x="48" y="46" class="title">Complete 7R joint-space incidence graph</text>',
 '<text x="48" y="77" class="note">Exact frozen MDH model; angles modulo 2pi; parallel edges are distinct arcs.</text>']
for offset,eta,ks in [(0,'+',(1,2)),(580,'-',(3,4))]:
 x1,x2=140+offset,440+offset;y1,y2=175,365
 svg +=[f'<text x="{x1+65}" y="116" class="label">K{eta}: two planar + two spatial circles</text>']
 # 每条横边用上下两弧组成一个平面闭合圆。
 for yy,a in [(y1,0),(y2,1)]:
  svg +=[f'<path d="M{x1},{yy} Q{x1+150},{yy-65} {x2},{yy} Q{x1+150},{yy+65} {x1},{yy}" fill="none" stroke="#23926b" stroke-width="3"/>',
          f'<text x="{x1+120}" y="{yy-42}" class="label">P{a}{eta}</text>']
 # 每条竖边用左右两弧组成一个空间闭合圆。
 for xx,k in [(x1,ks[0]),(x2,ks[1])]:
  svg +=[f'<path d="M{xx},{y1} Q{xx-62},{(y1+y2)/2} {xx},{y2} Q{xx+62},{(y1+y2)/2} {xx},{y1}" fill="none" stroke="#286ed0" stroke-width="3"/>',
          f'<text x="{xx-80}" y="{(y1+y2)/2+5}" class="label">S{k}</text>']
 for xx,k in [(x1,ks[0]),(x2,ks[1])]:
  for yy,side in [(y1,'G'),(y2,'H')]:
   svg +=[f'<circle cx="{xx}" cy="{yy}" r="7" fill="#182433"/>',
           f'<text x="{xx+12}" y="{yy+7}" class="label">{side}{k}</text>']
 svg +=[f'<text x="{x1+25}" y="436" class="note">V=4, E=8, cycle rank=5</text>']
svg +=['<text x="48" y="493" class="label">Four additional detached planar circles: q5=pi (coincident J4/J6 axes)</text>']
for i,(a,eps) in enumerate([(0,-1),(0,1),(1,-1),(1,1)]):
 x=160+290*i
 svg +=[f'<circle cx="{x}" cy="573" r="39" fill="none" stroke="#a35b12" stroke-width="3"/>',
         f'<text x="{x-34}" y="640" class="label">F{a},{eps:+d}</text>']
svg +=['<text x="48" y="695" class="note">8 planar circles; 4 spatial circles; 6 connected components; first Betti number = 14.</text>',
 '<text x="48" y="721" class="note">G and H lie on different planar circles. They connect through spatial arcs; BP signs are assembly labels.</text>','</svg>']
(OUT/'topology_graph.svg').write_text('\n'.join(svg),encoding='utf8')
print(json.dumps(graph['counts'],indent=2))
