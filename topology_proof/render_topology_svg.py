"""用现有Matplotlib渲染本次SVG为PNG，便于直接预览。

只处理build_topology_graph.py生成的矩形、圆、二次曲线及文字。
保留同一组节点/多重边几何，不修改原Fig.8，不读取轨迹。
"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch,Circle,Rectangle
OUT=Path(__file__).resolve().parent
root=ET.parse(OUT/'topology_graph.svg').getroot()
fig,ax=plt.subplots(figsize=(12.4,7.4),dpi=150)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.set_xlim(0,1240);ax.set_ylim(740,0);ax.set_aspect('equal');ax.axis('off')
for elem in root:
 tag=elem.tag.split('}')[-1];a=elem.attrib
 if tag=='rect':ax.add_patch(Rectangle((0,0),(1240),740,color=a['fill']))
 elif tag=='circle':ax.add_patch(Circle((float(a['cx']),float(a['cy'])),float(a['r']),
                       fill=a.get('fill')!='none',facecolor=a.get('fill') if a.get('fill')!='none' else 'none',
                       edgecolor=a.get('stroke',a.get('fill')),linewidth=float(a.get('stroke-width',0))))
 elif tag=='path':
  tokens=re.findall(r'[MQ]|-?\d+(?:\.\d+)?',a['d']);verts=[];codes=[];j=0
  while j<len(tokens):
   command=tokens[j];j+=1
   if command=='M':
    verts.append((float(tokens[j]),float(tokens[j+1])));codes.append(PlotPath.MOVETO);j+=2
   elif command=='Q':
    for _ in range(2):verts.append((float(tokens[j]),float(tokens[j+1])));codes.append(PlotPath.CURVE3);j+=2
  ax.add_patch(PathPatch(PlotPath(verts,codes),fill=False,edgecolor=a['stroke'],linewidth=float(a['stroke-width'])*.72))
 elif tag=='text':
  cls=a.get('class','note');px={'title':25,'label':19,'note':16}[cls]
  ax.text(float(a['x']),float(a['y']),elem.text,fontsize=px*.72,
          color='#182433',fontfamily='DejaVu Sans',fontweight='bold' if cls=='title' else 'normal',va='baseline')
fig.savefig(OUT/'topology_graph.png',dpi=150,facecolor='#fafbfc')
plt.close(fig)
print('topology_graph.png generated from topology_graph.svg')
