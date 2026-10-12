"""当前冻结J1--J7 MDH的有理数实现，仅供本次证明，不改动原模型。

采用长度l=1；不读取或重新求解已有721点轨迹。关节轴取固定MDH
前缀Rx(alpha)Tx(a)之后的物理轴，避免把旧frame-z误当作真实轴。
"""
import sympy as s
import numpy as np

PARAMS=((0,1,2,0,0),(1,0,1,0,0),(0,1,2,s.Rational(5,2),0),
        (0,1,0,0,s.pi),(0,1,s.Rational(5,2),0,0),
        (0,1,s.Rational(5,2),0,0),(0,1,0,s.Rational(5,2),s.pi))
q=s.symbols('q1:8',real=True)

def matrix(ca,sa,a,d,ct,st):
    """用精确三角值直接构造已注册的MDH矩阵。"""
    return s.Matrix([[ct,-st,0,a],[st*ca,ct*ca,-sa,-d*sa],
                     [st*sa,ct*sa,ca,d*ca],[0,0,0,1]])

def angle_matrix(row,theta):
    """theta是参考增量；此处只施加一次表中的固定偏置。"""
    ca,sa,a,d,off=row
    return matrix(ca,sa,a,d,s.cos(theta+off),s.sin(theta+off))

def symbolic_loop(angles):
    """完整七角闭环；调用者必须显式说明是否施加额外关系。"""
    T=s.eye(4)
    for row,theta in zip(PARAMS,angles):T=T*angle_matrix(row,theta)
    return T

def axes_prefix(angles):
    """返回所有物理轴的点与方向；点沿轴移动不改变轴线。"""
    T=s.eye(4);points=[];axes=[]
    for row,theta in zip(PARAMS,angles):
        ca,sa,a,d,off=row
        F=T*matrix(ca,sa,a,0,1,0)
        points.append(F[:3,3]);axes.append(F[:3,2])
        T=T*angle_matrix(row,theta)
    return points,axes

def numeric_loop(angles):
    """仅对新的精确候选或反例计算残差；不重算冻结CSV。"""
    T=np.eye(4)
    for row,theta in zip(PARAMS,angles):
        ca,sa,a,d,off=[float(v) for v in row]
        ct,st=np.cos(theta+off),np.sin(theta+off)
        T=T@np.array([[ct,-st,0,a],[st*ca,ct*ca,-sa,-d*sa],
                     [st*sa,ct*sa,ca,d*ca],[0,0,0,1.]])
    return T
