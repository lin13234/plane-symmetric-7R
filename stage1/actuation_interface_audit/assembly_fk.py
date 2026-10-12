import numpy as np
r = np.sqrt(41) / 2
MDH = [  # alpha, a, d, theta0
    (np.pi/2, 2, 0, 0), (0, 1, 0, 0), (np.pi/2, 2, 2.5, 0), (np.pi/2, 0, 0, np.pi),
    (np.pi/2, 2.5, 0, 0), (np.pi/2, 2.5, 0, 0), (np.pi/2, 0, 2.5, np.pi)]

def Rx(a):
    c, s = np.cos(a), np.sin(a); M = np.eye(4); M[1:3, 1:3] = [[c, -s], [s, c]]; return M
def Rz(a):
    c, s = np.cos(a), np.sin(a); M = np.eye(4); M[0:2, 0:2] = [[c, -s], [s, c]]; return M
def Tx(x): M = np.eye(4); M[0, 3] = x; return M
def Tz(z): M = np.eye(4); M[2, 3] = z; return M

def fk(q):
    T = np.eye(4); pts = []; axes = []
    for (al, a, d, th0), qi in zip(MDH, q):
        F = T @ Rx(al) @ Tx(a)
        pts.append(F[:3, 3].copy()); axes.append(F[:3, 2].copy())
        T = F @ Rz(qi + th0) @ Tz(d)
        pts.append(T[:3, 3].copy())
    return T, np.array(pts), np.array(axes)

def planar(t, eta, side):
    P4 = np.array([1, 0]) + np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) @ np.array([2, -2.5])
    d = np.linalg.norm(P4); e = -P4 / d; ep = np.array([-e[1], e[0]])
    k = (25 - r**2 + d**2) / (2 * d); H = np.sqrt(25 - k**2)
    P6 = P4 + k * e + eta * H * ep
    A = np.arctan2(*(P6 - P4)[::-1]) - np.pi
    B = np.arctan2(*(-P6)[::-1]) - np.arctan2(2.5, 2)
    if side == 0:
        q = [-B, t, 0, t - A, 0, B - A, 0]
    else:
        q = [-B, t, np.pi, np.pi - t + A, 0, np.pi - B + A, np.pi]
    return np.array(q), dict(P1=np.zeros(2), P2=np.array([1., 0]), P4=P4, P6=P6)

def folded(t, a, eps):
    c = (-4 + 10 * eps * np.sqrt(10)) / 41; s = (-5 - 8 * eps * np.sqrt(10)) / 41
    th = np.arctan2(s, c)
    return np.array([th, th, a * np.pi, t, np.pi, (-1) ** a * 2 * th - t, (a + 1) * np.pi])

if __name__ == "__main__":
    for t in [0.3, 1.5, 4.0]:
        for eta in (1, -1):
            for side in (0, 1):
                q, _ = planar(t, eta, side)
                T, _, _ = fk(q)
                print("P", t, eta, side, np.abs(T - np.eye(4)).max())
        for a in (0, 1):
            for eps in (1, -1):
                T, _, _ = fk(folded(t, a, eps))
                print("F", t, a, eps, np.abs(T - np.eye(4)).max())
