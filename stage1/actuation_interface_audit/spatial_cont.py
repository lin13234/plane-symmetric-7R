import numpy as np
from assembly_fk import fk, planar
wrap = lambda x: (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi
G = {1: 0.0, 2: np.pi + 2 * np.arctan((5 - np.sqrt(5)) / 10), 3: 2 * np.arctan(5 / 4), 4: 2 * np.pi - 2 * np.arctan((5 - np.sqrt(5)) / 2)}
eta = {1: 1, 2: 1, 3: -1, 4: -1}

def qfull(x, q3):
    q1, q4, q5 = x
    return np.array([q1, q1, q3, q4, q5, -q4, -q3])

def res(x, q3):
    T, _, _ = fk(qfull(x, q3)); return (T - np.eye(4))[:3, :].ravel()

def solve(x, q3):
    for _ in range(60):
        f = res(x, q3)
        J = np.array([(res(x + h, q3) - f) / 1e-7 for h in np.eye(3) * 1e-7]).T
        dx = np.linalg.lstsq(J, -f, rcond=None)[0]
        x = x + dx
        if np.linalg.norm(dx) < 1e-13: break
    return x, np.abs(res(x, q3)).max()

def spatial_path(k, n=60):
    q0, _ = planar(G[k], eta[k], 0)
    print("G", k, "coord check q1-q2", wrap(q0[0] - q0[1]), "q4+q6", wrap(q0[3] + q0[5]), "q3", q0[2], "q7", q0[6])
    x = np.array([q0[0], q0[3], q0[4]]); out = [qfull(x, 0.0)]
    for q3 in np.linspace(0, np.pi, n + 1)[1:]:
        # predictor: tiny perturbation to leave planar branch at the first step
        x, e = solve(x + (np.array([0, 0, 0.05]) if len(out) == 1 else 0), q3)
        out.append(qfull(x, q3))
    return np.array(out), e

if __name__ == "__main__":
    for k in (1, 3):
        path, e = spatial_path(k)
        qH, _ = planar(G[k], eta[k], 1)
        print(k, "final residual", e, "end q", np.round(wrap(path[-1]), 4))
        print("   q5 range", np.round(path[:, 4].min(), 3), np.round(path[:, 4].max(), 3))
