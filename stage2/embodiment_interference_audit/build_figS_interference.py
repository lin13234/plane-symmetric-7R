"""Build FigS_embodiment_interference (English) from recorded audit outputs only."""
import json, os, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.patches import Rectangle
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "stix"
HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.join(HERE, "cad_frames2")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "FigS_embodiment_interference.pdf")
meta = json.load(open(os.path.join(FR, "meta.json")))
aud = json.load(open(os.path.join(HERE, "interference_audit.json")))
order = [["pos_G1", "pos_mid", "pos_G2"], ["neg_G3", "neg_mid", "neg_G4"]]
lab = {"pos_G1": r"$G_1$ (parallel, uncrossed)", "pos_mid": "intermediate", "pos_G2": r"$G_2$ (parallel, crossed)",
       "neg_G3": r"$G_3$ (parallel, uncrossed)", "neg_mid": "intermediate", "neg_G4": r"$G_4$ (parallel, crossed)"}
fig = plt.figure(figsize=(13, 13.2))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.62], hspace=0.22, wspace=0.04)
letters = iter("abcdef")
for ri, row in enumerate(order):
    for ci, key in enumerate(row):
        info = meta[key]; ax = fig.add_subplot(gs[ri, ci])
        img = mpimg.imread(os.path.join(FR, os.path.basename(info["file"]).replace("\\", "/").split("/")[-1]))
        g = img[..., :3].mean(axis=2)
        ax.imshow(g[30:990, 30:990], cmap="gray", vmin=0, vmax=1, extent=(30, 990, 990, 30))
        for j, (u, w) in info["labels"].items():
            if j in ("J1", "J2", "J4", "J6"):
                ax.plot(u, w, "o", color="k", ms=4)
                ax.annotate(j.replace("J", "$J_") + "$", (u, w), xytext=(6, -6), textcoords="offset points", fontsize=11)
        ax.set_xlim(30, 990); ax.set_ylim(990, 30); ax.set_xticks([]); ax.set_yticks([])
        cs = info["link_contacts"]
        note = "no link penetration" if not cs else f"max penetration {abs(min(c[2] for c in cs))*1000:.0f} mm"
        s = "+" if info["eta"] > 0 else "-"
        ax.set_title(f"({next(letters)}) $\\eta={s}$, $q_2={info['q2_deg']:.1f}^\\circ$: {lab[key]}", fontsize=11)
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=10, va="bottom", bbox=dict(fc="white", ec="k", lw=0.6))
ax = fig.add_subplot(gs[2, :])
rows = [("P_0_+", r"$\mathcal{P}_{0,+}$", "planar"), ("P_pi_+", r"$\mathcal{P}_{\pi,+}$", "planar"),
        ("P_0_-", r"$\mathcal{P}_{0,-}$", "planar"), ("P_pi_-", r"$\mathcal{P}_{\pi,-}$", "planar"),
        ("S1", r"$\mathcal{B}_{S1}$", "spatial"), ("S2", r"$\mathcal{B}_{S2}$", "spatial"),
        ("S3", r"$\mathcal{B}_{S3}$", "spatial"), ("S4", r"$\mathcal{B}_{S4}$", "spatial")]
for i, (k, name, kind) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.add_patch(Rectangle((0, y - 0.35), 360, 0.7, fc="0.55", ec="k", lw=0.6))
    rec = aud[kind][k]
    for a, b in (rec["free_q2_deg"] if kind == "planar" else rec["free_q3_deg"]):
        for lo, hi in ([(a, b)] if b <= 360 else [(a, 360), (0, b - 360)]):
            ax.add_patch(Rectangle((lo, y - 0.35), max(hi - lo, 0.8), 0.7, fc="white", ec="k", lw=0.6))
    ax.text(362, y, f"{rec['free_fraction']*100:.0f}% free", va="center", fontsize=10)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows][::-1], fontsize=11)
ax.set_xlim(0, 360); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_xticks(range(0, 361, 45))
ax.set_xlabel(r"cycle parameter (deg): $q_2$ for planar cycles, $q_3$ for spatial cycles (sampled every $2^\circ$)", fontsize=11)
ax.set_title("(g) Interference along each analytical functional cycle: white = no link-link penetration deeper than 1 mm, grey = penetration",
             fontsize=11, loc="left", pad=8)
ax.axvline(210.9, ymin=0.75, ymax=1.0, color="k", ls=":", lw=1.2)  # q2 of G2 / H2
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), dpi=110, bbox_inches="tight")
print(OUT)
