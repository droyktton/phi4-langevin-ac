"""Plots the domain-wall evolution of one phi4langevin run.

Reads walls_seed<S>_nseed<N>.dat (one block of wall crossings per output
time) and the matching timeseries_seed<S>_nseed<N>.dat, and writes
<prefix>_evolution.png with three panels:
  (a) the wall at selected times, colored by t;
  (b) R_eff(t) = sqrt(A+/pi) and the field h(t);
  (c) r(theta), the wall distance to the domain's center of mass, at the
      same selected times.

usage:
  python3 scripts/plot_wall_evolution.py walls_seed1_nseed2.dat [--ntimes 8] [--L 256]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEADER_RE = re.compile(r"(\S+)=\s*(\S+)")


def read_walls(path):
    """Returns a list of (header dict, points array (n,2)) per output time."""
    blocks = []
    header, pts = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                if header is not None:
                    blocks.append((header, np.array(pts).reshape(-1, 2)))
                header = {k: float(v) for k, v in HEADER_RE.findall(line)}
                pts = []
            elif line:
                pts.append([float(v) for v in line.split()])
    if header is not None:
        blocks.append((header, np.array(pts).reshape(-1, 2)))
    return blocks


def polar_profile(pts, xcm, ycm, L):
    """(theta, r) of the wall points around (xcm, ycm), minimum-image."""
    d = pts - np.array([xcm, ycm])
    d -= L * np.round(d / L)
    theta = np.arctan2(d[:, 1], d[:, 0])
    r = np.hypot(d[:, 0], d[:, 1])
    order = np.argsort(theta)
    return theta[order], r[order]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walls", help="walls_seed<S>_nseed<N>.dat")
    ap.add_argument("--ntimes", type=int, default=8, help="number of times to draw in panels (a) and (c)")
    ap.add_argument("--L", type=int, default=None, help="lattice size (default: read from logfile.dat next to the file)")
    args = ap.parse_args()

    walls_path = Path(args.walls)
    ts_path = walls_path.with_name(walls_path.name.replace("walls_", "timeseries_", 1))
    L = args.L
    if L is None:
        log = walls_path.with_name("logfile.dat")
        L = int(re.search(r"L=\s*(\d+)", log.read_text()).group(1))

    blocks = read_walls(walls_path)
    ts = np.loadtxt(ts_path, ndmin=2)
    pick = np.unique(np.linspace(0, len(blocks) - 1, min(args.ntimes, len(blocks))).astype(int))
    colors = plt.cm.viridis(np.linspace(0, 1, len(pick)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    for c, i in zip(colors, pick):
        head, pts = blocks[i]
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], ".", ms=1.5, color=c, label=f"t={head['t']:g}")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("(a) domain wall")
    ax.legend(fontsize=7, markerscale=5, loc="upper right")

    ax = axes[1]
    ax.plot(ts[:, 0], ts[:, 4], color="C0")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$R_{\rm eff}=\sqrt{A_+/\pi}$", color="C0")
    ax2 = ax.twinx()
    ax2.plot(ts[:, 0], ts[:, 1], color="C3", lw=0.8)
    ax2.set_ylabel("h(t)", color="C3")
    ax.set_title("(b) effective radius and field")

    ax = axes[2]
    for c, i in zip(colors, pick):
        head, pts = blocks[i]
        if len(pts):
            theta, r = polar_profile(pts, head["xcm"], head["ycm"], L)
            ax.plot(theta, r, ".", ms=1.5, color=c)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("r")
    ax.set_title(r"(c) $r(\theta)$ around the center of mass")

    fig.tight_layout()
    out = walls_path.with_name(walls_path.stem.replace("walls_", "", 1) + "_evolution.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
