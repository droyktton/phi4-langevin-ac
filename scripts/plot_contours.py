"""Plots the phi=0 level curves written by extract_contours.py.

Writes, next to contours_<run>.dat:
  contours_<run>_grid.png   a grid of panels at selected times: the field
                            snapshot in grey with each closed wall drawn in
                            its own color (more than one curve when the
                            domain splits or droplets/holes nucleate);
  contours_<run>_ncurves.png  number of curves, total wall length and
                            enclosed + area vs t, with the field h(t);
  contours_<run>.gif        (with --gif) an animation over all snapshots.

usage:
  python3 scripts/plot_contours.py demo/contours_seed1_nseed2.dat [--panels 12] [--gif]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEADER_RE = re.compile(r"(\w+)=\s*(\S+)")


def read_contours(path):
    """Returns a list of (t, [(header dict, points (n,2)), ...]) per time."""
    frames = []
    curve = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("# t="):
                frames.append((float(HEADER_RE.findall(s)[0][1]), []))
            elif s.startswith("# curve="):
                curve = ({k: float(v) for k, v in HEADER_RE.findall(s)}, [])
                frames[-1][1].append(curve)
            elif s:
                curve[1].append([float(v) for v in s.split()])
    return [(t, [(h, np.array(p)) for h, p in curves]) for t, curves in frames]


def field_at(t, L, log_text):
    h0 = float(re.search(r"h0=\s*(\S+)", log_text).group(1))
    f = float(re.search(r"freq=\s*(\S+)", log_text).group(1))
    return h0 * np.cos(2 * np.pi * f * np.asarray(t))


def draw_frame(ax, t, curves, snapshot, L, h=None):
    ax.clear()
    if snapshot is not None and snapshot.exists():
        phi = np.loadtxt(snapshot, dtype=np.float32)
        ax.imshow(phi, origin="lower", cmap="Greys_r", vmin=-1.3, vmax=1.3,
                  extent=(-0.5, L - 0.5, -0.5, L - 0.5), interpolation="nearest")
    colors = plt.cm.Set1(np.arange(max(len(curves), 1)) % 9)
    for c, (head, pts) in zip(colors, curves):
        # the curve is in unwrapped coordinates: draw it and its periodic
        # images, the axes limits clip everything to the box
        for sx in (-L, 0, L):
            for sy in (-L, 0, L):
                ax.plot(pts[:, 0] + sx, pts[:, 1] + sy, "-", color=c, lw=2.0)
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-0.5, L - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    title = f"t={t:g}  curves={len(curves)}"
    if h is not None:
        title += f"  h={h:+.2f}"
    ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("contours", help="contours_<run>.dat from extract_contours.py")
    ap.add_argument("--panels", type=int, default=12, help="number of times shown in the grid")
    ap.add_argument("--gif", action="store_true", help="also write an animated GIF over all snapshots")
    args = ap.parse_args()

    path = Path(args.contours)
    d = path.parent
    run = path.stem.replace("contours_", "", 1)
    log_text = (d / "logfile.dat").read_text()
    L = int(re.search(r"L=\s*(\d+)", log_text).group(1))
    frames = read_contours(path)

    def snapshot(t):
        return d / f"config_t{t:g}_{run}.dat"

    # grid of selected times
    pick = np.unique(np.linspace(0, len(frames) - 1, min(args.panels, len(frames))).astype(int))
    ncols = 4
    nrows = int(np.ceil(len(pick) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.3 * nrows), squeeze=False)
    for ax in axes.flat[len(pick):]:
        ax.axis("off")
    for ax, i in zip(axes.flat, pick):
        t, curves = frames[i]
        draw_frame(ax, t, curves, snapshot(t), L, field_at(t, L, log_text))
    fig.tight_layout()
    out = d / f"contours_{run}_grid.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")

    # global observables of the level set vs t
    t = np.array([f[0] for f in frames])
    ncurves = np.array([len(f[1]) for f in frames])
    length = np.array([sum(h["length"] for h, _ in f[1]) for f in frames])
    area_plus = np.array([sum(h["area"] * (h["inside"] > 0) - h["area"] * (h["inside"] < 0) for h, _ in f[1])
                          for f in frames])
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for ax, y, lab in zip(axes, (ncurves, length, area_plus),
                          ("number of curves", "total wall length", "enclosed + area")):
        ax.plot(t, y, "o-", ms=3)
        ax.set_ylabel(lab)
        ax2 = ax.twinx()
        ax2.plot(t, field_at(t, L, log_text), color="C3", lw=0.8, alpha=0.6)
        ax2.set_ylabel("h(t)", color="C3")
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    out = d / f"contours_{run}_ncurves.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")

    if args.gif:
        from matplotlib.animation import FuncAnimation, PillowWriter
        fig, ax = plt.subplots(figsize=(5, 5.2))

        def update(i):
            t, curves = frames[i]
            draw_frame(ax, t, curves, snapshot(t), L, field_at(t, L, log_text))

        anim = FuncAnimation(fig, update, frames=len(frames))
        out = d / f"contours_{run}.gif"
        anim.save(out, writer=PillowWriter(fps=6), dpi=80)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
