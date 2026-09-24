"""Extracts the phi=0 level curves (domain walls) of phi4langevin field
snapshots as ordered, closed polylines, taking the periodic boundaries
into account.

Input: the config_t<t>_seed<S>_nseed<N>.dat snapshots written with
--tconf > 0 (L rows of L values, row index = y, column index = x).

For each snapshot the field is (optionally) smoothed with the same
5-point average used by the CUDA wall detector, and its zero level set
is traced with contourpy (marching squares, linear interpolation). Pieces
that leave the box through one side are stitched to the piece that
re-enters through the opposite side, so every wall comes out as a single
curve in unwrapped coordinates. If the domain splits, there are several
curves per time. A curve that wraps around the periodic box (e.g. a
stripe spanning the sample) has a nonzero winding number (wx, wy) and is
not closed in unwrapped coordinates.

Output (in the snapshots' directory, <run> = seed<S>_nseed<N>):
  contours_<run>.dat  one gnuplot "index" block per time,
      # t= .. ncurves= ..
      then per curve a header
      # curve= k npts= .. length= .. area= .. inside= +1|-1 wx= .. wy= .. xc= .. yc=
      followed by its ordered "x y" points (unwrapped, first point
      repeated at the end for closed curves); curves are separated by
      one blank line and times by two.
  contours_<run>.csv  one row per (time, curve) with the same summary.

Per curve: length is the polyline length, area the enclosed area (for
closed curves), inside the sign of phi inside the curve (+1: a piece of
the initial + domain), (xc, yc) the centroid folded back into [0, L).

usage:
  python3 scripts/extract_contours.py <dir> [--run seed1_nseed2] [--no-smooth] [--min-length 0]
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
from contourpy import contour_generator, LineType

SNAP_RE = re.compile(r"config_t([-\d.eE+]+)_(seed\d+_nseed\d+)\.dat$")


def read_snapshot(path):
    return np.loadtxt(path, dtype=np.float32)


def smooth5(phi):
    """5-point (self + 4 neighbors) periodic sum, as in wall_crossingop."""
    return (phi + np.roll(phi, 1, 0) + np.roll(phi, -1, 0)
            + np.roll(phi, 1, 1) + np.roll(phi, -1, 1))


def _key(p, L):
    return (round(float(p[0]) % L, 5), round(float(p[1]) % L, 5))


def periodic_contours(field):
    """Zero level curves of a periodic LxL field (field[y, x]).

    Returns a list of (points, winding) with points an (n, 2) array of
    unwrapped (x, y) coordinates and winding = (wx, wy) the number of
    times the curve wraps around the box in each direction.
    """
    L = field.shape[0]
    z = np.pad(field, ((0, 1), (0, 1)), mode="wrap")  # adds the x=L, y=L images of x=0, y=0
    grid = np.arange(L + 1, dtype=np.float64)
    gen = contour_generator(grid, grid, z, line_type=LineType.Separate, corner_mask=False)
    lines = gen.lines(0.0)

    curves = []
    open_pieces = []
    for ln in lines:
        if len(ln) > 2 and np.allclose(ln[0], ln[-1]):
            curves.append((ln, (0, 0)))
        elif len(ln) >= 2:
            open_pieces.append(ln)

    # stitch open pieces: every piece ends on the box border and continues
    # as the piece that starts at the periodic image of that point
    by_start = {}
    for i, ln in enumerate(open_pieces):
        by_start.setdefault(_key(ln[0], L), []).append(i)
    used = np.zeros(len(open_pieces), dtype=bool)

    for i0 in range(len(open_pieces)):
        if used[i0]:
            continue
        used[i0] = True
        chain = [open_pieces[i0]]
        start_key = _key(open_pieces[i0][0], L)
        while True:
            end = chain[-1][-1]
            k = _key(end, L)
            if k == start_key:
                break  # back to the starting point (mod L): curve closed
            nxt = [j for j in by_start.get(k, []) if not used[j]]
            if not nxt:
                break  # should not happen; keep the curve open
            j = nxt[0]
            used[j] = True
            piece = open_pieces[j]
            shift = np.round((end - piece[0]) / L) * L  # translate to continue the unwrapped curve
            chain.append(piece[1:] + shift)
        pts = np.vstack(chain)
        disp = pts[-1] - pts[0]
        winding = (int(round(disp[0] / L)), int(round(disp[1] / L)))
        if winding == (0, 0):
            pts[-1] = pts[0]  # exact closure
        curves.append((pts, winding))
    return curves


def curve_stats(pts, winding, field):
    """length, area, inside sign and folded centroid of one curve."""
    L = field.shape[0]
    seg = np.diff(pts, axis=0)
    length = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
    closed = winding == (0, 0)
    x, y = pts[:, 0], pts[:, 1]
    signed_area = 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])) if closed else 0.0

    # sign of the field on the left of the curve, by majority over segments
    mid = 0.5 * (pts[:-1] + pts[1:])
    norm = np.hypot(seg[:, 0], seg[:, 1])
    ok = norm > 0
    left = np.column_stack([-seg[ok, 1], seg[ok, 0]]) / norm[ok, None]
    probe = np.rint(mid[ok] + 0.75 * left).astype(int) % L
    left_sign = np.sign(np.sum(np.sign(field[probe[:, 1], probe[:, 0]])))
    # the interior is on the left for a counter-clockwise curve
    inside = int(left_sign if signed_area >= 0 else -left_sign) if closed else 0

    xc, yc = pts[:-1].mean(axis=0) % L
    return length, abs(signed_area), inside, float(xc), float(yc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="directory with config_t*_seed*_nseed*.dat snapshots")
    ap.add_argument("--run", default=None, help="run tag seed<S>_nseed<N> (default: the only one present)")
    ap.add_argument("--no-smooth", action="store_true", help="trace the raw field instead of the 5-point smoothed one")
    ap.add_argument("--min-length", type=float, default=0.0,
                    help="drop curves shorter than this (e.g. thermally nucleated micro-loops)")
    args = ap.parse_args()

    d = Path(args.dir)
    snaps = {}
    for p in d.glob("config_t*_seed*_nseed*.dat"):
        m = SNAP_RE.search(p.name)
        if m:
            snaps.setdefault(m.group(2), []).append((float(m.group(1)), p))
    if not snaps:
        raise SystemExit(f"no config_t*_seed*_nseed*.dat snapshots in {d} (run with --tconf > 0)")
    run = args.run or (list(snaps)[0] if len(snaps) == 1 else None)
    if run is None:
        raise SystemExit(f"several runs present, choose one with --run: {sorted(snaps)}")
    files = sorted(snaps[run])

    dat_path = d / f"contours_{run}.dat"
    csv_path = d / f"contours_{run}.csv"
    with open(dat_path, "w") as fdat, open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["t", "curve", "npts", "length", "area", "inside", "wx", "wy", "xc", "yc"])
        for t, path in files:
            phi = read_snapshot(path)
            field = phi if args.no_smooth else smooth5(phi)
            curves = []
            for pts, wind in periodic_contours(field):
                length, area, inside, xc, yc = curve_stats(pts, wind, field)
                if length >= args.min_length:
                    curves.append((pts, wind, length, area, inside, xc, yc))
            curves.sort(key=lambda c: -c[2])  # longest first

            fdat.write(f"# t= {t:g} ncurves= {len(curves)}\n")
            for k, (pts, wind, length, area, inside, xc, yc) in enumerate(curves):
                fdat.write(f"# curve= {k} npts= {len(pts)} length= {length:.4f} area= {area:.4f} "
                           f"inside= {inside:+d} wx= {wind[0]} wy= {wind[1]} xc= {xc:.4f} yc= {yc:.4f}\n")
                np.savetxt(fdat, pts, fmt="%.4f")
                fdat.write("\n")
                w.writerow([f"{t:g}", k, len(pts), f"{length:.4f}", f"{area:.4f}", inside,
                            wind[0], wind[1], f"{xc:.4f}", f"{yc:.4f}"])
            fdat.write("\n")
            print(f"t={t:g}: {len(curves)} curve(s)")
    print(f"wrote {dat_path} and {csv_path}")


if __name__ == "__main__":
    main()
