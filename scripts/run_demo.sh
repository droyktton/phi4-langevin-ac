#!/bin/bash
# Demo run: a circular + domain (L=512, R=150) in strong disorder
# (Delta=1) under an AC field (h0=0.3, f=0.004, i.e. period 250) at T=0.02,
# for three field periods. The wall roughens and follows the field; at this
# strong disorder, + droplets nucleate outside the domain when h>0 and -
# holes inside it when h<0, so the phi=0 level set has several curves.
#
# Writes everything to demo/ (or the directory given as first argument):
#   walls_*, timeseries_*        wall crossings / observables every tout=5
#   config_t*_*.dat              full field every tconf=10 (~2.4 MB each)
#   contours_*.dat / .csv        phi=0 level curves as ordered closed polylines
#   *_evolution.png, contours_*_grid.png, contours_*_ncurves.png, contours_*.gif
#
# usage: scripts/run_demo.sh [outdir]
set -e
here=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-$here/demo}

make -C "$here" -s
mkdir -p "$out"
cd "$out"
rm -f walls_*.dat timeseries_*.dat config_t*.dat contours_*

"$here/phi4langevin" --L 512 --radius 150 --delta 1.0 --T 0.02 \
    --h0 0.3 --freq 0.004 --dt 0.05 --tmax 750 --tout 5 --tconf 10 \
    --seed 1 --noise-seed 2

python3 "$here/scripts/extract_contours.py" . --min-length 8
python3 "$here/scripts/plot_wall_evolution.py" walls_seed1_nseed2.dat --ntimes 8
python3 "$here/scripts/plot_contours.py" contours_seed1_nseed2.dat --panels 16 --gif
