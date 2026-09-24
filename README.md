# phi4-langevin-ac

GPU (CUDA/Thrust) simulation of the **Langevin dynamics of a circular
domain** in the disordered phi^4 model. The domain is driven by an
**AC magnetic field** at finite temperature, on a lattice with periodic
boundary conditions. The code tracks the domain wall, i.e. the `phi = 0`
level set, as a function of time.

![demo: phi field (grey) and phi=0 level curves at different times](docs/demo_grid.png)

The model is the same as in A. B. Kolton, E. E. Ferrero, and A. Rosso,
"Depinning free of the elastic approximation," Phys. Rev. B **108**,
174201 (2023) ([arXiv:2306.13415](https://arxiv.org/abs/2306.13415));
see [vmc-phi4-depinning](https://github.com/droyktton/vmc-phi4-depinning).
The problem is different: this code does not study the quasistatic
depinning transition. It follows the real-time stochastic dynamics of a
closed domain wall under an oscillating drive.

## Contents

- [Quick start](#quick-start)
- [Model](#model)
- [Numerical method](#numerical-method)
- [Build](#build)
- [Run](#run)
- [Output files](#output-files)
- [Level curves phi=0](#level-curves-phi0)
- [Demo](#demo)
- [Checks](#checks)
- [Repository layout](#repository-layout)
- [Citing](#citing)
- [License](#license)

## Quick start

```
make ARCH=sm_86                  # use your GPU's compute capability
./phi4langevin --L 512 --radius 100 --delta 0.2 --T 0.01 \
    --h0 0.3 --freq 0.002 --dt 0.05 --tmax 1500 --tout 25 --tconf 50
python3 scripts/plot_wall_evolution.py walls_seed1_nseed2.dat
python3 scripts/extract_contours.py .
python3 scripts/plot_contours.py contours_seed1_nseed2.dat --gif
```

or just `scripts/run_demo.sh` (see [Demo](#demo)).

## Model

```
d phi/dt = c * Laplacian(phi) + eps0 * [ r0 * (1 + r(x,y)) * phi - phi^3 ] + h(t) + eta(x,y,t)

h(t) = h0 * cos(2 pi f t)
<eta(x,t) eta(x',t')> = 2 T delta(x - x') delta(t - t')
```

- `L x L` square lattice, **periodic** in both `x` and `y`, with the
  5-point discrete Laplacian.
- `r(x,y)`: quenched, uncorrelated random-bond disorder, uniform in
  `[-Delta, Delta]`. It is generated once from a counter-based RNG
  (Philox, `Random123/`) keyed by site and disorder seed. The layout is
  the same as in `vmc-phi4-depinning`, so the same seed gives the same
  disorder realization in both codes.
- `h(t)`: AC field with amplitude `h0` and frequency `f`. It is a cosine,
  so `f = 0` gives a constant (DC) field `h0`.
- `eta`: Gaussian white thermal noise at temperature `T`.
- `c`, `eps0`, `r0` are compile-time constants (`CEL`, `EPSILON0`, `R0`,
  default 1).

**Initial condition:** a disk of `phi = +1` with radius `R`, centered at
`(L/2, L/2)`, in a `phi = -1` background.

## Numerical method

The equation is integrated with the explicit **Euler–Maruyama** scheme,

```
phi_new = phi + dt * F(phi, h(t)) + sqrt(2 T dt) * xi,     xi ~ N(0,1)
```

- **Parallel update:** it is a Jacobi update with two buffers, so all
  sites update in parallel on the GPU (`eulerop` in `langevin.h`). The
  disorder is precomputed once.
- **Noise:** each site gets an independent standard Gaussian at every
  step, from Box–Muller on `Philox4x32` keyed by the noise seed. A run is
  therefore exactly reproducible given `(seed, noise-seed)`, independently
  of the GPU thread layout.
- **Field:** `h(t)` is evaluated at the beginning of each step (Itô
  convention). Time is computed as `n * dt`, not accumulated, so it does
  not drift.
- **Stability:** explicit Euler with the 5-point Laplacian is unstable for
  `dt * c >~ 0.25`. The program warns when `dt * c > 0.1`; `dt = 0.05` is
  a safe default for `c = 1`.

**Wall detection on the GPU:** the program writes a lightweight output of
the wall every `tout` time units.
1. The field is smoothed with a 5-point average (self + 4 neighbors),
   which removes spurious sign flips caused by thermal noise.
2. A wall crossing is a sign change of the smoothed field along a
   horizontal or vertical bond. Checking both directions catches every
   part of a closed contour.
3. Each crossing position is linearly interpolated along its bond.

The program also computes:
- the positive area `A+`;
- the effective radius `R_eff = sqrt(A+/pi)`;
- the center of mass of the `+` domain, as a circular mean in each
  direction so it is correct with periodic boundaries;
- the mean and variance of the distance from each wall point to that
  center of mass. The variance measures the contour roughness.

## Build

Requires the NVIDIA CUDA toolkit (`nvcc`, C++14) and a GPU. The only
dependency is `Random123/`, which is vendored and header-only.

```
make                    # ./phi4langevin, default arch sm_61
make ARCH=sm_86         # compile for a specific GPU architecture
make CEL=2.0            # change c (or EPSILON0, R0)
```

The analysis scripts need Python 3 with `numpy`, `matplotlib` and
`contourpy` (the latter comes with matplotlib >= 3.6). `pillow` is also
needed for GIFs.

## Run

```
./phi4langevin [--option value ...]      # --help lists the options
```

| option         | meaning                                       | default   |
|----------------|-----------------------------------------------|-----------|
| `--L`          | lattice size (`L x L`)                        | 256       |
| `--h0`         | AC field amplitude                            | 0         |
| `--freq`       | AC field frequency `f`                        | 0         |
| `--T`          | temperature                                   | 0         |
| `--radius`     | radius of the initial disk                    | 64        |
| `--delta`      | disorder amplitude `Delta`                    | 0.2       |
| `--dt`         | time step                                     | 0.05      |
| `--tmax`       | total simulated time                          | 100       |
| `--tout`       | time between wall outputs                     | 1         |
| `--tconf`      | time between full-field snapshots (0 = never) | 0         |
| `--seed`       | disorder seed                                 | 1         |
| `--noise-seed` | thermal-noise seed                            | seed + 1  |

To average over disorder realizations or thermal histories, loop over
`--seed` and/or `--noise-seed` in a shell script. Each run writes its
own files.

## Output files

All files are written to the working directory, with `<S>` = seed and
`<N>` = noise seed:

| file | content |
|------|---------|
| `walls_seed<S>_nseed<N>.dat` | wall crossings every `tout`. There is one gnuplot `index` block per time, headed by `# t= .. h= .. n_wall= .. A+= .. A-= .. xcm= .. ycm= ..` and followed by unordered `x y` points. |
| `timeseries_seed<S>_nseed<N>.dat` | one line per output time, with columns `t h(t) <phi> A+ R_eff n_wall xcm ycm <r> var(r)`. |
| `config_t<t>_seed<S>_nseed<N>.dat` | full `phi` field every `tconf`, as `L` rows of `L` values (row = `y`, column = `x`). Each file is ~2.4 MB for `L=512`. |
| `logfile.dat` | the run's parameters. |

For example, `plot 'walls_seed1_nseed2.dat' index 10 w d` draws the wall
at the 11th output time.

`scripts/plot_wall_evolution.py walls_seed<S>_nseed<N>.dat` plots three
panels:
- the wall at several times;
- `R_eff(t)` together with `h(t)`;
- the polar profile `r(theta)` around the center of mass.

![wall evolution, R_eff(t) and r(theta)](docs/demo_evolution.png)

## Level curves phi=0

For postprocessing the walls as **ordered closed curves**, run with
`--tconf > 0` and use `scripts/extract_contours.py`. It traces the
`phi = 0` level set of every snapshot:
- the field is smoothed with the same 5-point average as the GPU
  detector (`--no-smooth` traces the raw field);
- contours are traced with marching squares (`contourpy`);
- pieces cut by the periodic boundaries are stitched back together, so
  each wall is one polyline in unwrapped coordinates.

A time can have several curves, for example when the domain splits or
when droplets nucleate. A curve that wraps around the box, such as a
stripe spanning the sample, has a nonzero winding number `(wx, wy)`.

```
python3 scripts/extract_contours.py <dir> [--min-length 8] [--run seed1_nseed2]
python3 scripts/plot_contours.py <dir>/contours_seed1_nseed2.dat [--panels 16] [--gif]
```

**`contours_<run>.dat`** has one gnuplot `index` block per time. Each
block starts with `# t= .. ncurves= ..`, followed by one sub-block per
curve, longest first:

```
# curve= k npts= .. length= .. area= .. inside= +1|-1 wx= .. wy= .. xc= .. yc= ..
x0 y0
x1 y1
...
x0 y0          <- closed curves repeat the first point
```

Curves are separated by one blank line and times by two, so
`plot 'contours_seed1_nseed2.dat' index i w l` draws time `i` with
each curve as a separate line.
- `inside`: the sign of `phi` inside the curve. `+1` is a `+` domain,
  `-1` is a `-` hole or droplet.
- `area`: the enclosed area.
- `length`: the polyline length.
- `(xc, yc)`: the centroid, folded back into the box.

**`contours_<run>.csv`** has the same per-curve summary, one row per
`(t, curve)`.

`plot_contours.py` writes:
- a grid of snapshots with each curve in its own color (the figure at
  the top of this README);
- the number of curves, the total wall length and the enclosed `+` area
  vs `t`;
- with `--gif`, an animation.

## Demo

```
scripts/run_demo.sh [outdir]     # default outdir: demo/ (~190 MB, ~15 s on an RTX A4000)
```

The demo runs `L=512`, `R=150`, `Delta=1`, `T=0.02`, `h0=0.3`, `f=0.004`
(period 250) for three periods:
- walls every `t=5`;
- snapshots every `t=10`.

Then it extracts the level curves and produces all the plots.

![demo animation](docs/demo.gif)
![number of curves, wall length and + area vs t](docs/demo_ncurves.png)

What the demo shows:
- The `+` area oscillates steadily between ~47 000 and ~96 000 and lags
  behind the field.
- The wall roughens as it moves through the disorder.
- The disorder is strong: at `Delta = 1`, some sites have
  `1 + r ~ 0`. Small `+` droplets nucleate outside the domain when
  `h > 0`, and `-` holes nucleate inside it when `h < 0`. That is why the
  level set has up to ~6 curves per snapshot; `inside` distinguishes the
  two kinds.

Most of these extra curves are nucleation events, not pinch-off of the
main domain. To study splitting of the main domain, use weaker disorder
(e.g. `Delta ~ 0.5–0.8`) or lower `T` to suppress nucleation.

## Checks

- **Curvature-driven shrinking** (`--delta 0 --T 0 --h0 0`): Allen–Cahn
  predicts `R^2(t) = R^2(0) - 2 c t`.
  - With `c = 1`, `L = 256`, `R = 80`, the fitted slope is `-1.88 c`. The
    wall is only ~1.4 lattice spacings wide, so the lattice
    discretization reduces the slope.
  - With `c = 4` (a wider wall), the slope is `-1.975 c`.
- **DC field** (`--freq 0 --h0 0.1 --delta 0 --T 0`): the radius grows at
  a constant speed of ~0.197. The sharp-interface prediction is
  `v = 2 h / sigma - c/R`, with wall tension `sigma = 2 sqrt(2)/3`,
  which gives ~0.20.
- **Reproducibility:** repeated runs with the same seeds give
  byte-identical output. Changing `--noise-seed` changes it.
- **Contour stitching:** on synthetic periodic fields, `extract_contours.py`
  gives:
  - one closed curve for a disk that crosses the box corner;
  - `inside = -1` for a hole inside a domain;
  - winding `(±1, 0)` for a stripe that wraps around the box.

## Repository layout

```
main.cu                        option parsing, time loop, output
langevin.h                     System class: Euler-Maruyama kernel, disorder, noise, wall detection
Makefile
scripts/run_demo.sh            demo run + analysis
scripts/plot_wall_evolution.py walls / R_eff(t) / r(theta) from walls_*.dat
scripts/extract_contours.py    phi=0 level curves (ordered, periodic-stitched) from snapshots
scripts/plot_contours.py       grid / time series / GIF of the level curves
docs/                          figures used in this README
Random123/                     vendored counter-based RNG (D. E. Shaw Research)
```

## Citing

If you use this code, please cite the paper that introduced the model
setup:

```bibtex
@article{PhysRevB.108.174201,
  title     = {Depinning free of the elastic approximation},
  author    = {Kolton, Alejandro B. and Ferrero, Ezequiel E. and Rosso, Alberto},
  journal   = {Phys. Rev. B},
  volume    = {108},
  issue     = {17},
  pages     = {174201},
  year      = {2023},
  publisher = {American Physical Society},
  doi       = {10.1103/PhysRevB.108.174201}
}
```

## License

MIT — see [LICENSE](LICENSE) — for the code in this repository.

`Random123/` is a vendored, header-only copy of the
[Random123](https://www.deshawresearch.com/resources_random123.html)
counter-based RNG library by D. E. Shaw Research, BSD-3-Clause licensed
(license text in each file's header).
