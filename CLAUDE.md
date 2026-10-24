# UAV Inspection Trajectory Optimization

Research project extending Gottumukkala & Murphy, *"Spline-path based autonomous
infrastructure inspection using an unmanned aerial vehicle"* (Texas A&M). A drone
films a building for inspection. The paper flies a cubic spline through the pilot's
photo locations with the camera perpendicular to the path. This project replaces
that hand-placed spline with an **optimized trajectory** that minimizes an
objective J built from the paper's own evaluation metrics.

The goal is a simulation that compares three methods on the same buildings:
1. **Orbit**: circle around the building center, camera at the center (baseline in the paper)
2. **Naive spline**: the paper's method (spline through pilot waypoints, camera ⟂ path)
3. **Optimized**: spline waypoints and camera yaw chosen by minimizing J, solved offline

A teammate (Darwin) is presenting naive vs. optimized, and the codebases will be
merged. Keep the objective and metrics importable and decoupled from any one
optimizer or simulator.

## Existing code (do not rewrite without reason)

- `objective.py`: the core. Read its top docstring first; it holds the full math.
  - `Building(vertices, sub_len=1.0)`: polygon footprint, split into sub-edges
    (`m` midpoints, `n` outward normals, `l` lengths, `eidx` parent edge).
    Has `line_of_sight`, `signed_distance` and `cast_center_rays`.
  - `SplinePath(waypoints, yaw_offsets)` and `OrbitPath(center, radius)`: any path
    exposes `.sample(n) -> (xy (n,2), yaw_deg (n,), length_m)`, equispaced in
    arclength, closed loop, CCW. **This is the interface everything depends on.**
  - `evaluate(path, building, speed)`: hard metrics exactly as the paper defines
    them (bugs fixed), sampled at 4 Hz.
  - `objective(path, building, Weights, ObjectiveConfig)`: smooth surrogate J
    that the optimizer minimizes.
- `demo.py`: runs orbit / naive / optimized on U-shape, L-shape and rectangle,
  prints a metrics table and saves `comparison.png`. `optimize()` here is the
  reference optimizer (scipy L-BFGS-B, finite differences, about 10–50 s per building).

Run: `python demo.py` (needs numpy, scipy, matplotlib).

## The objective (summary; the docstring is authoritative)

Decision variables: K waypoints (x, y) of a closed periodic cubic spline, plus K
yaw offsets δ (±60°) added to "perpendicular to path, facing inward".
δ = 0 reproduces the paper.

```
J = 4.0·(1 − coverage) + 1.5·(angle_dev/90°) + 1.0·(obs_dist/50 m)
  + 1.0·(1 − stability) + 0.5·(yaw_rate²/(10°/s)²) + 0.3·(length/L_init)
  + 50·mean(relu(10 m − standoff)²)          # safety: stay ≥10 m from walls
  + 5·mean(relu(|yaw_rate| − 30°/s)²)/30²    # gimbal/yaw-rate limit
```

Binary quantities (is the sub-edge seen? does the center ray hit?) are replaced by
sigmoids so the optimizer gets a signal. Line-of-sight occlusion stays a hard mask
(piecewise constant).

## Metric definitions (keep these exact, since results must compare to the paper)

- **Coverage is center-ray based**, not field-of-view based. A sub-edge counts as
  covered when the camera's center ray hits within d/12 of it (the paper's
  "sub-edge length ≤ 1/6 of observation distance" rule). FOV-based coverage
  saturates at 100% for every method and is reported only as `fov_coverage_pct`.
- Angle deviation (target) = per covered sub-edge, best incidence angle vs. the
  wall normal (= |angle of depression − 90°| in the paper), length-weighted mean.
- Observation distance, angle (UAV) and stability all come from the center ray.
- Smoothness = mean |Δyaw| per 0.25 s frame. Use the absolute value; the paper's
  formula omits it.
- Distance traveled = total path length. The paper divides by n−1, which is a bug.
- Time = length / speed. Default speed 2 m/s (3DR Solo default in the paper).

## Conventions

- Units: metres, seconds, **degrees** for yaw and angles in the public API.
- Paths are closed and counter-clockwise. Yaw = heading + 90° faces inward. Run
  waypoints through `sort_ccw` first.
- Always wrap angle differences with `wrap_deg` before differencing yaw.
- 2D / fixed altitude, fixed gimbal pitch, matching the paper. 3D is a planned
  extension (see below), so don't hard-code 2D assumptions into new modules.
- Keep numpy vectorization; the objective is evaluated thousands of times.
- Any new trajectory type must implement `.sample(n)` so that `evaluate` and
  `objective` work unchanged.

## What to build: the simulation

Build in this order and check each step before moving on:

1. **Flight simulator** (`sim.py`): play a path back in time at constant speed
   with simple kinematic limits (max speed, max yaw rate, optional wind
   disturbance as noise on position). Output a log at 4 Hz with t, x, y, yaw and
   the commanded vs. actual values. Then `evaluate` should run on the *flown* log
   as well as the planned path, which shows whether optimized plans survive
   tracking error. A small adapter class whose `.sample(n)` resamples the log will do this.
2. **Visualization** (`animate.py`): matplotlib animation (MP4/GIF) of the drone
   flying each method side by side. Draw the building, the path, the camera's 60°
   FOV wedge, the center ray, and sub-edges recolored live from red (unseen) to
   green (seen head-on), like the paper's Figures 12–22.
3. **Experiment runner** (`experiments.py`): a set of buildings (convex, L, U,
   cross/plus, courtyard, multi-building) and randomized pilot waypoints (4–8
   points, jittered ±5 m), several seeds each. Run all three methods, save a CSV,
   and report a paired t-test of optimized vs. naive and vs. orbit per metric
   (mirroring the paper's Tables 1–2, split by concave vs. convex).
4. **Weight sweep**: vary `w_yaw` and `w_cov` and plot the coverage-vs-smoothness
   Pareto front. This trade-off is the paper's central finding, so it belongs in the presentation.
5. **Online update (Darwin's "solve offline, then update")**: re-optimize from the
   current position over the remaining path during simulated flight (receding
   horizon), warm-started from the offline solution, e.g. after a wind gust or
   an updated building outline. Confirm the intended meaning with the team
   before investing heavily here.

Optional, later: run the planned path on ArduPilot SITL via `dronekit-sitl` (the
paper's own setup) to replace the kinematic simulator.

## 3D extension (planned)

Buildings become extruded polygons with heights (walls plus roof). Add altitude and
gimbal pitch as decision variables. Coverage becomes facade-area coverage
(top-to-bottom of each wall), using the camera frustum and the wall normal for
incidence. Possibly fold in the nadir photo phase. Keep the same `.sample()`
contract, extended to return `(xyz, yaw, pitch, length)`.

## Known limitations

- Test buildings and waypoints are synthetic; the paper's 28 Hurricane Irma
  targets aren't available. Say so in any write-up.
- Finite-difference gradients through the hard occlusion mask are slow and a bit
  noisy. Porting `soft_terms` to JAX or PyTorch for autodiff is a worthwhile
  upgrade if the merged codebase uses one of them.
- Default weights are a first guess, not tuned.
- The naive spline violates the 10 m standoff on concave buildings (comes within
  about 4 m). This is a real finding, not a bug.

## Workflow

- After changing `objective.py`, rerun `python demo.py` and confirm: orbit
  coverage on the U and L shapes ≈ 78–79%, the optimized path keeps ≥10 m standoff,
  and the optimized angle deviation stays below the naive spline's.
- Add small pytest checks for geometry (CCW normals point outward, line of sight
  is blocked across a concave notch, a ray hits the expected wall).
- Save figures and CSVs to `results/`, not the repo root.
