"""
Kinematic flight simulator: fly a planned path and log what the drone actually did.

Any object with `.sample(n) -> (xy, yaw_deg, length)` can be flown. The
simulator replays it in time at constant speed with simple limits, and the
resulting `FlightLog` can be wrapped in `FlownPath` so that `objective.evaluate`
scores the *flown* trajectory exactly like a planned one:

    log = fly(path, SimConfig(wind_std=1.5), seed=0)
    evaluate(FlownPath(log), building, speed=2.0)

---------------------------------------------------------------------------
MODEL (integrated at dt = 0.05 s, logged at 4 Hz like the paper)
---------------------------------------------------------------------------
Reference:  a carrot moving along the planned path at the cruise speed,
            s_ref(t) = v t. It gives position p_ref, unit tangent T, camera
            yaw psi_ref and its rate dpsi/ds.
Position:   v_cmd = v T + k_p (p_ref - p)            (feedforward + P)
            |v_cmd| <= v_max,  |dv/dt| <= a_max
            p <- p + (v_drone + w) dt                 (wind adds to ground speed)
Yaw:        r_cmd = v dpsi/ds + k_yaw wrap(psi_ref - psi)
            |r_cmd| <= r_max                          (yaw / gimbal rate limit)
Wind:       w = mean + Ornstein-Uhlenbeck noise (std sigma_w, correlation
            time tau_w) + optional gusts (t_start, duration, vector).

Positions are (N, D) arrays throughout, so the 3D extension only needs a 3D
reference; yaw stays a separate channel (pitch would be another one).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from objective import LOG_DT, wrap_deg


@dataclass
class SimConfig:
    speed: float = 2.0             # m/s, cruise speed (3DR Solo default in the paper)
    max_speed: float = 5.0         # m/s, cap on commanded ground-relative speed
    max_accel: float = 2.5         # m/s^2
    max_yaw_rate: float = 30.0     # deg/s, matches ObjectiveConfig.yaw_rate_max
    k_pos: float = 1.0             # 1/s, position feedback gain
    k_yaw: float = 4.0             # 1/s, yaw feedback gain
    dt: float = 0.05               # s, integration step (must divide LOG_DT)
    wind_mean: tuple = (0.0, 0.0)  # m/s, steady wind
    wind_std: float = 0.0          # m/s, turbulence std per axis
    wind_tau: float = 4.0          # s, turbulence correlation time
    gusts: list = field(default_factory=list)   # [(t_start, duration, (wx, wy)), ...]
    laps: float = 1.0


class _Reference:
    """Planned path, densely sampled and interpolated periodically in arclength."""

    def __init__(self, path, spacing=0.1):
        _, _, L = path.sample(400)
        n = max(400, int(np.ceil(L / spacing)))
        xy, yaw, L = path.sample(n)
        self.L = L
        self.s = np.linspace(0, L, n, endpoint=False)
        self.xy = xy
        self.c, self.sn = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        ds = L / n
        tan = (np.roll(xy, -1, 0) - np.roll(xy, 1, 0)) / (2 * ds)
        self.tan = tan / np.linalg.norm(tan, axis=1, keepdims=True)
        self.dyaw_ds = wrap_deg(np.roll(yaw, -1) - np.roll(yaw, 1)) / (2 * ds)

    def _i(self, s, f):
        return np.interp(s, self.s, f, period=self.L)

    def __call__(self, s):
        p = np.array([self._i(s, self.xy[:, d]) for d in range(self.xy.shape[1])])
        T = np.array([self._i(s, self.tan[:, d]) for d in range(self.tan.shape[1])])
        T /= max(np.linalg.norm(T), 1e-9)
        yaw = np.degrees(np.arctan2(self._i(s, self.sn), self._i(s, self.c)))
        return p, T, yaw, self._i(s, self.dyaw_ds)


class _Wind:
    def __init__(self, cfg: SimConfig, dim: int, rng):
        self.cfg, self.rng = cfg, rng
        self.mean = np.zeros(dim)
        self.mean[:len(cfg.wind_mean)] = cfg.wind_mean
        self.ou = np.zeros(dim)
        if cfg.wind_std > 0:
            self.ou = rng.normal(0, cfg.wind_std, dim)   # start in steady state

    def step(self, t, dt):
        cfg = self.cfg
        if cfg.wind_std > 0:
            a = np.exp(-dt / cfg.wind_tau)
            self.ou = a * self.ou + cfg.wind_std * np.sqrt(1 - a * a) * \
                self.rng.normal(size=self.ou.shape)
        w = self.mean + self.ou
        for t0, dur, vec in cfg.gusts:
            if t0 <= t < t0 + dur:
                w[:len(vec)] += vec
        return w


@dataclass
class FlightLog:
    """4 Hz log. *_cmd is the reference the controller was tracking."""
    t: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    xy_cmd: np.ndarray
    yaw_cmd: np.ndarray
    vel: np.ndarray            # ground velocity (drone + wind)
    yaw_rate: np.ndarray       # deg/s, commanded after the limit (= actual)
    wind: np.ndarray
    planned_length: float
    cfg: SimConfig

    @property
    def pos_err(self):
        return np.linalg.norm(self.xy - self.xy_cmd, axis=1)

    @property
    def yaw_err(self):
        return np.abs(wrap_deg(self.yaw - self.yaw_cmd))

    def summary(self):
        lim = self.cfg.max_yaw_rate
        return {
            "pos_err_rms_m": float(np.sqrt(np.mean(self.pos_err ** 2))),
            "pos_err_max_m": float(self.pos_err.max()),
            "yaw_err_rms_deg": float(np.sqrt(np.mean(self.yaw_err ** 2))),
            "yaw_err_max_deg": float(self.yaw_err.max()),
            "yaw_rate_saturated_pct": float(100 * np.mean(np.abs(self.yaw_rate) >= lim - 1e-6)),
            "max_ground_speed_ms": float(np.linalg.norm(self.vel, axis=1).max()),
        }

    def to_csv(self, filename):
        D = self.xy.shape[1]
        ax = "xyz"[:D]
        cols = (["t"] + list(ax) + ["yaw"] + [f"{a}_cmd" for a in ax] + ["yaw_cmd"]
                + [f"v{a}" for a in ax] + ["yaw_rate"] + [f"wind_{a}" for a in ax])
        data = np.column_stack([self.t, self.xy, self.yaw, self.xy_cmd, self.yaw_cmd,
                                self.vel, self.yaw_rate, self.wind])
        with open(filename, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(np.round(data, 4).tolist())


def fly(path, cfg: SimConfig = None, seed: int = 0) -> FlightLog:
    """Fly `path` once around (cfg.laps) starting on the path, on-heading."""
    cfg = cfg or SimConfig()
    if cfg.speed > cfg.max_speed:
        raise ValueError("cruise speed exceeds max_speed")
    sub = int(round(LOG_DT / cfg.dt))
    if not np.isclose(sub * cfg.dt, LOG_DT):
        raise ValueError("dt must divide LOG_DT")

    ref = _Reference(path)
    rng = np.random.default_rng(seed)
    p, T, yaw, _ = ref(0.0)
    D = len(p)
    v = cfg.speed * T
    wind = _Wind(cfg, D, rng)
    w = np.zeros(D)
    r = 0.0
    n_log = int(round(cfg.laps * ref.L / (cfg.speed * LOG_DT)))
    rows = []
    for k in range(n_log * sub):
        t = k * cfg.dt
        p_ref, T, yaw_ref, dyaw_ds = ref(cfg.speed * t)

        if k % sub == 0:
            rows.append((t, p.copy(), yaw, p_ref, yaw_ref, v + w, r, w.copy()))

        v_cmd = cfg.speed * T + cfg.k_pos * (p_ref - p)
        sp = np.linalg.norm(v_cmd)
        if sp > cfg.max_speed:
            v_cmd *= cfg.max_speed / sp
        dv = v_cmd - v
        dv_n = np.linalg.norm(dv)
        if dv_n > cfg.max_accel * cfg.dt:
            dv *= cfg.max_accel * cfg.dt / dv_n
        v = v + dv

        w = wind.step(t, cfg.dt)
        p = p + (v + w) * cfg.dt

        r = cfg.speed * dyaw_ds + cfg.k_yaw * wrap_deg(yaw_ref - yaw)
        r = float(np.clip(r, -cfg.max_yaw_rate, cfg.max_yaw_rate))
        yaw = float(wrap_deg(yaw + r * cfg.dt))

    t_, p_, y_, pc, yc, v_, r_, w_ = zip(*rows)
    return FlightLog(np.array(t_), np.array(p_), np.array(y_), np.array(pc),
                     np.array(yc), np.array(v_), np.array(r_), np.array(w_),
                     ref.L, cfg)


class FlownPath:
    """Adapter: exposes a FlightLog through the `.sample(n)` path interface.

    by="time" (default) resamples uniformly in time, so `evaluate`, which reads
    `n_frames`, scores exactly the 4 Hz frames the camera recorded.
    by="arclength" spaces samples equally along the flown polyline instead, as
    planned paths do. Under wind the flown polyline is longer than the plan
    (wiggles), so frame counts derived from length would overcount frames and
    understate the per-frame yaw change; `n_frames` avoids that.
    Length is always the flown arclength (including the closing gap).
    """

    def __init__(self, log: FlightLog, by: str = "time"):
        if by not in ("arclength", "time"):
            raise ValueError(by)
        self.log, self.by = log, by
        self.n_frames = len(log.t)
        P = np.vstack([log.xy, log.xy[:1]])
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        self._s = np.concatenate([[0], np.cumsum(seg + 1e-9)])   # strictly increasing
        self.length = self._s[-1]
        self._P = P
        self._yaw_u = np.unwrap(np.radians(np.append(log.yaw, log.yaw[0])))
        # closing segment keeps the last-to-first yaw change (unwrap handles the 360)
        T_end = log.t[-1] + LOG_DT
        self._t = np.append(log.t, T_end)

    def sample(self, n):
        if self.by == "arclength":
            xp, q = self._s, np.linspace(0, self.length, n, endpoint=False)
        else:
            xp, q = self._t, np.linspace(0, self._t[-1], n, endpoint=False)
        xy = np.stack([np.interp(q, xp, self._P[:, d]) for d in range(self._P.shape[1])], 1)
        yaw = np.degrees(np.interp(q, xp, self._yaw_u))
        return xy, wrap_deg(yaw), self.length


# ---------------------------------------------------------------------------
# Planned vs. flown on the demo buildings
# ---------------------------------------------------------------------------
def main():
    import argparse
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from demo import L_SHAPE, RECT, U_SHAPE, WP_L, WP_R, WP_U, optimize
    from objective import (Building, ObjectiveConfig, SplinePath, Weights, evaluate,
                           orbit_from_waypoints, sort_ccw)

    ap = argparse.ArgumentParser()
    ap.add_argument("--wind", type=float, default=1.5, help="turbulence std, m/s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-opt", action="store_true", help="skip the slow optimizer")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    scenarios = {"calm": SimConfig(),
                 "wind": SimConfig(wind_std=args.wind, wind_mean=(0.5, 0.0))}
    keys = ["coverage_pct", "angle_dev_target_deg", "obs_distance_m",
            "smoothness_deg_per_frame", "max_yaw_step_deg", "min_standoff_m"]
    tkeys = ["pos_err_rms_m", "pos_err_max_m", "yaw_err_max_deg", "yaw_rate_saturated_pct"]
    cases = [("U-shape", U_SHAPE, WP_U), ("L-shape", L_SHAPE, WP_L), ("Rectangle", RECT, WP_R)]

    rows = []
    fig, axes = plt.subplots(len(cases), 3, figsize=(11, 11))
    for r, (name, verts, wps) in enumerate(cases):
        b = Building(verts)
        methods = [("Orbit", orbit_from_waypoints(b, wps)),
                   ("Naive spline", SplinePath(sort_ccw(wps)))]
        if not args.no_opt:
            print(name)
            methods.append(("Optimized", optimize(b, wps, Weights(), ObjectiveConfig())))
        for c, (label, path) in enumerate(methods):
            rows.append((name, label, "planned", evaluate(path, b), {}))
            for sc, cfg in scenarios.items():
                log = fly(path, cfg, seed=args.seed)
                tag = f"{name}_{label}_{sc}".replace(" ", "_").lower()
                log.to_csv(f"results/flightlog_{tag}.csv")
                rows.append((name, label, sc, evaluate(FlownPath(log), b, cfg.speed),
                             log.summary()))
                if sc == "wind":
                    _plot(axes[r, c], b, path, log, f"{name} - {label} (wind {args.wind} m/s)")

    for ax in axes.ravel():
        if not ax.has_data():
            ax.axis("off")
    fig.tight_layout()
    fig.savefig("results/sim_tracking.png", dpi=130)

    header = ["building", "method", "run"] + keys + tkeys
    with open("results/sim_metrics.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for name, label, run, m, s in rows:
            wr.writerow([name, label, run] + [f"{m[k]:.3f}" for k in keys]
                        + [f"{s[k]:.3f}" if s else "" for k in tkeys])

    short = ["cov%", "ang°", "dist", "smooth", "maxstep", "standoff",
             "e_rms", "e_max", "yawerr", "sat%"]
    print(f"\n{'building':10s} {'method':13s} {'run':8s}" + "".join(f"{h:>9s}" for h in short))
    for name, label, run, m, s in rows:
        vals = [m[k] for k in keys] + [s.get(k, np.nan) for k in tkeys]
        print(f"{name:10s} {label:13s} {run:8s}" + "".join(f"{v:9.2f}" for v in vals))
    print("\nwrote results/sim_metrics.csv, results/sim_tracking.png, results/flightlog_*.csv")


def _plot(ax, building, path, log, title):
    for R in building.rings:
        V = np.vstack([R, R[:1]])
        ax.fill(V[:, 0], V[:, 1], color="#ccc")
        ax.plot(V[:, 0], V[:, 1], "k-", lw=1)
    P, _, _ = path.sample(400)
    ax.plot(*np.vstack([P, P[:1]]).T, "--", color="#888", lw=1, label="planned")
    ax.plot(*log.xy.T, "-", color="#1f6feb", lw=1.2, label="flown")
    k = slice(None, None, 12)
    ax.quiver(log.xy[k, 0], log.xy[k, 1], np.cos(np.radians(log.yaw[k])),
              np.sin(np.radians(log.yaw[k])), color="#d9480f", scale=18, width=0.006)
    s = log.summary()
    ax.set_title(f"{title}\npos err rms {s['pos_err_rms_m']:.1f} m, "
                 f"yaw sat {s['yaw_rate_saturated_pct']:.0f}%", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, loc="upper right")


if __name__ == "__main__":
    main()
