"""
Side-by-side animation of the drone flying each method (cf. paper Figs. 12-22).

Each panel shows the building, the planned path (dashed), the flown trail, the
camera's 60 deg FOV wedge and center ray, and the building's sub-edges recolored
live: red = not yet covered, then amber -> green as the best incidence angle
seen so far improves toward head-on. Coverage uses `center_ray_footprint`, the
same rule as `evaluate`, so the final frame's coverage % equals the metric.

    logs = {"Orbit": fly(orbit), "Naive": fly(naive), "Optimized": fly(opt)}
    animate(logs, building, "results/anim.mp4")

Input is a FlightLog per method (fly a planned path with SimConfig() for the
ideal case). Plots use the first two coordinates only, so 3D logs still draw
as a top view.
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Wedge

from objective import FOV_DEG, LOG_DT, Building, center_ray_footprint

UNSEEN = "#d62728"
# amber (grazing, 90 deg) -> green (head-on, 0 deg); lightness drops along the
# ramp so it still reads without red/green discrimination
SEEN_CMAP = LinearSegmentedColormap.from_list(
    "incidence", ["#f2c14e", "#9bc53d", "#2e8b57", "#145a32"])
PATH_C, TRAIL_C, DRONE_C, CAM_C = "#9a9a9a", "#1f6feb", "#111111", "#d9480f"


def _ffmpeg():
    """ffmpeg on PATH, else next to this interpreter (conda/micromamba envs)."""
    exe = shutil.which("ffmpeg") or os.path.join(sys.prefix, "bin", "ffmpeg")
    return exe if os.path.exists(exe) else None


def coverage_history(log, building: Building):
    """Per-frame running best incidence (N, J); inf = not yet covered."""
    P = log.xy[:, :2]
    C = np.stack([np.cos(np.radians(log.yaw)), np.sin(np.radians(log.yaw))], 1)
    hit, ray_d, inc, F = center_ray_footprint(building, P, C)
    best = np.minimum.accumulate(np.where(F, inc[:, None], np.inf), axis=0)
    cov = 100 * (np.isfinite(best) * building.l).sum(1) / building.perimeter
    return best, cov, hit, ray_d


def _subedge_segments(b: Building):
    d = np.stack([-b.n[:, 1], b.n[:, 0]], 1) * (b.l / 2)[:, None]   # along-wall
    return np.stack([b.m - d, b.m + d], 1)


def _colors(best):
    rgba = SEEN_CMAP(1 - np.clip(np.nan_to_num(best, posinf=90), 0, 90) / 90)
    rgba[~np.isfinite(best)] = matplotlib.colors.to_rgba(UNSEEN)
    return rgba


def animate(logs: dict, building: Building, out: str, speedup: float = 8.0,
            fps: int = 16, planned: dict = None, ray_len_miss: float = 30.0,
            dpi: int = 110, title: str = None):
    """Render logs {label: FlightLog} side by side to out (.mp4 or .gif).

    Returns (file written, {label: final coverage %}).

    speedup: sim seconds per video second. planned: optional {label: path}
    drawn dashed underneath; defaults to each log's commanded trajectory.
    """
    labels = list(logs)
    hist = {k: coverage_history(logs[k], building) for k in labels}
    step = max(1, int(round(speedup / (fps * LOG_DT))))          # log frames per video frame
    n_log = max(len(logs[k].t) for k in labels)
    frames = list(range(0, n_log, step)) + [n_log - 1] * fps      # hold the end 1 s

    allxy = np.vstack([logs[k].xy[:, :2] for k in labels] + [building.V])
    lo, hi = allxy.min(0) - 8, allxy.max(0) + 8
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), squeeze=False,
                             layout="constrained")
    axes = axes[0]
    segs = _subedge_segments(building)
    art = {}
    for ax, k in zip(axes, labels):
        log = logs[k]
        for R in building.rings:
            ax.fill(R[:, 0], R[:, 1], color="#e6e6e6", zorder=0)
        if planned and k in planned:
            P, _, _ = planned[k].sample(600)
        else:
            P = log.xy_cmd[:, :2]
        ax.plot(*np.vstack([P, P[:1]]).T, "--", color=PATH_C, lw=1, zorder=1)
        edges = LineCollection(segs, linewidths=4, capstyle="butt", zorder=3)
        ax.add_collection(edges)
        trail, = ax.plot([], [], "-", color=TRAIL_C, lw=1.5, zorder=2)
        wedge = Wedge((0, 0), 1, 0, 1, color=CAM_C, alpha=0.15, lw=0, zorder=2)
        ax.add_patch(wedge)
        ray, = ax.plot([], [], "-", color=CAM_C, lw=1.2, zorder=4)
        drone, = ax.plot([], [], "o", color=DRONE_C, ms=6, zorder=5)
        txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=8,
                      family="monospace")
        ax.set_title(k, fontsize=10)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#bbbbbb")
        art[k] = (edges, trail, wedge, ray, drone, txt)

    sm = plt.cm.ScalarMappable(cmap=SEEN_CMAP.reversed(), norm=plt.Normalize(0, 90))
    cb = fig.colorbar(sm, ax=list(axes), orientation="horizontal", fraction=0.04,
                      pad=0.02, aspect=50, shrink=0.8)
    cb.set_label("best incidence so far (deg from wall normal; 0 = head-on)   "
                 "red = not yet covered", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    if title:
        fig.suptitle(title, fontsize=11)

    def draw(f):
        out_art = []
        for k in labels:
            log = logs[k]
            best, cov, hit, ray_d = hist[k]
            i = min(f, len(log.t) - 1)
            edges, trail, wedge, ray, drone, txt = art[k]
            p = log.xy[i, :2]
            yaw = log.yaw[i]
            r = ray_d[i] if hit[i] else ray_len_miss
            edges.set_color(_colors(best[i]))
            trail.set_data(log.xy[: i + 1, 0], log.xy[: i + 1, 1])
            wedge.set_center(tuple(p))
            wedge.set_radius(r)
            wedge.set_theta1(yaw - FOV_DEG / 2)
            wedge.set_theta2(yaw + FOV_DEG / 2)
            end = p + r * np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
            ray.set_data([p[0], end[0]], [p[1], end[1]])
            ray.set_linestyle("-" if hit[i] else ":")
            drone.set_data([p[0]], [p[1]])
            txt.set_text(f"t {log.t[i]:6.1f} s\ncov {cov[i]:5.1f} %")
            out_art += [edges, trail, wedge, ray, drone, txt]
        return out_art

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    if not out.endswith(".gif"):
        if _ffmpeg():
            matplotlib.rcParams["animation.ffmpeg_path"] = _ffmpeg()
        else:
            out = os.path.splitext(out)[0] + ".gif"
            print(f"  ffmpeg not found, writing {out} instead")
    if out.endswith(".gif"):
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(out, writer=writer, dpi=dpi)
    plt.close(fig)
    return out, {k: hist[k][1][-1] for k in labels}


def main():
    import argparse

    matplotlib.use("Agg")
    from demo import L_SHAPE, RECT, U_SHAPE, WP_L, WP_R, WP_U, optimize
    from objective import (ObjectiveConfig, SplinePath, Weights, evaluate,
                           orbit_from_waypoints, sort_ccw)
    from sim import FlownPath, SimConfig, fly

    cases = {"U": ("U-shape", U_SHAPE, WP_U), "L": ("L-shape", L_SHAPE, WP_L),
             "rect": ("Rectangle", RECT, WP_R)}
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", choices=list(cases), nargs="+", default=list(cases))
    ap.add_argument("--wind", type=float, default=0.0, help="turbulence std, m/s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speedup", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--gif", action="store_true", help="write GIF instead of MP4")
    ap.add_argument("--no-opt", action="store_true", help="skip the slow optimizer")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    cfg = SimConfig(wind_std=args.wind, wind_mean=(0.5, 0.0) if args.wind else (0.0, 0.0))
    for key in args.building:
        name, verts, wps = cases[key]
        b = Building(verts)
        paths = {"Orbit": orbit_from_waypoints(b, wps),
                 "Naive spline": SplinePath(sort_ccw(wps))}
        if not args.no_opt:
            print(name)
            paths["Optimized"] = optimize(b, wps, Weights(), ObjectiveConfig())
        logs = {k: fly(p, cfg, seed=args.seed) for k, p in paths.items()}
        tag = f"{key.lower()}_{'wind' if args.wind else 'calm'}"
        out = f"results/anim_{tag}.{'gif' if args.gif else 'mp4'}"
        cond = f"wind {args.wind} m/s" if args.wind else "calm"
        out, final = animate(logs, b, out, args.speedup, args.fps, planned=paths,
                        title=f"{name}, {cond}, {args.speedup:g}x speed")
        for k in logs:
            m = evaluate(FlownPath(logs[k]), b, cfg.speed)
            print(f"  {k:13s} animated cov {final[k]:6.2f}%   evaluate cov "
                  f"{m['coverage_pct']:6.2f}%")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
