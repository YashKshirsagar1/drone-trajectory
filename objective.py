"""
Objective for UAV inspection trajectory optimization (2D / fixed-altitude version).

Built to reproduce the evaluation metrics of Gottumukkala & Murphy,
"Spline-path based autonomous infrastructure inspection using a UAV",
so that minimizing J improves the numbers the paper reports.

---------------------------------------------------------------------------
SETUP
---------------------------------------------------------------------------
Building:  polygon footprint, perimeter split into sub-edges j with
           midpoint m_j, outward normal n_j, length l_j.
Path:      closed curve sampled at i = 1..N, constant speed v
           (dt_i = |p_{i+1} - p_i| / v). Position p_i, camera yaw psi_i,
           camera axis c_i = (cos psi_i, sin psi_i). Fixed altitude, fixed
           pitch, as in the paper.

Decision variables (SplinePath):
    theta = [ K waypoints (x, y) ,  K_y yaw offsets delta_k ]
    p(s)   = closed periodic cubic spline through the K waypoints
    psi(s) = heading(s) + 90 deg (perpendicular, facing inward)
             + delta(s), where delta(s) is a periodic spline of offsets.
    delta = 0 reproduces the paper's "camera perpendicular to course".

Per sample/sub-edge geometry:
    u_ij    = unit vector from m_j to p_i,   d_ij = |p_i - m_j|
    cos phi_ij = n_j . u_ij      (phi = incidence angle; phi = |angle of
                                  depression - 90| in the paper)
    alpha_ij   = angle between camera axis c_i and (m_j - p_i)

---------------------------------------------------------------------------
HARD METRICS (evaluation; mirror the paper, with its formula bugs fixed)
---------------------------------------------------------------------------
  Center ray r_i = first front-facing building edge hit by camera axis c_i,
                   at distance t_i. Marks covered the sub-edges within
                   t_i/12 of the hit point (the paper's d/6 sub-edge rule).
  Coverage C     = covered perimeter / total perimeter              (maximize)
  Angle (target) = mean over covered j of best incidence phi        (minimize)
  FOV coverage   = extra, lenient: anything inside the 60 deg frame
  Stability S    = fraction of samples whose center ray hits        (maximize)
  Obs. distance  = mean over hitting samples of ray length          (minimize)
  Angle (UAV)    = mean incidence of center ray on hit edge         (minimize)
  Smoothness     = mean |wrap(psi_i - psi_{i-1})| per 0.25 s frame  (minimize)
                   (paper omits the absolute value; signed changes cancel)
  Distance       = total path length  (paper divides by n-1 by mistake)
  Time           = length / v

---------------------------------------------------------------------------
SOFT OBJECTIVE (what the optimizer minimizes)
---------------------------------------------------------------------------
  sigma(x) = logistic function.  Soft "center ray on sub-edge j":
    w_ij = sigma((cos alpha_ij - cos(CENTER_CONE/2)) / tau_c)  near camera axis
         * sigma(cos phi_ij / tau_f)                      front-facing
         * sigma((r_max - d_ij) / tau_r)                  in range
         * occ_ij  (hard line-of-sight mask, piecewise constant)

  Coverage        C~   = sum_j l_j [1 - prod_i (1 - w_ij)] / sum_j l_j
  Best angle      b_j  = softmax_i over (w_ij cos phi_ij)  (log-sum-exp, beta)
                  phi~ = coverage-weighted mean of arccos(b_j)
  Center ray hits o_i  = 1 - prod_j (1 - w_ij)
  Stability       S~   = mean_i o_i
  Obs. distance   D~   = sum_i o_i dbar_i / sum_i o_i,
                         dbar_i = w-weighted mean of d_ij
  Yaw rate        R    = mean_i (wrap(psi_{i+1}-psi_i) / dt_i)^2
  Length          L

  J(theta) = w_cov  (1 - C~)
           + w_ang  phi~ / 90
           + w_dist D~ / D_ref
           + w_stab (1 - S~)
           + w_yaw  R / R_ref^2
           + w_len  L / L_ref
           + mu_s * mean_i relu(d_min - sd(p_i))^2       standoff (safety)
           + mu_r * mean_i relu(|psi_dot_i| - psi_dot_max)^2   yaw-rate limit

  Every term is normalized to roughly [0, 1], so the weights read as
  priorities. Coverage, angle and distance pull the path in close and
  conformal to the walls; stability, yaw rate and length pull it toward a
  smooth loop. That trade-off is the paper's main finding, and the weights
  choose a point on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.interpolate import CubicSpline

FOV_DEG = 60.0           # 3DR Solo / GoPro horizontal FOV used in the paper
CENTER_CONE_DEG = 10.0   # soft stand-in for "center ray hits target"
                         # (a d/6 sub-edge at distance d subtends ~9.5 deg)
LOG_DT = 0.25            # paper logs at 4 Hz


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


class Building:
    """Polygon footprint split into sub-edges.

    `vertices` is one polygon [(x, y), ...] or, for a group of separate
    buildings, a list of polygons [[(x, y), ...], ...]. Each ring is oriented
    CCW so normals point outward; edges of all rings share one index space.
    """

    def __init__(self, vertices, sub_len: float = 1.0):
        polys = vertices if np.ndim(vertices[0]) == 2 else [vertices]
        rings = []
        for poly in polys:
            R = np.asarray(poly, float)
            if 0.5 * np.sum(_cross(R, np.roll(R, -1, axis=0))) < 0:
                R = R[::-1]
            rings.append(R)
        self.rings = rings
        V = np.vstack(rings)
        self.V = V
        self.A = V
        self.B = np.vstack([np.roll(R, -1, axis=0) for R in rings])
        d = self.B - self.A
        elen = np.linalg.norm(d, axis=1)
        self.edge_normals = np.stack([d[:, 1], -d[:, 0]], 1) / elen[:, None]

        mids, lens, normals, eidx = [], [], [], []
        for k in range(len(V)):
            n_sub = max(1, int(np.ceil(elen[k] / sub_len)))
            t = (np.arange(n_sub) + 0.5) / n_sub
            mids.append(self.A[k] + t[:, None] * d[k])
            lens.append(np.full(n_sub, elen[k] / n_sub))
            normals.append(np.repeat(self.edge_normals[k][None], n_sub, 0))
            eidx.append(np.full(n_sub, k))
        self.m = np.concatenate(mids)
        self.l = np.concatenate(lens)
        self.n = np.concatenate(normals)
        self.eidx = np.concatenate(eidx)
        self.perimeter = self.l.sum()
        self.centroid = V.mean(0)
        self.mpl_path = MplPath.make_compound_path(
            *[MplPath(np.vstack([R, R[:1]]), closed=True) for R in rings])

    # line-of-sight mask: segment p_i -> m_j must not cross the boundary
    def line_of_sight(self, P):
        Q = self.m + 0.05 * self.n                      # nudge off the wall
        P4 = P[:, None, None, :]
        Q4 = Q[None, :, None, :]
        A4 = self.A[None, None]
        B4 = self.B[None, None]
        d1 = _cross(B4 - A4, P4 - A4)
        d2 = _cross(B4 - A4, Q4 - A4)
        d3 = _cross(Q4 - P4, A4 - P4)
        d4 = _cross(Q4 - P4, B4 - P4)
        hit = (d1 * d2 < 0) & (d3 * d4 < 0)
        return ~hit.any(axis=2)

    def signed_distance(self, P):
        AP = P[:, None, :] - self.A[None]
        AB = (self.B - self.A)[None]
        t = np.clip(np.sum(AP * AB, -1) / np.sum(AB * AB, -1), 0, 1)
        dist = np.linalg.norm(AP - t[..., None] * AB, axis=-1).min(1)
        inside = self.mpl_path.contains_points(P)
        return np.where(inside, -dist, dist)

    def cast_center_rays(self, P, C):
        """First front-facing edge hit by each camera axis ray."""
        AB = (self.B - self.A)[None]                    # (1,E,2)
        AP = self.A[None] - P[:, None]                  # (N,E,2)
        Cn = C[:, None, :]
        denom = _cross(Cn, AB)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = _cross(AP, AB) / denom
            u = _cross(AP, Cn) / denom
        front = np.sum(C[:, None] * self.edge_normals[None], -1) < 0
        valid = (t > 0) & (u >= 0) & (u <= 1) & front & np.isfinite(t)
        t = np.where(valid, t, np.inf)
        k = np.argmin(t, 1)
        dist = t[np.arange(len(P)), k]
        hit = np.isfinite(dist)
        cos_inc = -np.sum(C * self.edge_normals[k], 1)
        inc = np.degrees(np.arccos(np.clip(cos_inc, -1, 1)))
        return hit, dist, inc, k


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------
def _resample_closed(xy_dense, n):
    closed = np.vstack([xy_dense, xy_dense[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    sq = np.linspace(0, L, n, endpoint=False)
    xy = np.stack([np.interp(sq, s, closed[:, 0]), np.interp(sq, s, closed[:, 1])], 1)
    return xy, sq / L, L


def _closed_heading(xy):
    d = np.roll(xy, -1, 0) - np.roll(xy, 1, 0)
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


def sort_ccw(points):
    P = np.asarray(points, float)
    c = P.mean(0)
    ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
    return P[np.argsort(ang)]


@dataclass
class SplinePath:
    """Closed spline through waypoints; yaw = perpendicular + optional offsets."""
    waypoints: np.ndarray
    yaw_offsets: np.ndarray = field(default=None)
    dense: int = 1500

    def __post_init__(self):
        self.waypoints = np.asarray(self.waypoints, float)
        if self.yaw_offsets is None:
            self.yaw_offsets = np.zeros(len(self.waypoints))

    def sample(self, n):
        W = self.waypoints
        K = len(W)
        t = np.arange(K + 1)
        cs = CubicSpline(t, np.vstack([W, W[:1]]), bc_type="periodic")
        xy_dense = cs(np.linspace(0, K, self.dense, endpoint=False))
        xy, s, L = _resample_closed(xy_dense, n)
        heading = _closed_heading(xy)
        Ky = len(self.yaw_offsets)
        ys = CubicSpline(np.linspace(0, 1, Ky + 1),
                         np.append(self.yaw_offsets, self.yaw_offsets[0]),
                         bc_type="periodic")
        yaw = heading + 90.0 + ys(s)            # +90: face left = inward for CCW
        return xy, wrap_deg(yaw), L

    # flatten / unflatten for the optimizer
    def to_vector(self):
        return np.concatenate([self.waypoints.ravel(), self.yaw_offsets])

    @staticmethod
    def from_vector(x, K, Ky):
        return SplinePath(x[: 2 * K].reshape(K, 2), x[2 * K: 2 * K + Ky])


@dataclass
class OrbitPath:
    center: np.ndarray
    radius: float

    def sample(self, n):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xy = self.center + self.radius * np.stack([np.cos(a), np.sin(a)], 1)
        d = self.center - xy
        yaw = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        return xy, yaw, 2 * np.pi * self.radius


def orbit_from_waypoints(building, waypoints):
    """Paper's orbit setup: center = building center, radius = farthest waypoint."""
    c = building.centroid
    r = np.max(np.linalg.norm(np.asarray(waypoints) - c, axis=1))
    return OrbitPath(c, r)


# ---------------------------------------------------------------------------
# Hard metrics (evaluation)
# ---------------------------------------------------------------------------
def center_ray_footprint(building: Building, P, C):
    """Sub-edges each frame's center ray covers (the paper's d/6 sub-edge rule).

    Returns hit (N,), ray length (N,), incidence deg (N,) and F (N, J) bool,
    F[i, j] = frame i covers sub-edge j.
    """
    hit, ray_d, inc, k = building.cast_center_rays(P, C)
    H = P + np.where(hit, ray_d, 0)[:, None] * C
    half = np.maximum(ray_d / 12.0, 0.5)
    near = np.linalg.norm(building.m[None] - H[:, None], axis=-1) <= half[:, None]
    F = hit[:, None] & (building.eidx[None] == k[:, None]) & near
    return hit, ray_d, inc, F


def evaluate(path, building: Building, speed: float = 2.0, r_max: float = 150.0):
    n = getattr(path, "n_frames", None)               # flown logs: use the real frames
    if n is None:
        _, _, L = path.sample(400)
        n = max(8, int(round(L / (speed * LOG_DT))))   # 4 Hz at given speed
    P, yaw, L = path.sample(n)
    C = np.stack([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))], 1)

    # FOV coverage (lenient, extra): anything inside the 60 deg frame
    diff = P[:, None] - building.m[None]
    d = np.linalg.norm(diff, axis=-1)
    u = diff / d[..., None]
    cos_phi = np.sum(u * building.n[None], -1)
    cos_alpha = np.sum(-u * C[:, None], -1)
    los = building.line_of_sight(P)
    seen_fov = ((cos_alpha >= np.cos(np.radians(FOV_DEG / 2))) & (cos_phi > 0)
                & los & (d <= r_max))
    fov_cov = 100 * building.l[seen_fov.any(0)].sum() / building.perimeter

    # Paper-faithful coverage: center ray hits a sub-edge of length d/6
    hit, ray_d, inc, F = center_ray_footprint(building, P, C)
    best_phi = np.where(F, inc[:, None], np.inf).min(0)
    covered = np.isfinite(best_phi)
    coverage = 100 * building.l[covered].sum() / building.perimeter
    ang_target = (best_phi[covered] * building.l[covered]).sum() / max(
        building.l[covered].sum(), 1e-9)

    stability = 100 * hit.mean()
    obs_dist = ray_d[hit].mean() if hit.any() else np.nan
    ang_uav = inc[hit].mean() if hit.any() else np.nan

    dpsi = np.abs(wrap_deg(np.diff(np.append(yaw, yaw[0]))))
    sd = building.signed_distance(P)
    return {
        "coverage_pct": coverage,
        "fov_coverage_pct": fov_cov,
        "angle_dev_target_deg": ang_target,
        "angle_dev_uav_deg": ang_uav,
        "obs_distance_m": obs_dist,
        "visual_stability_pct": stability,
        "smoothness_deg_per_frame": dpsi.mean(),
        "max_yaw_step_deg": dpsi.max(),
        "distance_m": L,
        "time_s": L / speed,
        "min_standoff_m": sd.min(),
    }


# ---------------------------------------------------------------------------
# Soft objective (optimization)
# ---------------------------------------------------------------------------
@dataclass
class Weights:
    cov: float = 4.0
    ang: float = 1.5
    dist: float = 1.0
    stab: float = 1.0
    yaw: float = 0.5
    length: float = 0.3


@dataclass
class ObjectiveConfig:
    speed: float = 2.0            # m/s
    d_min: float = 10.0           # m, standoff from walls
    yaw_rate_max: float = 30.0    # deg/s
    r_max: float = 120.0          # m
    D_ref: float = 50.0           # m
    R_ref: float = 10.0           # deg/s
    L_ref: float = None           # m; default = initial path length
    mu_standoff: float = 50.0
    mu_yawrate: float = 5.0
    tau_a: float = 0.02
    tau_f: float = 0.05
    tau_r: float = 5.0
    beta: float = 25.0
    n_samples: int = 160


def _sig(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


def soft_terms(path, building: Building, cfg: ObjectiveConfig):
    P, yaw, L = path.sample(cfg.n_samples)
    C = np.stack([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))], 1)

    diff = P[:, None] - building.m[None]
    d = np.linalg.norm(diff, axis=-1) + 1e-9
    u = diff / d[..., None]
    cos_phi = np.sum(u * building.n[None], -1)
    cos_alpha = np.sum(-u * C[:, None], -1)
    los = building.line_of_sight(P).astype(float)

    face = _sig(cos_phi / cfg.tau_f)
    rng = _sig((cfg.r_max - d) / cfg.tau_r)
    base = face * rng * los
    w = _sig((cos_alpha - np.cos(np.radians(CENTER_CONE_DEG / 2))) / cfg.tau_a / 10) * base

    eps = 1e-6
    cov_j = 1 - np.exp(np.sum(np.log1p(-np.clip(w, 0, 1 - eps)), 0))
    C_soft = np.sum(building.l * cov_j) / building.perimeter

    g = w * np.clip(cos_phi, 0, 1)
    gmax = g.max(0)
    b = gmax + np.log(np.mean(np.exp(cfg.beta * (g - gmax)), 0) + 1e-12) / cfg.beta
    b = np.clip(b, 0, 1)
    phi_j = np.degrees(np.arccos(b))
    wj = building.l * cov_j
    phi_soft = np.sum(wj * phi_j) / (wj.sum() + eps)

    o = 1 - np.exp(np.sum(np.log1p(-np.clip(w, 0, 1 - eps)), 1))
    dbar = np.sum(w * d, 1) / (w.sum(1) + eps)
    D_soft = np.sum(o * dbar) / (o.sum() + eps)
    S_soft = o.mean()

    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    dt = np.maximum(seg / cfg.speed, 1e-3)
    psi_dot = wrap_deg(np.roll(yaw, -1) - yaw) / dt
    R = np.mean(psi_dot ** 2)

    sd = building.signed_distance(P)
    pen_standoff = np.mean(np.maximum(cfg.d_min - sd, 0) ** 2)
    pen_yaw = np.mean(np.maximum(np.abs(psi_dot) - cfg.yaw_rate_max, 0) ** 2)

    return dict(coverage=C_soft, angle=phi_soft, distance=D_soft,
                stability=S_soft, yaw_rate_sq=R, length=L,
                pen_standoff=pen_standoff, pen_yawrate=pen_yaw)


def objective(path, building, weights: Weights, cfg: ObjectiveConfig, return_terms=False):
    t = soft_terms(path, building, cfg)
    L_ref = cfg.L_ref or t["length"]
    parts = {
        "coverage": weights.cov * (1 - t["coverage"]),
        "angle": weights.ang * t["angle"] / 90.0,
        "distance": weights.dist * t["distance"] / cfg.D_ref,
        "stability": weights.stab * (1 - t["stability"]),
        "yaw": weights.yaw * t["yaw_rate_sq"] / cfg.R_ref ** 2,
        "length": weights.length * t["length"] / L_ref,
        "standoff": cfg.mu_standoff * t["pen_standoff"],
        "yaw_limit": cfg.mu_yawrate * t["pen_yawrate"] / cfg.yaw_rate_max ** 2,
    }
    J = sum(parts.values())
    return (J, parts, t) if return_terms else J
