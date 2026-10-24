"""
Fly the optimized inspection path on ArduPilot SITL + Gazebo Harmonic.

Pipeline: demo.optimize() -> Gazebo world (building walls extruded, camera-free
Iris at the path start) -> ArduCopter SITL (JSON backend) -> GUIDED position +
velocity + yaw targets streamed over MAVLink -> pose logged at 4 Hz into a
sim.FlightLog -> objective.evaluate(FlownPath(log)) and animate.animate().

    source ~/sitl/env.sh
    python gazebo_demo.py --building U          # headless, ~5 min
    python gazebo_demo.py --building rect --gui # also open the Gazebo GUI

Frames: our (x, y) is (East, North) in metres = Gazebo world ENU. ArduPilot's
local NED origin is the spawn point. Yaw here is degrees CCW from East;
ArduPilot heading is degrees CW from North, so heading = 90 - yaw.
The reference is clocked on the autopilot's time_boot_ms, so it stays correct
when Gazebo runs slower or faster than real time.

Setup (user-space, done once): ArduPilot Copter-4.7.2 built for SITL in
~/sitl/ardupilot, ArduPilot/ardupilot_gazebo built in ~/sitl/ardupilot_gazebo,
micromamba env `sitl` (gz-sim8, pymavlink). See ~/sitl/env.sh.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

from objective import LOG_DT, Building, wrap_deg

SITL_ROOT = os.environ.get("SITL_ROOT", os.path.expanduser("~/sitl"))


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------
def world_sdf(building: Building, spawn_xy, height=20.0, wall=0.5, rtf=1.0):
    walls = []
    for k, (a, b) in enumerate(zip(building.A, building.B)):
        d = b - a
        L = np.linalg.norm(d)
        c = (a + b) / 2 - wall / 2 * building.edge_normals[k]     # outer face on the edge
        yaw = np.arctan2(d[1], d[0])
        geom = f"<geometry><box><size>{L + wall:.3f} {wall} {height}</size></box></geometry>"
        walls.append(f"""
      <collision name="c{k}"><pose>{c[0]:.3f} {c[1]:.3f} {height / 2} 0 0 {yaw:.5f}</pose>{geom}</collision>
      <visual name="v{k}"><pose>{c[0]:.3f} {c[1]:.3f} {height / 2} 0 0 {yaw:.5f}</pose>{geom}
        <material><diffuse>0.75 0.72 0.68 1</diffuse><ambient>0.6 0.58 0.55 1</ambient></material></visual>""")
    return f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="inspection">
    <physics name="1ms" type="ignore">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>{rtf}</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <scene><ambient>1 1 1</ambient><background>0.8 0.8 0.8</background><sky/></scene>
    <spherical_coordinates>
      <latitude_deg>-35.363262</latitude_deg><longitude_deg>149.165237</longitude_deg>
      <elevation>584</elevation><heading_deg>0</heading_deg>
      <surface_model>EARTH_WGS84</surface_model>
    </spherical_coordinates>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows><pose>0 0 50 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse><specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <model name="ground"><static>true</static><link name="link">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>1000 1000</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>1000 1000</size></plane></geometry>
        <material><diffuse>0.55 0.6 0.5 1</diffuse><ambient>0.55 0.6 0.5 1</ambient></material></visual>
    </link></model>
    <model name="building"><static>true</static><link name="link">{''.join(walls)}
    </link></model>
    <include>
      <uri>model://iris_with_ardupilot</uri>
      <pose degrees="true">{spawn_xy[0]:.3f} {spawn_xy[1]:.3f} 0.195 0 0 90</pose>
    </include>
  </world>
</sdf>
"""


def _sitl_env():
    env = dict(os.environ)
    gz = os.path.join(SITL_ROOT, "ardupilot_gazebo")
    env["PATH"] = os.path.join(sys.prefix, "bin") + os.pathsep + env.get("PATH", "")
    env["GZ_VERSION"] = "harmonic"
    env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = os.path.join(gz, "build")
    env["GZ_SIM_RESOURCE_PATH"] = os.path.join(gz, "models") + os.pathsep + os.path.join(gz, "worlds")
    env["GZ_RENDERING_PLUGIN_PATH"] = os.path.join(sys.prefix, "lib/gz-rendering-8/engine-plugins")
    # conda-forge libsystemd0 (>= 261, pulled in via pulseaudio) needs glibc's
    # removexattrat; on hosts whose glibc lacks it the ArduPilot plugin fails to
    # load and SITL waits forever for physics. The system copy matches the host.
    for sysd in ("/lib/aarch64-linux-gnu/libsystemd.so.0", "/lib/x86_64-linux-gnu/libsystemd.so.0"):
        if os.path.exists(sysd):
            env["LD_PRELOAD"] = sysd + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
            break
    return env


# ---------------------------------------------------------------------------
# Flight
# ---------------------------------------------------------------------------
class Copter:
    def __init__(self, url="tcp:127.0.0.1:5760", timeout=90):
        from pymavlink import mavutil
        self.mavutil = mavutil
        t0 = time.time()
        while True:
            try:
                self.m = mavutil.mavlink_connection(url, source_system=255)
                break
            except OSError:
                if time.time() - t0 > timeout:
                    raise
                time.sleep(1)
        self.m.wait_heartbeat(timeout=timeout)
        for msg_id in (30, 32):                    # ATTITUDE, LOCAL_POSITION_NED at 20 Hz
            self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                         mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                                         0, msg_id, 50000, 0, 0, 0, 0, 0)
        self.att = self.pos = None
        self.gps_fused = False

    def poll(self, timeout=1.0):
        msg = self.m.recv_match(type=["ATTITUDE", "LOCAL_POSITION_NED", "STATUSTEXT"],
                                blocking=True, timeout=timeout)
        if msg is None:
            return None
        t = msg.get_type()
        if t == "ATTITUDE":
            self.att = msg
        elif t == "LOCAL_POSITION_NED":
            self.pos = msg
        elif t == "STATUSTEXT":
            print(f"    [ap] {msg.text}")
            if "is using GPS" in msg.text:
                self.gps_fused = True
        return msg

    def wait_gps(self, timeout=180):
        """Block until the EKF fuses GPS, so the local frame can't jump in flight."""
        t0 = time.time()
        while not self.gps_fused:
            self.poll(0.5)
            if time.time() - t0 > timeout:
                raise TimeoutError("EKF never started using GPS")

    def local_ne(self, seconds=2.0):
        """Mean (north, east) of LOCAL_POSITION_NED over a short window."""
        end, ne = time.time() + seconds, []
        while time.time() < end or not ne:
            if self.poll(0.5) is not None and self.pos is not None:
                ne.append((self.pos.x, self.pos.y))
        return np.mean(ne, 0)

    def arm_guided(self, timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.m.set_mode("GUIDED")
            self.m.arducopter_arm()
            end = time.time() + 3
            while time.time() < end:
                self.poll(0.2)
            if self.m.motors_armed():
                return
        raise TimeoutError("could not arm (pre-arm checks never passed)")

    def takeoff(self, alt, timeout=60):
        mav = self.mavutil.mavlink
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                     mav.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt)
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.poll(0.5)
            if self.pos is not None and -self.pos.z > 0.95 * alt:
                return
        raise TimeoutError("takeoff did not reach altitude")

    def target(self, north, east, down, vn, ve, heading_deg):
        mav = self.mavutil.mavlink
        mask = 0b0000_1001_1100_0000             # use pos + vel + yaw; ignore accel, yaw rate
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component, mav.MAV_FRAME_LOCAL_NED, mask,
            north, east, down, vn, ve, 0, 0, 0, 0, np.radians(heading_deg), 0)

    def land(self):
        self.m.set_mode("LAND")


def fly_gazebo(path, building, alt=10.0, speed=2.0, rtf=1.0, gui=False, workdir=None,
               verbose=False):
    """Fly `path` once around on SITL + Gazebo; returns a sim.FlightLog."""
    from sim import FlightLog, SimConfig, _Reference

    ref = _Reference(path)
    p0, T0, yaw0, _ = ref(0.0)
    workdir = workdir or tempfile.mkdtemp(prefix="gz_inspection_")
    world = os.path.join(workdir, "inspection.sdf")
    with open(world, "w") as f:
        f.write(world_sdf(building, p0, height=2 * alt, rtf=rtf))

    env = _sitl_env()
    procs = []
    print(f"  logs in {workdir}")

    def spawn(cmd, name):
        out = None if verbose else open(os.path.join(workdir, f"{name}.log"), "w")
        p = subprocess.Popen(cmd, cwd=workdir, env=env, stdout=out, stderr=subprocess.STDOUT,
                             start_new_session=True)
        procs.append(p)
        return p

    params = os.path.join(SITL_ROOT, "ardupilot/Tools/autotest/default_params")
    try:
        spawn(["gz", "sim", "-s", "-r", "-v", "3", world], "gz_server")
        if gui:
            spawn(["gz", "sim", "-g"], "gz_gui")
        spawn([os.path.join(SITL_ROOT, "ardupilot/build/sitl/bin/arducopter"),
               "--model", "JSON", "--speedup", "1", "--slave", "0", "-w", "-I0",
               "--sim-address=127.0.0.1",
               "--defaults", f"{params}/copter.parm,{params}/gazebo-iris.parm"], "arducopter")
        print("  connecting to SITL ...")
        cop = Copter()
        print("  waiting for the EKF to use GPS ...")
        cop.wait_gps()
        # ArduPilot's local origin need not be the spawn point (with the JSON
        # backend it is Gazebo's world origin), so measure where we are.
        off_n, off_e = cop.local_ne()
        print(f"  spawn is at local N {off_n:.1f} E {off_e:.1f}; arming ...")
        cop.arm_guided()
        print(f"  taking off to {alt:.0f} m ...")
        cop.takeoff(alt)

        def ned(p):                                 # world (x=E, y=N) -> local (north, east)
            return off_n + p[1] - p0[1], off_e + p[0] - p0[0]

        # turn to the starting camera heading, hover 4 s
        end = time.time() + 4 / rtf
        while time.time() < end:
            cop.target(*ned(p0), -alt, 0, 0, 90 - yaw0)
            cop.poll(0.1)

        n_log = int(round(ref.L / (speed * LOG_DT)))
        print(f"  flying {ref.L:.0f} m at {speed} m/s ({n_log * LOG_DT:.0f} s sim time) ...")
        rows, t_start, next_log = [], None, 0.0
        while len(rows) < n_log:
            msg = cop.poll(2.0)
            if not cop.m.motors_armed():
                raise RuntimeError(f"vehicle disarmed at t={rows[-1][0] if rows else 0:.1f} s "
                                   "(crash?); see the .BIN log in " + workdir)
            if msg is None or msg.get_type() != "LOCAL_POSITION_NED" or cop.att is None:
                continue
            t_boot = msg.time_boot_ms / 1000
            t_start = t_start if t_start is not None else t_boot
            t = t_boot - t_start
            p_ref, T, yaw_ref, _ = ref(speed * t)
            cop.target(*ned(p_ref), -alt, speed * T[1], speed * T[0], 90 - yaw_ref)
            if t >= next_log:
                xy = np.array([p0[0] + msg.y - off_e, p0[1] + msg.x - off_n])
                yaw = float(wrap_deg(90 - np.degrees(cop.att.yaw)))
                p_cmd, _, y_cmd, _ = ref(speed * next_log)
                rows.append((next_log, xy, yaw, p_cmd, y_cmd, np.array([msg.vy, msg.vx]),
                             -np.degrees(cop.att.yawspeed), np.zeros(2)))
                next_log += LOG_DT
        print("  lap complete, landing")
        cop.land()
    finally:
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p in procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)

    t_, p_, y_, pc, yc, v_, r_, w_ = zip(*rows)
    return FlightLog(np.array(t_), np.array(p_), np.array(y_), np.array(pc), np.array(yc),
                     np.array(v_), np.array(r_), np.array(w_), ref.L,
                     SimConfig(speed=speed, max_yaw_rate=90.0))


def main():
    from animate import animate
    from demo import L_SHAPE, RECT, U_SHAPE, WP_L, WP_R, WP_U, optimize
    from objective import ObjectiveConfig, Weights, evaluate
    from sim import FlownPath

    cases = {"U": ("U-shape", U_SHAPE, WP_U), "L": ("L-shape", L_SHAPE, WP_L),
             "rect": ("Rectangle", RECT, WP_R)}
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", choices=list(cases), default="U")
    ap.add_argument("--alt", type=float, default=10.0, help="flight altitude, m")
    ap.add_argument("--rtf", type=float, default=1.0, help="Gazebo real-time factor")
    ap.add_argument("--gui", action="store_true", help="also open the Gazebo GUI")
    ap.add_argument("--verbose", action="store_true", help="show gz / SITL output")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    name, verts, wps = cases[args.building]
    b = Building(verts)
    print(f"{name}: optimizing ...")
    path = optimize(b, wps, Weights(), ObjectiveConfig())
    log = fly_gazebo(path, b, alt=args.alt, rtf=args.rtf, gui=args.gui, verbose=args.verbose)

    tag = f"gazebo_{args.building.lower()}"
    log.to_csv(f"results/{tag}_flightlog.csv")
    planned, flown = evaluate(path, b), evaluate(FlownPath(log), b)
    s = log.summary()
    print(f"\n{'metric':28s}{'planned':>10s}{'flown (Gazebo)':>16s}")
    for k in ("coverage_pct", "angle_dev_target_deg", "obs_distance_m",
              "visual_stability_pct", "smoothness_deg_per_frame", "min_standoff_m",
              "distance_m"):
        print(f"{k:28s}{planned[k]:10.2f}{flown[k]:16.2f}")
    print(f"tracking error rms {s['pos_err_rms_m']:.2f} m, max {s['pos_err_max_m']:.2f} m; "
          f"yaw error max {s['yaw_err_max_deg']:.1f} deg")
    out, _ = animate({"Optimized, ArduPilot SITL + Gazebo": log}, b,
                     f"results/{tag}.mp4", planned={"Optimized, ArduPilot SITL + Gazebo": path},
                     title=f"{name}, flown on ArduCopter 4.7.2 SITL + Gazebo Harmonic")
    print(f"wrote {out}, results/{tag}_flightlog.csv")


if __name__ == "__main__":
    main()
