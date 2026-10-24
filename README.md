# UAV inspection trajectory optimization

Extends Gottumukkala & Murphy, *"Spline-path based autonomous infrastructure
inspection using an unmanned aerial vehicle"*: instead of a hand-placed spline
through pilot photo points, the spline waypoints and camera yaw are chosen by
minimizing an objective J built from the paper's own evaluation metrics.
`CLAUDE.md` has the full project notes; `objective.py`'s docstring has the math.

| File | What it does |
|---|---|
| `objective.py` | Building geometry, paths, paper metrics (`evaluate`), objective `J` |
| `demo.py` | Orbit / naive spline / optimized on U, L, rectangle; reference optimizer |
| `sim.py` | Kinematic flight simulator (speed/accel/yaw-rate limits, wind) + `FlownPath` adapter |
| `animate.py` | Side-by-side MP4/GIF: FOV wedge, center ray, sub-edges red → green |
| `experiments.py` | 7 building types × seeds, synthetic pilots, paired t-tests |
| `gazebo_demo.py` | Fly the optimized path on ArduPilot SITL + Gazebo Harmonic |
| `tests/` | pytest: geometry, simulator, animation |

Synthetic buildings and waypoints only; the paper's 28 Hurricane Irma targets
are not available.

## Python setup

```bash
micromamba create -n drone -c conda-forge python=3.12 numpy scipy matplotlib pytest ffmpeg
micromamba activate drone
python demo.py            # metrics table + comparison.png
python sim.py             # planned vs flown (calm, wind) -> results/
python animate.py         # results/anim_*.mp4
python experiments.py     # results/experiments*.csv, t-tests
pytest -q tests
```

## Gazebo + ArduPilot SITL (optional)

Built user-space (no sudo) on Ubuntu 26.04 / aarch64:

```bash
micromamba create -n sitl -c conda-forge python=3.12 gz-sim8=8.10 gz-tools2 gz-rendering8=8.2.2 \
  compilers cmake ninja make pkg-config rapidjson numpy scipy matplotlib pytest ffmpeg \
  pexpect future lxml pip ruby libopencv=4.13 gstreamer gst-plugins-base
micromamba run -n sitl pip install pymavlink "empy==3.3.4" dronecan

mkdir -p ~/sitl && cd ~/sitl
git clone --depth 1 --branch Copter-4.7.2 --recurse-submodules --shallow-submodules \
  https://github.com/ArduPilot/ardupilot.git
git clone --depth 1 https://github.com/ArduPilot/ardupilot_gazebo.git
cp <this repo>/sitl/env.sh ~/sitl/env.sh && source ~/sitl/env.sh
(cd ardupilot && ./waf configure --board sitl && ./waf copter)
(cd ardupilot_gazebo && CMAKE_PREFIX_PATH=$CONDA_PREFIX cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build)

source ~/sitl/env.sh
python gazebo_demo.py --building U          # headless; add --gui to watch
```

Gotchas found while setting this up:
- conda-forge `libsystemd0` ≥ 261 needs a glibc symbol (`removexattrat`) this
  host lacks, so `libArduPilotPlugin.so` silently fails to load and SITL waits
  forever for physics. `env.sh` and `gazebo_demo.py` preload the system copy.
- With the JSON backend ArduPilot's local origin is Gazebo's world origin, not
  the spawn point; `gazebo_demo.py` measures the offset after GPS lock and
  only arms once the EKF is using GPS.
- The GUI needs `GZ_RENDERING_RESOURCE_PATH` (set in `env.sh`); on Wayland run
  it with `QT_QPA_PLATFORM=xcb`.
