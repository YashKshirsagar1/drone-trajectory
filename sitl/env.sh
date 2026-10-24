# source this: environment for ArduPilot SITL + Gazebo Harmonic (user-space, micromamba env "sitl")
export SITL_ROOT=$HOME/sitl
export PATH=$HOME/micromamba/envs/sitl/bin:$PATH
export GZ_VERSION=harmonic
export GZ_SIM_SYSTEM_PLUGIN_PATH=$SITL_ROOT/ardupilot_gazebo/build${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}
export GZ_SIM_RESOURCE_PATH=$SITL_ROOT/ardupilot_gazebo/models:$SITL_ROOT/ardupilot_gazebo/worlds${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}
export ARDUCOPTER=$SITL_ROOT/ardupilot/build/sitl/bin/arducopter
export ARDUPILOT_PARAMS=$SITL_ROOT/ardupilot/Tools/autotest/default_params
# conda-forge libsystemd0 (>=261) needs removexattrat, which this glibc lacks; use the system copy
[ -e /lib/aarch64-linux-gnu/libsystemd.so.0 ] && export LD_PRELOAD=/lib/aarch64-linux-gnu/libsystemd.so.0${LD_PRELOAD:+:$LD_PRELOAD}
export GZ_RENDERING_PLUGIN_PATH=$HOME/micromamba/envs/sitl/lib/gz-rendering-8/engine-plugins
export GZ_RENDERING_RESOURCE_PATH=$HOME/micromamba/envs/sitl/share/gz/gz-rendering8
