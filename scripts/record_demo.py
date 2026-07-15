#!/usr/bin/env python3
"""Fly the trained policy through four fixed start -> target legs, for demo videos.

Usage — three terminals (the GUI must be visible on screen to record):

    ros2 launch quadrotor_sim quadrotor.launch.py
    gz sim -g --gui-config src/quadrotor_sim/config/demo_gui.config
    python3 scripts/record_demo.py --model checkpoints_goal/best_eval

Then click the VideoRecorder button (top-left of the GUI) to start/stop capture
around the flight. Roughly 60 s of video: 4 legs x 14 s plus resets.

Recording is manual on purpose — on this machine it is the only thing that works,
and it is not for lack of trying:
  * camera sensor + CameraVideoRecorder system -> uniformly blank frames. WSL
    exposes no /dev/dri, so the headless server renders no geometry at all
    (verified under both ogre2 and ogre1; even a camera pointed straight down at
    a 100x100 ground plane came back empty).
  * ffmpeg x11grab / mss screen capture -> pure black. WSLg composites through
    Wayland, so the X root window holds no readable content.
  * the GUI's VideoRecorder plugin renders correctly (via /dev/dxg) but
    advertises no gz-transport service, so it cannot be driven from a script.
On a machine with a real GPU / DRI node, the camera-sensor route would allow this
to run unattended.

The legs are fixed rather than random (eval_hover.py already covers random). The
policy's observation is purely relative (dx_body, dy_body, dz, ... — no absolute
position), so the drone can start anywhere without leaving the training
distribution; only the spawn->target DISPLACEMENT must stay inside what it trained
on (targets were sampled within xy +/-1.5 m of a near-origin spawn, i.e. up to
~2.5 m away). The legs below sit at ~2.0-2.3 m for exactly that reason.
"""

import argparse
import math
import os
import pickle
import sys
import time

import numpy as np


# (spawn x, y, z, yaw), (target x, y, z) — displacement ~2.0-2.3 m per leg.
DEMO_LEGS = [
    ((-2.0, -2.0, 1.0, 0.0),          (-0.2, -1.0, 1.9)),
    (( 2.0, -2.0, 1.6, math.pi / 2),  ( 0.3, -1.2, 0.8)),
    (( 2.0,  2.0, 1.0, math.pi),      ( 0.6,  0.8, 1.9)),
    ((-2.0,  2.0, 1.8, -math.pi / 2), (-0.4,  0.9, 0.9)),
]


# Red sphere marking the leg's target. Spawned at runtime rather than added to
# empty.sdf, since eval_hover.py shares that world and would inherit a stray
# marker parked at whatever the last demo leg happened to be. No <collision>:
# the drone flies through it, so it cannot perturb the episode it illustrates.
TARGET_MARKER_NAME = "demo_target"
TARGET_MARKER_SDF = """<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><sphere><radius>0.075</radius></sphere></geometry>
        <material>
          <ambient>0.9 0.05 0.05 1</ambient>
          <diffuse>0.9 0.05 0.05 1</diffuse>
          <emissive>0.35 0.0 0.0 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


def _prefer_source_tree():
    ws_src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'src', 'quadrotor_sim')
    )
    if ws_src not in sys.path:
        sys.path.insert(0, ws_src)


def _ensure_runtime_dirs():
    ws = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.environ.setdefault('ROS_HOME', os.path.join(ws, '.ros'))
    os.environ.setdefault('ROS_LOG_DIR', os.path.join(ws, 'log', 'ros'))
    os.environ.setdefault('GZ_HOME', os.path.join(ws, '.gz'))


def _spawn_target_marker(gz_node):
    """Create the red target sphere. Returns True if the world accepted it.

    Reuses the env's gz-transport node and the same /world/empty services the
    env already drives for reset teleports, so the marker needs no extra
    connection or launch-file wiring.
    """
    from gz.msgs10.entity_factory_pb2 import EntityFactory
    from gz.msgs10.boolean_pb2 import Boolean

    req = EntityFactory()
    req.sdf = TARGET_MARKER_SDF.format(name=TARGET_MARKER_NAME)
    # Off to the side and below the floor until the first leg positions it, so
    # it never appears mid-shot at the origin.
    req.pose.position.x = 0.0
    req.pose.position.y = 0.0
    req.pose.position.z = -5.0
    req.allow_renaming = False
    ok, res = gz_node.request(
        '/world/empty/create', req, EntityFactory, Boolean, 5000
    )
    return bool(ok and res.data)


def _move_target_marker(gz_node, target):
    """Teleport the marker onto this leg's target. Best-effort: a failure here
    costs the dot, not the flight, so the demo continues either way."""
    from gz.msgs10.pose_pb2 import Pose
    from gz.msgs10.boolean_pb2 import Boolean

    req = Pose()
    req.name = TARGET_MARKER_NAME
    req.position.x = float(target[0])
    req.position.y = float(target[1])
    req.position.z = float(target[2])
    req.orientation.w = 1.0
    ok, res = gz_node.request(
        '/world/empty/set_pose', req, Pose, Boolean, 5000
    )
    return bool(ok and res.data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='checkpoints_goal/best_eval')
    parser.add_argument(
        '--steps-per-leg', type=int, default=350,
        help='Steps per leg before truncation (350 * 0.04 s = 14 s of sim time)',
    )
    parser.add_argument('--settle-seconds', type=float, default=1.5)
    parser.add_argument('--control-dt', type=float, default=0.04)
    parser.add_argument(
        '--countdown', type=float, default=8.0,
        help='Seconds to wait before the first leg, to click record in the GUI',
    )
    parser.add_argument(
        '--no-target-marker', dest='target_marker', action='store_false', default=True,
        help='Skip the red target sphere (it is visual only and never collides)',
    )
    args = parser.parse_args()

    _ensure_runtime_dirs()
    _prefer_source_tree()

    from stable_baselines3 import PPO  # noqa: E402
    from quadrotor_sim.envs.quadrotor_hover_env import QuadrotorHoverEnv  # noqa: E402

    model_path = args.model if args.model.endswith('.zip') else args.model + '.zip'
    if not os.path.isfile(model_path):
        print(f"[demo] Model not found: {model_path}")
        sys.exit(1)

    print(f"[demo] Loading {model_path} ...")
    # obs_noise=False and control_dt=0.04 mirror train_hover.py; terminate_on_success
    # stays off so each leg keeps holding station on camera after arriving.
    env = QuadrotorHoverEnv(
        max_steps=args.steps_per_leg,
        randomize=True,
        obs_noise=False,
        control_dt=args.control_dt,
    )
    model = PPO.load(model_path)

    vec_normalize = None
    vecnorm_path = model_path[:-4] + '_vecnormalize.pkl'
    if os.path.isfile(vecnorm_path):
        with open(vecnorm_path, 'rb') as f:
            vec_normalize = pickle.load(f)
        print(f"[demo] Loaded VecNormalize stats from {vecnorm_path}")

    def policy_obs(raw_obs):
        if vec_normalize is None:
            return raw_obs
        return vec_normalize.normalize_obs(
            raw_obs.reshape(1, -1)
        ).reshape(-1).astype(np.float32)

    # __init__ already waited for sensors; without this the first reset skips the
    # teleport and leg 1 would start from wherever the launch spawned the drone.
    env._first_reset = False

    marker_ok = False
    if args.target_marker:
        marker_ok = _spawn_target_marker(env._gz_node)
        print(f"[demo] Target marker: {'spawned' if marker_ok else 'FAILED (continuing without it)'}",
              flush=True)

    if args.countdown > 0:
        print(f"[demo] Click record in the GUI now — starting in {args.countdown:.0f} s ...",
              flush=True)
        time.sleep(args.countdown)

    t0 = time.time()
    try:
        for i, (spawn, target) in enumerate(DEMO_LEGS, start=1):
            sx, sy, sz, syaw = spawn
            # The env samples spawn/target inside reset(); pin them for this leg.
            env._sample_spawn_pose = lambda s=spawn: (s[0], s[1], s[2], s[3], 0.0, 0.0)
            env._sample_target = lambda _x, _y, _z, t=target: np.array(t, dtype=np.float64)

            # Before reset(), so the dot is already in place when the drone
            # teleports to the start of the leg.
            if marker_ok:
                _move_target_marker(env._gz_node, target)

            obs, _ = env.reset()
            print(
                f"[demo] leg {i}/{len(DEMO_LEGS)}: start=({sx:+.1f},{sy:+.1f},{sz:.1f}) "
                f"yaw={math.degrees(syaw):+.0f} deg -> target={target} "
                f"(d={float(np.linalg.norm(obs[0:3])):.2f} m)",
                flush=True,
            )

            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                action, _ = model.predict(policy_obs(obs), deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                steps += 1

            print(f"[demo]   final_dist={float(np.linalg.norm(obs[0:3])):.3f} m "
                  f"over {steps} steps ({time.time() - t0:.0f} s wall)", flush=True)

            if args.settle_seconds > 0 and i < len(DEMO_LEGS):
                time.sleep(args.settle_seconds)
    finally:
        print(f"[demo] Done in {time.time() - t0:.0f} s — click stop in the GUI.")
        env.close()


if __name__ == '__main__':
    main()
