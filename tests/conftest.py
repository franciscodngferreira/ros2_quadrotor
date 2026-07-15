"""Test setup: import paths, and ROS stubs so CI needs no ROS install.

`quadrotor_hover_env` imports rclpy / gz.transport at module scope, but the
checks in this suite are pure math — they build a bare env via `__new__` and
never touch a node, a topic or a Gazebo service. Without stubbing, running the
tests would need a full ROS Jazzy + gz install to reach code that does
arithmetic on numpy arrays.

So: if the real modules are importable (a dev machine with ROS sourced), use
them and the real import path gets exercised. If they are not (GitHub Actions),
stub only the modules the env imports but these tests don't use. The stub is a
fallback, never an override — it cannot mask a breakage in any environment that
has ROS.

What this suite does NOT cover: anything touching the live sim. That surface is
`scripts/check_env_smoke.py` and `train/test_reset_timing.py`, both of which
need a running Gazebo and so cannot run in CI.
"""
import os
import sys
from unittest.mock import MagicMock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for _p in (os.path.join(_ROOT, "scripts"), os.path.join(_ROOT, "src", "quadrotor_sim")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# gz.transport13 / gz.msgs10 are Debian packages under /usr/lib/python3/dist-packages,
# which rl_venv does not see (created without --system-site-packages). The env module
# appends this itself at import time; mirror it here so the probe below reflects what
# the env will actually manage to import, rather than stubbing a ROS box unnecessarily.
_DIST_PACKAGES = "/usr/lib/python3/dist-packages"
if os.path.isdir(_DIST_PACKAGES) and _DIST_PACKAGES not in sys.path:
    sys.path.append(_DIST_PACKAGES)

# Every module quadrotor_hover_env imports at module scope, and none of which the
# pure-math paths under test actually call.
_ROS_MODULES = (
    "rclpy",
    "rclpy.executors",
    "rclpy.context",
    "rclpy.qos",
    "geometry_msgs",
    "geometry_msgs.msg",
    "nav_msgs",
    "nav_msgs.msg",
    "sensor_msgs",
    "sensor_msgs.msg",
    "std_msgs",
    "std_msgs.msg",
    "gz",
    "gz.transport13",
    "gz.msgs10",
    "gz.msgs10.world_control_pb2",
    "gz.msgs10.boolean_pb2",
    "gz.msgs10.pose_pb2",
)


def _probe_real_ros():
    """True only if every module the env needs is genuinely importable."""
    import importlib

    for name in _ROS_MODULES:
        try:
            importlib.import_module(name)
        except Exception:
            return False
    return True


ROS_AVAILABLE = _probe_real_ros()

if not ROS_AVAILABLE:
    for _name in _ROS_MODULES:
        # No spec= : these stand in for modules whose attributes (qos_profile_sensor_data,
        # Twist, WorldControl, ...) are looked up by `from X import Y`, and a spec'd mock
        # would reject every one of them.
        sys.modules.setdefault(_name, MagicMock())


def pytest_report_header(config):
    mode = "real ROS modules" if ROS_AVAILABLE else "stubbed ROS (pure-math checks only)"
    return f"quadrotor_sim: {mode}"
