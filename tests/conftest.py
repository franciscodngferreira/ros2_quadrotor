"""Import paths, plus ROS stubs so CI needs no ROS install.

`quadrotor_hover_env` imports rclpy/gz at module scope, but these checks are pure
math on a bare env and never touch them. Real modules win when importable, so the
stub can only ever apply where ROS is genuinely absent.
"""
import importlib
import os
import sys
from unittest.mock import MagicMock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIST_PACKAGES = "/usr/lib/python3/dist-packages"  # where the gz debs live; rl_venv can't see it

for _p in (os.path.join(_ROOT, "scripts"), os.path.join(_ROOT, "src", "quadrotor_sim")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if os.path.isdir(_DIST_PACKAGES) and _DIST_PACKAGES not in sys.path:
    sys.path.append(_DIST_PACKAGES)

_ROS_MODULES = (
    "rclpy", "rclpy.executors", "rclpy.context", "rclpy.qos",
    "geometry_msgs", "geometry_msgs.msg",
    "nav_msgs", "nav_msgs.msg",
    "sensor_msgs", "sensor_msgs.msg",
    "std_msgs", "std_msgs.msg",
    "gz", "gz.transport13", "gz.msgs10",
    "gz.msgs10.world_control_pb2", "gz.msgs10.boolean_pb2", "gz.msgs10.pose_pb2",
)


def _real_ros_importable():
    for name in _ROS_MODULES:
        try:
            importlib.import_module(name)
        except Exception:
            return False
    return True


ROS_AVAILABLE = _real_ros_importable()

if not ROS_AVAILABLE:
    for _name in _ROS_MODULES:
        # No spec=: `from X import Y` must resolve arbitrary attributes.
        sys.modules.setdefault(_name, MagicMock())


def pytest_report_header(config):
    return "quadrotor_sim: " + ("real ROS" if ROS_AVAILABLE else "stubbed ROS (pure-math checks)")
