import os
import time
import subprocess

from stable_baselines3.common.env_checker import check_env

from quadrotor_sim.envs.quadrotor_hover_env import QuadrotorHoverEnv


def main():
    proc = subprocess.Popen(
        ["ros2", "launch", "quadrotor_sim", "quadrotor.launch.py", "gz_args:=-s -r"],
        env=os.environ.copy(),
    )
    time.sleep(10.0)
    env = QuadrotorHoverEnv()
    print("[smoke] running check_env")
    check_env(env, warn=True)
    print("[smoke] check_env done")
    env.close()
    proc.terminate()
    proc.wait(timeout=10)


if __name__ == "__main__":
    main()

