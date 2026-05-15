"""Stable-Baselines3 callbacks for single-process Gazebo training."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


def unwrap_env(env):
    """Reach the underlying Gymnasium env through SB3 / Gym wrappers."""
    if hasattr(env, "venv"):
        env = env.venv
    if hasattr(env, "envs"):
        env = env.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


class _RolloutHookCallback(BaseCallback):
    """SB3 requires ``_on_step``; these callbacks only hook ``_on_rollout_end``."""

    def _on_step(self) -> bool:
        return True


class SaveBestRolloutMeanCallback(_RolloutHookCallback):
    """
    Save a copy of the policy when SB3's rolling mean train reward improves.

    Uses ``rollout/ep_rew_mean`` from the logger (no extra sim episodes).
    """

    def __init__(
        self,
        save_path: str,
        min_delta: float = 1.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.best_timesteps = 0

    def _on_rollout_end(self) -> bool:
        if self.logger is None:
            return True

        mean_reward = self.logger.name_to_value.get("rollout/ep_rew_mean")
        if mean_reward is None:
            return True

        timesteps = int(self.num_timesteps)
        if mean_reward > self.best_mean_reward + self.min_delta:
            self.best_mean_reward = float(mean_reward)
            self.best_timesteps = timesteps
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            self.model.save(self.save_path)
            _write_meta(
                self.save_path + "_meta.json",
                {
                    "metric": "rollout/ep_rew_mean",
                    "mean_reward": self.best_mean_reward,
                    "timesteps": self.best_timesteps,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            if self.verbose:
                print(
                    f"[callback] New best train mean reward {self.best_mean_reward:.1f} "
                    f"@ {self.best_timesteps} steps -> {self.save_path}.zip"
                )
        return True


class DeterministicEvalCallback(_RolloutHookCallback):
    """
    Run deterministic rollouts on the *same* env and save the best eval checkpoint.

    Suitable when a second eval env cannot share the Gazebo instance. Resets the
  env after evaluation so training rollouts start from a clean episode when possible.
    """

    def __init__(
        self,
        save_path: str,
        eval_freq_rollouts: int = 5,
        n_eval_episodes: int = 4,
        min_delta: float = 1.0,
        min_timesteps_before_save: int = 20_480,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        if eval_freq_rollouts < 1:
            raise ValueError("eval_freq_rollouts must be >= 1")
        self.save_path = save_path
        self.eval_freq_rollouts = eval_freq_rollouts
        self.n_eval_episodes = n_eval_episodes
        self.min_delta = min_delta
        self.min_timesteps_before_save = min_timesteps_before_save
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.best_timesteps = 0
        self.last_eval_mean: float | None = None
        self.last_eval_timesteps = 0
        self._rollout_count = 0

    def _on_rollout_end(self) -> bool:
        self._rollout_count += 1
        if self._rollout_count % self.eval_freq_rollouts != 0:
            return True

        base_env = unwrap_env(self.model.get_env())
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []

        if self.verbose:
            print(
                f"[callback] Deterministic eval ({self.n_eval_episodes} ep) "
                f"@ {self.num_timesteps} steps ..."
            )

        for ep in range(self.n_eval_episodes):
            obs, _ = base_env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, _ = base_env.step(action)
                ep_reward += float(reward)
                ep_len += 1
                done = terminated or truncated
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
            if self.verbose:
                print(
                    f"[callback]   eval ep {ep + 1}: reward={ep_reward:.1f} len={ep_len}"
                )

        mean_reward = float(np.mean(episode_rewards))
        mean_len = float(np.mean(episode_lengths))
        timesteps = int(self.num_timesteps)
        self.last_eval_mean = mean_reward
        self.last_eval_timesteps = timesteps

        # Leave training rollouts in a known state (teleport reset, zero cmd).
        try:
            base_env.reset()
        except Exception as exc:
            if self.verbose:
                print(f"[callback] Warning: post-eval reset failed: {exc!r}")

        if timesteps < self.min_timesteps_before_save:
            if self.verbose:
                print(
                    f"[callback] Eval mean={mean_reward:.1f} len={mean_len:.0f} "
                    f"(skip save until {self.min_timesteps_before_save} steps)"
                )
            return True

        if mean_reward > self.best_mean_reward + self.min_delta:
            self.best_mean_reward = mean_reward
            self.best_timesteps = timesteps
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            self.model.save(self.save_path)
            _write_meta(
                self.save_path + "_meta.json",
                {
                    "metric": "deterministic_eval_mean",
                    "mean_reward": self.best_mean_reward,
                    "timesteps": self.best_timesteps,
                    "n_eval_episodes": self.n_eval_episodes,
                    "episode_rewards": episode_rewards,
                    "episode_lengths": episode_lengths,
                    "mean_episode_length": mean_len,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            if self.verbose:
                print(
                    f"[callback] New best eval mean {self.best_mean_reward:.1f} "
                    f"(len={mean_len:.0f}) @ {self.best_timesteps} -> {self.save_path}.zip"
                )
        elif self.verbose:
            print(
                f"[callback] Eval mean={mean_reward:.1f} len={mean_len:.0f} "
                f"(best={self.best_mean_reward:.1f} @ {self.best_timesteps})"
            )

        return True


class EarlyStopNoImprovementCallback(_RolloutHookCallback):
    """Stop training when eval or train mean reward stalls (uses eval if available)."""

    def __init__(
        self,
        patience_rollouts: int = 15,
        min_delta: float = 5.0,
        use_eval_metric: bool = True,
        eval_callback: DeterministicEvalCallback | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.patience_rollouts = patience_rollouts
        self.min_delta = min_delta
        self.use_eval_metric = use_eval_metric
        self.eval_callback = eval_callback
        self._best = -np.inf
        self._wait = 0

    def _on_rollout_end(self) -> bool:
        if self.use_eval_metric and self.eval_callback is not None:
            if self.eval_callback._rollout_count % self.eval_callback.eval_freq_rollouts != 0:
                return True
            current = self.eval_callback.last_eval_mean
            if current is None:
                return True
        else:
            if self.logger is None:
                return True
            current = self.logger.name_to_value.get("rollout/ep_rew_mean")
            if current is None:
                return True

        if current > self._best + self.min_delta:
            self._best = float(current)
            self._wait = 0
            if self.verbose:
                print(f"[callback] Early-stop tracker: best={self._best:.1f}, wait=0")
        else:
            self._wait += 1
            if self.verbose:
                print(
                    f"[callback] Early-stop tracker: best={self._best:.1f}, "
                    f"wait={self._wait}/{self.patience_rollouts}"
                )
            if self._wait >= self.patience_rollouts:
                print(
                    f"[callback] Early stopping: no improvement for "
                    f"{self.patience_rollouts} rollouts."
                )
                return False
        return True


def build_checkpoint_callback(checkpoint_dir: str, save_freq: int, verbose: int = 1):
    os.makedirs(checkpoint_dir, exist_ok=True)
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix="quadrotor_hover",
        verbose=verbose,
    )


def _write_meta(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_training_summary(path: str, summary: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _write_meta(path, summary)
