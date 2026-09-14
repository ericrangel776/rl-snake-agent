"""
snake_env.py
------------
Wraps SnakeGame as a Gymnasium environment so it can be used
with any RL library (Stable Baselines3, CleanRL, custom DQN, etc.).

Gymnasium contract:
    observation_space  - what the agent sees each step
    action_space       - what actions the agent can take
    reset()            - start new episode, return (obs, info)
    step(action)       - apply action, return (obs, reward, terminated, truncated, info)
    render()           - optional visual output
    close()            - cleanup
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from snake_game import SnakeGame

# Optional reward shaping variants from the proposal
REWARD_VARIANT = "baseline"   # "baseline" | "shaped" | "loop_penalty"


class SnakeEnv(gym.Env):
    """
    Gymnasium environment for the Snake game.

    Observation space : Box(11,) – binary float32 vector (see SnakeGame.get_state)
    Action space      : Discrete(3) – 0=straight, 1=right, 2=left
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 15}

    def __init__(self, render_mode: str = "none",
                 reward_variant: str = REWARD_VARIANT,
                 seed: int | None = None):
        super().__init__()

        self.reward_variant = reward_variant
        self._seed          = seed

        # -- Spaces ----------------------------------------------------
        # 11-dimensional binary observation exactly as the proposal describes
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(11,), dtype=np.float32
        )
        # Three relative actions: straight, right-turn, left-turn
        self.action_space = spaces.Discrete(3)

        # -- Underlying game -------------------------------------------
        render = (render_mode == "human")
        self.game = SnakeGame(render=render, seed=seed)

        # For shaped reward: track previous distance to food
        self._prev_dist = None
        # For loop penalty: track recently visited cells
        self._visit_window: list[tuple] = []

    # -- reset --------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        obs = self.game.get_state()

        # Initialise shaping helpers
        self._prev_dist    = self._food_dist()
        self._visit_window = []

        return obs, {}

    # -- step ----------------------------------------------------------
    def step(self, action: int):
        reward_base, terminated, score = self.game.step(int(action))

        # -- Reward shaping ----------------------------------------------
        reward = reward_base

        if self.reward_variant == "shaped":
            # Small dense signal: +1 if moved closer, -1 if moved farther
            new_dist = self._food_dist()
            if self._prev_dist is not None and reward_base == 0.0:
                reward += 1.0 if new_dist < self._prev_dist else -1.0
            self._prev_dist = new_dist

        elif self.reward_variant == "loop_penalty":
            # Penalise revisiting cells within the last 20 steps
            head = self.game.snake[0]
            self._visit_window.append((head.x, head.y))
            if len(self._visit_window) > 20:
                self._visit_window.pop(0)
            if self._visit_window.count((head.x, head.y)) > 1 and reward_base == 0.0:
                reward -= 1.0

        obs       = self.game.get_state()
        truncated = False                        # termination handled in game
        info      = {"score": score}

        return obs, reward, terminated, truncated, info

    # -- render / close -------------------------------------------------
    def render(self):
        pass   # Pygame renders inside game.step() when render=True

    def close(self):
        self.game.close()

    # -- helpers ---------------------------------------------------------
    def _food_dist(self) -> float:
        """Manhattan distance from snake head to food."""
        h = self.game.snake[0]
        f = self.game.food
        return abs(h.x - f.x) + abs(h.y - f.y)


# -- Registration with Gymnasium ------------------------------------------
# Lets you create the env with gym.make("Snake-v0") anywhere in your code
gym.register(
    id="Snake-v0",
    entry_point="snake_env:SnakeEnv",
    max_episode_steps=10_000,
)


# -- Smoke test ------------------------------------------------------
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = SnakeEnv()
    print("Running Gymnasium env checker ...")
    check_env(env, warn=True)
    print("Environment passed all Gymnasium checks.\n")

    # Random agent rollout
    obs, _ = env.reset()
    total_reward = 0.0
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"  Episode ended at step {step} | score={info['score']} | "
                  f"total_reward={total_reward:.1f}")
            obs, _ = env.reset()
            total_reward = 0.0

    env.close()
    print("Smoke test complete.")
