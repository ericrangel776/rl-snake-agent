# watch_ppo.py
import os
os.environ["SDL_VIDEODRIVER"] = "windib"   # remove if it errors on Windows

from snake_env import SnakeEnv
from stable_baselines3 import PPO

# Load whichever variant you want to watch
model = PPO.load(r"ppo_final")   # or "ppo_best/best_model" if you used EvalCallback

env = SnakeEnv(render_mode="human", reward_variant="baseline")
obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, info = env.step(int(action))
    if terminated or truncated:
        print(f"Score: {info['score']}")
        obs, _ = env.reset()