import os
os.environ["SDL_VIDEODRIVER"] = "windib" 

from snake_env import SnakeEnv
from dqn_agent import DQNAgent
import torch

env   = SnakeEnv(render_mode="human")
agent = DQNAgent()
agent.load("DQN 1M Runs Baseline/dqn_final.pt") # runs the final trained model from the 1M runs baseline
agent.policy_net.eval()

obs, _ = env.reset()
while True:
    with torch.no_grad():
        action = agent.select_action(obs)
    obs, _, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Score: {info['score']}")
        obs, _ = env.reset()