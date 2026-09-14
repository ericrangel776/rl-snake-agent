"""
train.py
--------
Main training loop for Phase 2 of the proposal.

Run:
    python train.py                        # DQN, baseline rewards
    python train.py --variant shaped       # DQN, shaped rewards
    python train.py --algo ppo             # PPO via Stable Baselines3
    python train.py --render               # watch the agent play (slow)
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from snake_env  import SnakeEnv
from dqn_agent  import DQNAgent

# -- CLI arguments -------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--algo",       default="dqn",      choices=["dqn", "ppo"])
parser.add_argument("--variant",    default="baseline", choices=["baseline", "shaped", "loop_penalty"])
parser.add_argument("--steps",      default=1_000_000,  type=int)
parser.add_argument("--seed",       default=42,         type=int)
parser.add_argument("--eval_every", default=10_000,     type=int)
parser.add_argument("--render",     action="store_true")
args = parser.parse_args()


# -- Helpers -------------------------------------------------------------------
def plot_curves(step_log, score_log, loss_log, eps_log, tag="dqn"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(step_log, score_log);  axes[0].set_title("Mean Score (last 100 ep)")
    axes[1].plot(step_log, loss_log);   axes[1].set_title("Mean Loss")
    axes[2].plot(step_log, eps_log);    axes[2].set_title("Epsilon")
    for ax in axes:
        ax.set_xlabel("Training steps")
    plt.tight_layout()
    fname = f"learning_curve_{tag}.png"
    plt.savefig(fname, dpi=120)
    print(f"Learning curve saved → {fname}")
    plt.close()


# -- DQN Training --------------------------------------------------------------
def train_dqn():
    render_mode = "human" if args.render else "none"
    env   = SnakeEnv(render_mode=render_mode, reward_variant=args.variant,
                     seed=args.seed)
    agent = DQNAgent()

    obs, _ = env.reset(seed=args.seed)

    # Logging
    score_window = deque(maxlen=100)
    loss_window  = deque(maxlen=1000)
    episode_score = 0.0

    step_log, score_log, loss_log, eps_log = [], [], [], []

    total_steps = 0
    ep          = 0
    t0          = time.time()

    print(f"\nTraining DQN | variant={args.variant} | steps={args.steps:,}")
    print("-" * 60)

    while total_steps < args.steps:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.buffer.push(obs, action, reward, next_obs, terminated or truncated)
        loss = agent.learn()
        if loss is not None:
            loss_window.append(loss)

        obs           = next_obs
        episode_score += reward
        total_steps   += 1

        if terminated or truncated:
            ep += 1
            score_window.append(info["score"])
            obs, _ = env.reset()
            episode_score = 0.0

        # -- Periodic evaluation log --------------------------------------
        if total_steps % args.eval_every == 0:
            mean_score = np.mean(score_window) if score_window else 0.0
            mean_loss  = np.mean(loss_window)  if loss_window  else 0.0
            elapsed    = time.time() - t0
            sps        = total_steps / elapsed   # steps per second

            step_log.append(total_steps)
            score_log.append(mean_score)
            loss_log.append(mean_loss)
            eps_log.append(agent.epsilon)

            print(f"  step={total_steps:>8,} | ep={ep:>5} | "
                  f"score={mean_score:5.2f} | loss={mean_loss:.4f} | "
                  f"ε={agent.epsilon:.3f} | {sps:.0f} sps")

            # Auto-save checkpoint
            agent.save(f"checkpoint_{total_steps}.pt")

    print("\nTraining complete.")
    agent.save("dqn_final.pt")
    env.close()

    tag = f"dqn_{args.variant}_seed{args.seed}"
    plot_curves(step_log, score_log, loss_log, eps_log, tag=tag)


# -- PPO Training (Stable Baselines3) ------------------------------------------
def train_ppo():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        print("Stable Baselines3 not installed. Run:  pip install stable-baselines3")
        return

    import numpy as np
    import matplotlib.pyplot as plt

    # -- Custom callback that tracks actual pellets eaten ------------------
    class ScoreCallback(BaseCallback):
        def __init__(self, eval_env, eval_freq=10_000, verbose=0):
            super().__init__(verbose)
            self.eval_env   = eval_env
            self.eval_freq  = eval_freq
            self.timesteps  = []
            self.scores     = []

        def _on_step(self) -> bool:
            if self.n_calls % self.eval_freq == 0:
                episode_scores = []
                for _ in range(10):          # run 10 eval episodes
                    obs, _ = self.eval_env.reset()
                    done   = False
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, _, terminated, truncated, info = self.eval_env.step(int(action))
                        done = terminated or truncated
                    episode_scores.append(info["score"])   # actual pellets eaten

                mean_score = np.mean(episode_scores)
                self.timesteps.append(self.num_timesteps)
                self.scores.append(mean_score)
                print(f"  step={self.num_timesteps:>8,} | pellets={mean_score:.2f}")
            return True

    # -- Training setup ----------------------------------------------------
    env      = make_vec_env(lambda: SnakeEnv(reward_variant=args.variant,
                                              seed=args.seed), n_envs=1) # single env wrapped for SB3
    eval_env = SnakeEnv(reward_variant=args.variant, seed=args.seed + 999)

    callback = ScoreCallback(eval_env, eval_freq=10_000)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=0,
        seed=args.seed,
    )

    print(f"\nTraining PPO | variant={args.variant} | steps={args.steps:,}")
    model.learn(total_timesteps=args.steps, callback=callback)
    model.save("ppo_final")
    print("PPO training complete. Model saved → ppo_final.zip")

    # -- Plot --------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(callback.timesteps, callback.scores)
    plt.title("PPO – Mean Pellets Eaten vs Training Steps")
    plt.xlabel("Training steps")
    plt.ylabel("Mean pellets eaten")
    plt.tight_layout()
    fname = f"learning_curve_ppo_{args.variant}_seed{args.seed}.png"
    plt.savefig(fname, dpi=120)
    print(f"Learning curve saved → {fname}")

    env.close()
    eval_env.close()


# -- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    if args.algo == "dqn":
        train_dqn()
    else:
        train_ppo()
