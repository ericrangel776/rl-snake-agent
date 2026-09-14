# Reinforcement Learning in a Snake Game

**Course:** CPSC 425-01  
**Instructor:** Dr. Banweer  
**Team Member:** Eric Rangel  
**Date:** Spring 2026

---

## Problem Description

This project trains a reinforcement learning agent to play the classic Snake game from scratch using only a reward signal — no labeled data, no human demonstrations. The agent starts with zero knowledge of the game and learns entirely through trial and error.

The game is built in Pygame on a 20×20 grid and wrapped as a standard Gymnasium environment. At each step the agent observes an 11-dimensional binary state vector and selects one of three relative actions (go straight, turn right, turn left). A reward of +10 is given for eating food and −10 for dying.

Two algorithms are implemented and compared:
- **DQN (Deep Q-Network)** — custom implementation with experience replay and a target network
- **PPO (Proximal Policy Optimization)** — via Stable Baselines3

Three reward variants are tested across both algorithms:
- **Baseline** — sparse +10 / −10 only
- **Shaped** — adds ±1 per step based on distance to food
- **Loop Penalty** — adds −1 for revisiting cells within 20 steps

---

## Project Structure

```
snake_rl/
│
├── snake_game.py       # Core Pygame Snake game logic
├── snake_env.py        # Gymnasium environment wrapper
├── dqn_agent.py        # DQN neural network, replay buffer, agent
├── train.py            # Training loop for both DQN and PPO
├── watch.py            # Watch a trained DQN agent play
├── watch_ppo.py        # Watch a trained PPO agent play
├── plot_ppo.py         # Plot PPO learning curves from saved logs
├── requirements.txt    # All required dependencies
└── README.md           # This file
```

---

## Required Libraries and Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install torch>=2.2.0
pip install gymnasium>=0.29.0
pip install stable-baselines3>=2.3.0
pip install pygame>=2.5.0
pip install numpy>=1.26.0
pip install matplotlib>=3.8.0
```

> **Note on PyTorch:** If you do not have a GPU, install the CPU-only version of PyTorch instead (smaller download, sufficient for this project):
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```
> Then install the remaining dependencies from `requirements.txt`.

**Python version:** 3.10 or higher is recommended. The project was developed and tested on Python 3.13.

---

## Running the Code

### Step 1 — Verify the environment

Before training, confirm the Gymnasium environment passes all checks:

```bash
python snake_env.py
```

Expected output:
```
✓ Environment passed all Gymnasium checks.
Smoke test complete.
```

### Step 2 — Play the game manually (optional)

Verify the Pygame rendering works and play a quick game yourself:

```bash
python snake_game.py
```

Use arrow keys to control the snake. Close the window to exit.

### Step 3 — Run a short sanity check

Before committing to a full training run, do a quick 50k-step test:

```bash
python train.py --steps 50000
```

If the mean score increases and the loss is non-zero by step 30k, everything is working correctly.

### Step 4 — Train DQN

**Baseline reward (1 million steps):**
```bash
python train.py --algo dqn --variant baseline --steps 1000000 --seed 42
```

**Shaped reward:**
```bash
python train.py --algo dqn --variant shaped --steps 1000000 --seed 42
```

**Loop penalty reward:**
```bash
python train.py --algo dqn --variant loop_penalty --steps 1000000 --seed 42
```

Each run saves:
- `dqn_final.pt` — final trained model weights
- `checkpoint_N.pt` — checkpoints every 10,000 steps
- `learning_curve_dqn_<variant>_seed<N>.png` — learning curve plot

### Step 5 — Train PPO

```bash
python train.py --algo ppo --variant baseline --steps 1000000 --seed 42
python train.py --algo ppo --variant shaped --steps 1000000 --seed 42
python train.py --algo ppo --variant loop_penalty --steps 1000000 --seed 42
```

Each run saves:
- `ppo_final.zip` — final trained model
- `ppo_best/best_model.zip` — best checkpoint during training
- `learning_curve_ppo_<variant>_seed<N>.png` — learning curve plot

### Step 6 — Watch a trained agent play

**DQN:**
```bash
python watch.py
```

**PPO:**
```bash
python watch_ppo.py
```

Both scripts open the Pygame window and show the agent playing in real time. The score is printed to the terminal at the end of each episode.

> **Important:** Place the saved model file (`dqn_final.pt` or `ppo_final.zip`) in the same directory as the watch scripts before running them.

### Optional — Render during training (slow)

```bash
python train.py --render --steps 5000
```

Opens the Pygame window and shows the agent playing live during training. Not recommended for full runs as it significantly slows training.

---

## Command-Line Arguments (train.py)

| Argument | Default | Options | Description |
|---|---|---|---|
| `--algo` | `dqn` | `dqn`, `ppo` | Algorithm to train |
| `--variant` | `baseline` | `baseline`, `shaped`, `loop_penalty` | Reward function |
| `--steps` | `1000000` | any int | Total training steps |
| `--seed` | `42` | any int | Random seed |
| `--eval_every` | `10000` | any int | Logging interval |
| `--render` | off | flag | Show Pygame window during training |

---

## Dataset and Environment

This project does not use an external dataset. The agent generates its own training data by interacting with the Snake game environment.

**Environment details:**
- Grid size: 20×20 cells
- Observation space: `Box(11,)` — binary float32 vector
- Action space: `Discrete(3)` — straight (0), right turn (1), left turn (2)
- Episode termination: wall collision, self-collision, or exceeding 100 × snake length steps
- Food placement: random, seeded via numpy RNG

**State vector layout:**

| Index | Meaning |
|---|---|
| 0 | Danger straight ahead |
| 1 | Danger to the right |
| 2 | Danger to the left |
| 3 | Currently moving right |
| 4 | Currently moving down |
| 5 | Currently moving left |
| 6 | Currently moving up |
| 7 | Food is to the left |
| 8 | Food is to the right |
| 9 | Food is above |
| 10 | Food is below |

No preprocessing is applied — the state is computed directly from the game state at each step and passed to the agent as-is.

---

## Key Results

| Algorithm | Variant | Stable Score | Peak Score | Converged |
|---|---|---|---|---|
| DQN | Baseline | 22–25 | ~26 | Yes (1M steps) |
| DQN | Shaped | 23–25 | ~25 | Yes (1M steps) |
| DQN | Loop Penalty | 21–23 | ~23 | Yes (1M steps) |
| PPO | Baseline | 28–35 | ~41 | Trending up |
| PPO | Shaped | 28–42 | ~42 | Trending up |
| PPO | Loop Penalty | 30–46 | ~46 | Trending up |

Both algorithms surpassed the project success threshold of 15+ pellets per episode.

---

## Known Issues and Limitations

- All runs used a single random seed (42). Multi-seed averaging was not completed.
- PPO runs had not fully converged by 1M steps — results are mid-training snapshots.
- The 11-value binary state representation has no spatial memory of the snake's body, creating an architectural performance ceiling for both algorithms.
- The Pygame window requires a display. On headless machines, set `render_mode="none"` (the default during training).

---

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Pygame Documentation](https://www.pygame.org/docs/)
- [Mnih et al. (2015) — Human-level control through deep RL (DQN)](https://www.nature.com/articles/nature14236)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io)
- [Patrick Loeber — Snake AI PyTorch](https://github.com/patrickloeber/snake-ai-pytorch)