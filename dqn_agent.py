"""
dqn_agent.py
------------
Deep Q-Network (DQN) agent for the Snake environment.

Implements the proposal's technical design:
  - Neural network Q-function approximator
  - Experience replay buffer
  - Target network (updated every N steps)
  - Epsilon-greedy exploration (Epsilon: 1.0 => 0.01 over 100k steps)
"""

import random
import math
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# -- Q-Network -----------------------------------------------------------------
class QNetwork(nn.Module):
    """
    Three-layer fully-connected network.
    Input  : 11-dimensional state vector
    Output : Q-value for each of the 3 actions
    """

    def __init__(self, state_dim: int = 11, action_dim: int = 3,
                 hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -- Replay Buffer -------------------------------------------------------------
class ReplayBuffer:
    """
    Circular buffer storing (state, action, reward, next_state, done) tuples.
    Random sampling breaks temporal correlations in the training data.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.stack(states)),
            torch.tensor(actions),
            torch.tensor(rewards),
            torch.tensor(np.stack(next_states)),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# -- DQN Agent -----------------------------------------------------------------
class DQNAgent:
    """
    Full DQN agent with experience replay and a target network.

    Key hyperparameters (all from the proposal):
        lr              : 1e-3    (learning rate)
        gamma           : 0.99   (discount factor)
        eps_start/end   : 1.0 / 0.01
        eps_decay_steps : 100_000
        batch_size      : 64
        target_update   : 1_000  (steps between target network syncs)
        buffer_capacity : 100_000
    """

    def __init__(
        self,
        state_dim:        int   = 11,
        action_dim:       int   = 3,
        lr:               float = 1e-3,
        gamma:            float = 0.99,
        eps_start:        float = 1.0,
        eps_end:          float = 0.01,
        eps_decay_steps:  int   = 100_000,
        batch_size:       int   = 64,
        target_update:    int   = 1_000,
        buffer_capacity:  int   = 100_000,
        device:           str   = "auto",
    ):
        self.action_dim       = action_dim
        self.gamma            = gamma
        self.batch_size       = batch_size
        self.target_update    = target_update
        self.eps_start        = eps_start
        self.eps_end          = eps_end
        self.eps_decay_steps  = eps_decay_steps
        self.steps_done       = 0

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self._sync_target()
        self.target_net.eval()          # target net is never trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)

        print(f"DQN agent ready on {self.device}")
        print(f"  Policy network: {sum(p.numel() for p in self.policy_net.parameters()):,} params")

    # -- Action selection ------------------------------------------------------
    @property
    def epsilon(self) -> float:
        """Linear epsilon decay from eps_start to eps_end over eps_decay_steps."""
        progress = min(self.steps_done / self.eps_decay_steps, 1.0)
        return self.eps_start + (self.eps_end - self.eps_start) * progress

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item())

    # -- Learning step ---------------------------------------------------------
    def learn(self) -> float | None:
        """
        Sample a mini-batch and do one gradient step on the Bellman error.
        Returns the loss value, or None if the buffer is not yet full enough.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Q(s, a) – current estimate
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # r + γ * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, target)   # Huber loss (more stable than MSE)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Periodically sync target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self._sync_target()

        return loss.item()

    # -- Target network sync ---------------------------------------------------
    def _sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # -- Save / Load -----------------------------------------------------------
    def save(self, path: str):
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, path)
        print(f"Model saved → {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.steps_done = checkpoint["steps_done"]
        self._sync_target()
        print(f"Model loaded ← {path}")
