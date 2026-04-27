#!/usr/bin/env python3
"""GreenDyGNN RL controller for cache-window adaptation.

The agent runs at each cache-rebuild boundary. It observes per-owner miss
fractions, per-owner congestion, current cache allocation, overall hit rate,
step-time ratio, current W, and epoch progress. It emits a new window size W
and a cost-weight multiplier alpha that the prefetcher uses to bias cache
allocation toward congested owners.

Architecture: Dueling Double DQN with prioritized experience replay.
  - State    : num_owners*3 + 4  (== 13 for P=4)
  - Actions  : 7 W-deltas x 3 alpha levels  (== 21)
  - Updates  : smooth-L1 on Double-DQN target, soft target update (tau=0.08)
"""

import copy, random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

W_MIN, W_MAX = 8, 64
W_DELTAS = [-4, -2, -1, 0, +1, +2, +4]
ALPHA_LEVELS = [1.0, 3.0, 6.0]


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, num_actions, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, num_actions))

    def forward(self, x):
        s = self.shared(x)
        v = self.value(s)
        a = self.advantage(s)
        return v + a - a.mean(dim=-1, keepdim=True)


class PrioritizedReplay:
    def __init__(self, capacity=500, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer, self.priorities = [], []
        self.pos = 0

    def push(self, transition):
        max_p = max(self.priorities) if self.priorities else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_p)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return None, None, None
        probs = np.array(self.priorities[:len(self.buffer)]) ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td) + 1e-6

    def __len__(self):
        return len(self.buffer)


class GreenDyGNNAgent:
    """Dueling Double-DQN cache controller.

    State (num_owners*3 + 4 dims):
      per_owner_miss_fraction (num_owners)
      per_owner_congestion    (num_owners)
      per_owner_cache_share   (num_owners)
      cache_hit_rate, step_time_ratio, normalized_W, epoch_progress (4)

    Actions (7 * 3 = 21): W-delta (7) x alpha level (3).
    """

    def __init__(self, num_owners=3, initial_w=16, lr=3e-3, gamma=0.95,
                 buffer_size=500, batch_size=8, tau=0.08,
                 epsilon_start=0.4, epsilon_min=0.05, epsilon_decay=0.90):
        self.num_owners = num_owners
        self.current_w = initial_w
        self.current_alpha = 1.0
        self.state_dim = num_owners * 3 + 4
        self.num_w_actions = len(W_DELTAS)
        self.num_alpha_actions = len(ALPHA_LEVELS)
        self.num_actions = self.num_w_actions * self.num_alpha_actions

        self.q_net = DuelingDQN(self.state_dim, self.num_actions)
        self.target_net = copy.deepcopy(self.q_net)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.buffer = PrioritizedReplay(capacity=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.owner_miss_fractions = np.zeros(num_owners)
        self.owner_congestion = np.ones(num_owners)
        self.owner_cache_share = np.ones(num_owners) / max(1, num_owners)

        self.overall_fetch_times = deque(maxlen=200)
        self.baseline_fetch = None
        self.step_times = deque(maxlen=100)
        self.window_step_times = deque(maxlen=50)
        self.baseline_step_time = None
        self.hit_rate_accum = deque(maxlen=100)
        self.overall_hit_rate = 0.5
        self.epoch_progress = 0.0

        self._prev_state = None
        self._prev_action = None
        self._decision_count = 0
        self._total_updates = 0

    def update_owner_misses(self, owner_miss_counts, local_rank):
        total = sum(owner_miss_counts.values())
        if total == 0:
            return
        oid = 0
        for pid in sorted(owner_miss_counts.keys()):
            if pid == local_rank:
                continue
            if oid < self.num_owners:
                frac = owner_miss_counts.get(pid, 0) / max(1, total)
                self.owner_miss_fractions[oid] = 0.3 * frac + 0.7 * self.owner_miss_fractions[oid]
                oid += 1

    def update_owner_congestion_from_misses(self):
        # Under congestion the congested owner's miss fraction grows because fetches
        # from it slow down and the cache was selected without knowing which owner
        # is slow. Higher miss share => higher inferred congestion.
        if self.owner_miss_fractions.sum() < 0.01:
            self.owner_congestion = np.ones(self.num_owners)
            return
        uniform = 1.0 / max(1, self.num_owners)
        for oid in range(self.num_owners):
            ratio = self.owner_miss_fractions[oid] / max(0.01, uniform)
            self.owner_congestion[oid] = max(0.1, min(5.0, ratio))

    def record_fetch_time(self, t):
        self.overall_fetch_times.append(t)

    def record_step_time(self, t):
        self.step_times.append(t)
        self.window_step_times.append(t)

    def record_hit_rate(self, hr):
        if hr > 0:
            self.hit_rate_accum.append(hr / 100.0 if hr > 1.0 else hr)
            if len(self.hit_rate_accum) >= 3:
                self.overall_hit_rate = float(np.mean(list(self.hit_rate_accum)[-20:]))

    def calibrate_baseline(self):
        if len(self.overall_fetch_times) >= 20:
            self.baseline_fetch = float(np.percentile(list(self.overall_fetch_times), 15))
        if len(self.step_times) >= 20:
            self.baseline_step_time = float(np.percentile(list(self.step_times), 15))

    def _build_state(self):
        norm_w = (self.current_w - W_MIN) / max(1, W_MAX - W_MIN)
        step_ratio = 1.0
        if self.baseline_step_time and len(self.window_step_times) >= 2:
            step_ratio = min(5.0, float(np.median(list(self.window_step_times)))
                             / max(1e-9, self.baseline_step_time))
        return np.concatenate([
            self.owner_miss_fractions,
            self.owner_congestion,
            self.owner_cache_share,
            np.array([self.overall_hit_rate, step_ratio, norm_w, self.epoch_progress]),
        ]).astype(np.float32)

    def select_action(self):
        state = self._build_state()
        if self._prev_state is not None:
            reward = self._compute_reward()
            self.buffer.push((self._prev_state, self._prev_action, reward, state, False))
            if len(self.buffer) >= self.batch_size:
                self._update()

        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                q = self.q_net(torch.FloatTensor(state).unsqueeze(0))
                action = q.argmax(dim=-1).item()

        w_idx = action // self.num_alpha_actions
        alpha_idx = action % self.num_alpha_actions
        self.current_w = max(W_MIN, min(W_MAX, self.current_w + W_DELTAS[w_idx]))
        self.current_alpha = ALPHA_LEVELS[alpha_idx]

        self._prev_state = state
        self._prev_action = action
        self._decision_count += 1
        self.window_step_times.clear()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.current_w, self.current_alpha

    def _compute_reward(self):
        times = list(self.window_step_times) if len(self.window_step_times) >= 1 \
            else list(self.step_times)[-5:]
        if not times:
            return 0.0
        recent = float(np.mean(times))
        if self.baseline_step_time and self.baseline_step_time > 0:
            reward = -(recent / self.baseline_step_time)
        else:
            reward = -recent * 10
        if self._prev_action is not None:
            w_idx = self._prev_action // self.num_alpha_actions
            reward -= 0.01 * abs(W_DELTAS[w_idx])
        return reward

    def _update(self):
        samples, indices, is_weights = self.buffer.sample(self.batch_size)
        if samples is None:
            return
        s, a, r, s2, d = zip(*samples)
        s  = torch.FloatTensor(np.array(s))
        a  = torch.LongTensor(a)
        r  = torch.FloatTensor(r)
        s2 = torch.FloatTensor(np.array(s2))
        d  = torch.FloatTensor(d)

        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a2 = self.q_net(s2).argmax(dim=-1)
            q2 = self.target_net(s2).gather(1, a2.unsqueeze(1)).squeeze(1)
            targets = r + self.gamma * q2 * (1 - d)

        td_errors = (q - targets).detach().cpu().numpy()
        loss = (is_weights * nn.functional.smooth_l1_loss(q, targets, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_errors)
        for tp, sp in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        self._total_updates += 1

    def compute_cost_weights(self, local_rank, num_partitions):
        """Per-owner cost weights for the prefetcher (higher alpha => more cache
        pushed to congested owners). Returns None when no rebalancing is useful."""
        if self.current_alpha <= 1.0:
            return None
        if max(self.owner_congestion) < 1.15:
            return None

        weights = {}
        oid = 0
        for pid in range(num_partitions):
            if pid == local_rank:
                continue
            if oid < self.num_owners:
                weights[pid] = 1.0 + self.current_alpha * max(0, self.owner_congestion[oid] - 1.0)
                oid += 1

        total_w = sum(weights.values())
        if total_w > 0:
            oid = 0
            for pid in sorted(weights.keys()):
                if oid < self.num_owners:
                    self.owner_cache_share[oid] = weights[pid] / total_w
                    oid += 1
        return weights

    def get_stats(self):
        return {
            "w": self.current_w,
            "alpha": round(self.current_alpha, 1),
            "epsilon": round(self.epsilon, 3),
            "decisions": self._decision_count,
            "updates": self._total_updates,
            "buffer": len(self.buffer),
            "miss_frac": [round(f, 3) for f in self.owner_miss_fractions],
            "congestion": [round(c, 2) for c in self.owner_congestion],
            "hr": round(self.overall_hit_rate, 3),
        }
