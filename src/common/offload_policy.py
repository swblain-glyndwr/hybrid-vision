"""Decision policies for shifting computation between edge and cloud."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Tuple, List

import numpy as np


class RLDecisionPolicy:
    """Simple tabular Q-learning policy."""

    def __init__(
        self,
        actions: Tuple[str, str] = ("edge", "cloud"),
        learning_rate: float = 0.1,
        discount: float = 0.9,
        epsilon: float = 0.1,
    ) -> None:
        self.actions: List[str] = list(actions)
        self.q = defaultdict(lambda: np.zeros(len(self.actions)))
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon

    @staticmethod
    def _state_key(metrics: Iterable[float]) -> Tuple[float, ...]:
        return tuple(round(float(m), 2) for m in metrics)

    def choose(self, metrics: Iterable[float]) -> str:
        key = self._state_key(metrics)
        if random.random() < self.epsilon or key not in self.q:
            return random.choice(self.actions)
        return self.actions[int(np.argmax(self.q[key]))]

    def update(
        self,
        metrics: Iterable[float],
        action: str,
        reward: float,
        next_metrics: Iterable[float],
    ) -> None:
        key = self._state_key(metrics)
        next_key = self._state_key(next_metrics)
        a = self.actions.index(action)
        best_next = np.max(self.q[next_key])
        self.q[key][a] += self.alpha * (
            reward + self.gamma * best_next - self.q[key][a]
        )


class QuickDecisionPolicy:
    """Threshold-based decision policy."""

    def __init__(
        self,
        latency_threshold: float = 100.0,
        queue_threshold: int = 10,
        bandwidth_threshold: float | None = None,
    ) -> None:
        self.latency_threshold = latency_threshold
        self.queue_threshold = queue_threshold
        self.bandwidth_threshold = bandwidth_threshold

    def choose(
        self,
        infer_ms: float,
        enc_ms: float,
        queue_depth: int | None = None,
        bandwidth: float | None = None,
    ) -> str:
        score = infer_ms + enc_ms
        if queue_depth is not None:
            depth = queue_depth
            # cap queue contribution using the configured threshold
            if self.queue_threshold is not None:
                depth = min(queue_depth, self.queue_threshold)
            score += depth * 2
        if bandwidth is not None and self.bandwidth_threshold is not None:
            if bandwidth < self.bandwidth_threshold:
                score += self.latency_threshold
        return "cloud" if score > self.latency_threshold else "edge"
