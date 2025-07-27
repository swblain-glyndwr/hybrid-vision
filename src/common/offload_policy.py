"""Simple offload policy used to decide when to process frames on the cloud.

The policy combines queue depth, network bandwidth and a base latency
measurement to produce a score. Queue depth contribution is scaled
by ``queue_threshold`` which effectively sets how many queued items are
considered acceptable. Larger thresholds reduce the impact of the queue
on the latency score, while smaller thresholds increase it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OffloadPolicy:
    queue_threshold: int = 5
    bandwidth_min_mbps: float = 10.0

    def latency_score(self, queue_depth: int, base_latency_ms: float) -> float:
        """Return latency score given a queue depth and base latency.

        The queue contribution is scaled by ``queue_threshold`` and capped at 1.0
        so that extremely long queues do not dominate the score.
        """
        if self.queue_threshold <= 0:
            # avoid divide by zero; treat as direct addition
            queue_factor = float(queue_depth)
        else:
            queue_factor = queue_depth / self.queue_threshold
        queue_factor = min(queue_factor, 1.0)
        return base_latency_ms + queue_factor

    def should_offload(
        self, queue_depth: int, base_latency_ms: float, bandwidth_mbps: float
    ) -> bool:
        """Return ``True`` if the frame should be offloaded to the cloud."""
        if bandwidth_mbps < self.bandwidth_min_mbps:
            return False
        score = self.latency_score(queue_depth, base_latency_ms)
        return score > base_latency_ms
