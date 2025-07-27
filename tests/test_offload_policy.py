import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.offload_policy import OffloadPolicy


def test_latency_score_respects_threshold():
    base = 10.0
    depth = 10
    small = OffloadPolicy(queue_threshold=5)
    large = OffloadPolicy(queue_threshold=20)

    score_small = small.latency_score(depth, base)
    score_large = large.latency_score(depth, base)

    # higher threshold should reduce queue contribution
    assert score_small > score_large
    # queue contribution is capped at 1.0 when depth >= threshold
    assert score_small - base == 1.0
    assert score_large - base == 0.5


def test_latency_score_cap():
    pol = OffloadPolicy(queue_threshold=3)
    # depth far exceeding threshold should still only add 1.0
    assert pol.latency_score(10, 5.0) == 6.0
