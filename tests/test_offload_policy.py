import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.offload_policy import RLDecisionPolicy, QuickDecisionPolicy


def test_quick_decision_simple():
    policy = QuickDecisionPolicy(latency_threshold=50)
    action = policy.choose(infer_ms=30, enc_ms=30)
    assert action == "cloud"
    action2 = policy.choose(infer_ms=10, enc_ms=10)
    assert action2 == "edge"


def test_rl_decision_policy_update():
    policy = RLDecisionPolicy(epsilon=0.0)
    state = (10.0, 5.0)
    next_state = (8.0, 4.0)
    policy.update(state, "edge", reward=1.0, next_metrics=next_state)
    action = policy.choose(state)
    assert action in {"edge", "cloud"}


def test_queue_threshold_caps_penalty():
    policy = QuickDecisionPolicy(latency_threshold=100, queue_threshold=5)
    # Without capping, queue_depth=20 would push score above threshold.
    action = policy.choose(infer_ms=80, enc_ms=5, queue_depth=20)
    assert action == "edge"
