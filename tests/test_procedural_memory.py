"""Tests for ProceduralMemory — learning from experience."""

from __future__ import annotations

from askme.brain.procedural_memory import ProceduralMemory, Procedure


def test_record_outcome_creates_procedure(tmp_path):
    """record_outcome creates a new procedure when name is unseen."""
    pm = ProceduralMemory(data_dir=str(tmp_path))
    pm.record_outcome("route_north", "navigate", success=True, duration=60.0,
                       description="North corridor route")
    procs = pm.get_procedures("navigate")
    assert len(procs) == 1
    assert procs[0].name == "route_north"
    assert procs[0].description == "North corridor route"


def test_record_outcome_updates_beta_posterior(tmp_path):
    """Successes increment alpha, failures increment beta."""
    pm = ProceduralMemory(data_dir=str(tmp_path))
    pm.record_outcome("route_a", "navigate", success=True)
    pm.record_outcome("route_a", "navigate", success=True)
    pm.record_outcome("route_a", "navigate", success=False)
    proc = pm.get_procedures("navigate")[0]
    # Initial alpha=1, +2 successes = 3
    assert proc.alpha == 3.0
    # Initial beta=1, +1 failure = 2
    assert proc.beta == 2.0
    assert proc.total_uses == 3


def test_success_rate_computed_correctly(tmp_path):
    """success_rate = alpha / (alpha + beta)."""
    pm = ProceduralMemory(data_dir=str(tmp_path))
    # 4 successes: alpha = 1+4 = 5, beta = 1 (prior)
    for _ in range(4):
        pm.record_outcome("good_route", "patrol", success=True)
    proc = pm.get_procedures("patrol")[0]
    expected = 5.0 / 6.0  # alpha=5, beta=1
    assert abs(proc.success_rate - expected) < 1e-9


def test_confidence_increases_with_data(tmp_path):
    """confidence grows as total_uses increases."""
    pm = ProceduralMemory(data_dir=str(tmp_path))
    pm.record_outcome("proc_x", "inspect", success=True)
    c1 = pm.get_procedures("inspect")[0].confidence
    for _ in range(9):
        pm.record_outcome("proc_x", "inspect", success=True)
    c10 = pm.get_procedures("inspect")[0].confidence
    assert c10 > c1
    # confidence = 1 - 1/(1+n); at n=10 -> 1 - 1/11 ~ 0.909
    assert abs(c10 - (1.0 - 1.0 / 11.0)) < 1e-9


def test_get_best_procedure_returns_highest(tmp_path):
    """get_best_procedure returns the procedure with best score."""
    pm = ProceduralMemory(data_dir=str(tmp_path))
    # Good route: 5 successes, 0 failures
    for _ in range(5):
        pm.record_outcome("good", "navigate", success=True)
    # Bad route: 1 success, 4 failures
    pm.record_outcome("bad", "navigate", success=True)
    for _ in range(4):
        pm.record_outcome("bad", "navigate", success=False)
    best = pm.get_best_procedure("navigate")
    assert best is not None
    assert best.name == "good"


def test_get_best_procedure_min_confidence(tmp_path):
    """get_best_procedure filters out low-confidence procedures."""
    pm = ProceduralMemory(data_dir=str(tmp_path))
    # Only 1 use: confidence = 1 - 1/(1+1) = 0.5
    pm.record_outcome("one_shot", "navigate", success=True)
    # Require high confidence
    best = pm.get_best_procedure("navigate", min_confidence=0.8)
    assert best is None
    # Lower threshold should find it
    best = pm.get_best_procedure("navigate", min_confidence=0.3)
    assert best is not None


def test_get_procedures_filtered_by_type(tmp_path):
    """get_procedures with task_type returns only matching procedures."""
    pm = ProceduralMemory(data_dir=str(tmp_path))
    pm.record_outcome("nav1", "navigate", success=True)
    pm.record_outcome("insp1", "inspect", success=True)
    pm.record_outcome("nav2", "navigate", success=False)
    nav_procs = pm.get_procedures("navigate")
    assert len(nav_procs) == 2
    assert all(p.task_type == "navigate" for p in nav_procs)
    all_procs = pm.get_procedures()
    assert len(all_procs) == 3


def test_persistence_save_load(tmp_path):
    """Procedures survive save/load round-trip."""
    pm1 = ProceduralMemory(data_dir=str(tmp_path))
    pm1.record_outcome("route_a", "navigate", success=True, duration=120.0,
                        description="Via north corridor")
    pm1.record_outcome("route_a", "navigate", success=False, duration=0.0)

    # Fresh instance from same dir
    pm2 = ProceduralMemory(data_dir=str(tmp_path))
    procs = pm2.get_procedures("navigate")
    assert len(procs) == 1
    proc = procs[0]
    assert proc.name == "route_a"
    assert proc.alpha == 2.0   # 1 prior + 1 success
    assert proc.beta == 2.0    # 1 prior + 1 failure
    assert proc.total_uses == 2
    assert proc.description == "Via north corridor"


def test_get_context_returns_string(tmp_path):
    """get_context returns formatted string with procedure info."""
    pm = ProceduralMemory(data_dir=str(tmp_path))
    pm.record_outcome("patrol_east", "patrol", success=True,
                       description="East wing patrol")
    ctx = pm.get_context()
    assert isinstance(ctx, str)
    assert "patrol_east" in ctx
    assert "patrol" in ctx


def test_get_context_empty(tmp_path):
    """get_context returns empty string when no procedures exist."""
    pm = ProceduralMemory(data_dir=str(tmp_path))
    assert pm.get_context() == ""
