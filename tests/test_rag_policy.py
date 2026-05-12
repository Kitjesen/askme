from askme.pipeline.rag_policy import forced_rag_reply, is_rag_policy_blocking


def test_rag_policy_blocks_stale_conflict_and_unapproved_states() -> None:
    for state in ("stale", "conflict", "unapproved", "filtered"):
        assert is_rag_policy_blocking({"state": state, "action": "clarify"}) is True
        assert forced_rag_reply({"state": state, "action": "clarify"})


def test_rag_policy_only_blocks_no_evidence_on_explicit_refusal() -> None:
    assert is_rag_policy_blocking({"state": "no_evidence", "action": "clarify_or_refuse"}) is False
    assert forced_rag_reply({"state": "no_evidence", "action": "clarify_or_refuse"}) == ""
    assert is_rag_policy_blocking({"state": "no_evidence", "action": "refuse"}) is True
    assert "没有可靠依据" in forced_rag_reply({"state": "no_evidence", "action": "refuse"})


def test_rag_policy_uses_surface_template_override() -> None:
    assert forced_rag_reply(
        {"state": "conflict", "action": "clarify"},
        templates={"conflict": "请管理员确认后再回答。"},
    ) == "请管理员确认后再回答。"
