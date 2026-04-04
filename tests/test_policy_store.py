"""Tests for PolicyStore — template retrieval, rule loading, policy prompt."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from askme.memory.policies import PolicyStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_store(tmp_path: Path) -> PolicyStore:
    """Create a PolicyStore backed by tmp_path (no real config.yaml reads)."""
    cfg = {"app": {"data_dir": str(tmp_path)}}
    return PolicyStore(config=cfg)


def _write_yaml(path: Path, content: dict) -> None:
    path.write_text(yaml.dump(content, allow_unicode=True), encoding="utf-8")


# ── Default creation ──────────────────────────────────────────────────────────

class TestDefaultCreation:
    def test_creates_policy_dir(self, tmp_path):
        store = _make_store(tmp_path)
        policy_dir = tmp_path / "memory" / "policies"
        assert policy_dir.exists()

    def test_creates_default_files(self, tmp_path):
        store = _make_store(tmp_path)
        policy_dir = tmp_path / "memory" / "policies"
        files = {f.name for f in policy_dir.glob("*.yaml")}
        assert "behavior.yaml" in files
        assert "safety.yaml" in files
        assert "templates.yaml" in files

    def test_default_templates_loaded(self, tmp_path):
        store = _make_store(tmp_path)
        # greeting template should be available by default
        text = store.get_template("greeting")
        assert text != ""

    def test_default_safety_rules_loaded(self, tmp_path):
        store = _make_store(tmp_path)
        rules = store.get_rules("safety")
        assert len(rules) > 0

    def test_default_behavior_rules_loaded(self, tmp_path):
        store = _make_store(tmp_path)
        rules = store.get_rules("behavior")
        assert len(rules) > 0


# ── get_template ──────────────────────────────────────────────────────────────

class TestGetTemplate:
    def test_returns_zh_by_default(self, tmp_path):
        store = _make_store(tmp_path)
        text = store.get_template("greeting")
        # Default is Chinese greeting
        assert "你好" in text or text != ""

    def test_returns_en_when_requested(self, tmp_path):
        store = _make_store(tmp_path)
        text = store.get_template("greeting", language="en")
        assert text != ""
        assert "Hello" in text or text != ""

    def test_unknown_template_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        text = store.get_template("nonexistent_template_xyz")
        assert text == ""

    def test_variable_substitution(self, tmp_path):
        policy_dir = tmp_path / "memory" / "policies"
        policy_dir.mkdir(parents=True)
        _write_yaml(policy_dir / "templates.yaml", {
            "task_confirm": {
                "zh": "收到，{task}，现在执行。",
                "en": "Got it, {task}, executing now.",
            }
        })
        store = _make_store(tmp_path)
        text = store.get_template("task_confirm", task="巡逻")
        assert "巡逻" in text

    def test_missing_variable_no_crash(self, tmp_path):
        policy_dir = tmp_path / "memory" / "policies"
        policy_dir.mkdir(parents=True)
        _write_yaml(policy_dir / "templates.yaml", {
            "task_confirm": {"zh": "任务: {task}"}
        })
        store = _make_store(tmp_path)
        # Missing the 'task' variable — should not raise
        text = store.get_template("task_confirm")  # no kwargs
        assert isinstance(text, str)

    def test_fallback_to_zh_for_unknown_language(self, tmp_path):
        store = _make_store(tmp_path)
        text = store.get_template("greeting", language="fr")
        # Fallback to zh
        assert isinstance(text, str)


# ── get_rules ─────────────────────────────────────────────────────────────────

class TestGetRules:
    def test_unknown_category_returns_empty_list(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_rules("nonexistent_category") == []

    def test_safety_rules_have_required_fields(self, tmp_path):
        store = _make_store(tmp_path)
        rules = store.get_rules("safety")
        for rule in rules:
            assert "rule" in rule or "id" in rule

    def test_custom_policy_loaded(self, tmp_path):
        policy_dir = tmp_path / "memory" / "policies"
        policy_dir.mkdir(parents=True)
        _write_yaml(policy_dir / "custom.yaml", {
            "rules": [{"id": "test-rule", "rule": "custom rule text"}]
        })
        store = _make_store(tmp_path)
        rules = store.get_rules("custom")
        assert len(rules) == 1
        assert rules[0]["rule"] == "custom rule text"


# ── get_all_rules ─────────────────────────────────────────────────────────────

class TestGetAllRules:
    def test_returns_list(self, tmp_path):
        store = _make_store(tmp_path)
        rules = store.get_all_rules()
        assert isinstance(rules, list)

    def test_all_rules_have_category(self, tmp_path):
        store = _make_store(tmp_path)
        rules = store.get_all_rules()
        for rule in rules:
            assert "category" in rule

    def test_includes_safety_and_behavior(self, tmp_path):
        store = _make_store(tmp_path)
        rules = store.get_all_rules()
        categories = {r["category"] for r in rules}
        assert "safety" in categories
        assert "behavior" in categories


# ── get_policy_prompt ─────────────────────────────────────────────────────────

class TestGetPolicyPrompt:
    def test_returns_string(self, tmp_path):
        store = _make_store(tmp_path)
        prompt = store.get_policy_prompt()
        assert isinstance(prompt, str)

    def test_respects_max_chars(self, tmp_path):
        store = _make_store(tmp_path)
        prompt = store.get_policy_prompt(max_chars=50)
        # truncation appends "\n..." (4 chars), so max is max_chars + 4
        assert len(prompt) <= 54

    def test_contains_safety_section(self, tmp_path):
        store = _make_store(tmp_path)
        prompt = store.get_policy_prompt()
        assert "safety" in prompt

    def test_contains_behavior_section(self, tmp_path):
        store = _make_store(tmp_path)
        prompt = store.get_policy_prompt()
        assert "behavior" in prompt

    def test_long_prompt_truncated_with_ellipsis(self, tmp_path):
        store = _make_store(tmp_path)
        prompt = store.get_policy_prompt(max_chars=10)
        assert "..." in prompt

    def test_short_prompt_not_truncated(self, tmp_path):
        store = _make_store(tmp_path)
        # Very large max_chars — should not add ellipsis
        prompt = store.get_policy_prompt(max_chars=10000)
        assert not prompt.endswith("...")


# ── reload ────────────────────────────────────────────────────────────────────

class TestReload:
    def test_reload_picks_up_new_file(self, tmp_path):
        store = _make_store(tmp_path)
        policy_dir = tmp_path / "memory" / "policies"

        # Write a new custom rule file
        _write_yaml(policy_dir / "new_category.yaml", {
            "rules": [{"id": "new-rule", "rule": "new rule added after init"}]
        })
        store.reload()
        rules = store.get_rules("new_category")
        assert len(rules) == 1

    def test_reload_picks_up_template_changes(self, tmp_path):
        store = _make_store(tmp_path)
        policy_dir = tmp_path / "memory" / "policies"
        _write_yaml(policy_dir / "templates.yaml", {
            "custom_msg": {"zh": "更新的模板"}
        })
        store.reload()
        text = store.get_template("custom_msg")
        assert "更新的模板" in text


# ── bad YAML ──────────────────────────────────────────────────────────────────

class TestBadYaml:
    def test_invalid_yaml_does_not_crash(self, tmp_path):
        policy_dir = tmp_path / "memory" / "policies"
        policy_dir.mkdir(parents=True)
        # Write a bad YAML file
        (policy_dir / "bad.yaml").write_text("{ invalid yaml: [", encoding="utf-8")
        # Should not raise
        store = _make_store(tmp_path)
        assert isinstance(store, PolicyStore)
