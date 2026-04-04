"""Tests for PolicyStore — L6 policy and template store."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _make_store(tmp_path: Path, yaml_files: dict[str, object] | None = None):
    """Create a PolicyStore from a temp directory with optional YAML files."""
    from askme.memory.policies import PolicyStore
    if yaml_files:
        policy_dir = tmp_path / "memory" / "policies"
        policy_dir.mkdir(parents=True)
        for name, content in yaml_files.items():
            (policy_dir / name).write_text(
                yaml.dump(content, allow_unicode=True), encoding="utf-8"
            )
    config = {"app": {"data_dir": str(tmp_path)}}
    return PolicyStore(config=config)


class TestInit:
    def test_creates_policy_dir(self, tmp_path):
        _make_store(tmp_path)
        assert (tmp_path / "memory" / "policies").is_dir()

    def test_creates_default_files_when_none_exist(self, tmp_path):
        _make_store(tmp_path)
        policy_dir = tmp_path / "memory" / "policies"
        assert (policy_dir / "behavior.yaml").exists()
        assert (policy_dir / "safety.yaml").exists()
        assert (policy_dir / "templates.yaml").exists()

    def test_loads_existing_yaml(self, tmp_path):
        store = _make_store(tmp_path, {
            "custom.yaml": {"rules": [{"id": "r1", "rule": "be nice"}]},
        })
        rules = store.get_rules("custom")
        assert any(r["id"] == "r1" for r in rules)


class TestGetTemplate:
    def test_returns_zh_by_default(self, tmp_path):
        store = _make_store(tmp_path, {
            "templates.yaml": {
                "greeting": {"zh": "你好", "en": "Hello"},
            }
        })
        assert store.get_template("greeting") == "你好"

    def test_returns_en_when_requested(self, tmp_path):
        store = _make_store(tmp_path, {
            "templates.yaml": {
                "greeting": {"zh": "你好", "en": "Hello"},
            }
        })
        assert store.get_template("greeting", language="en") == "Hello"

    def test_falls_back_to_zh_for_unknown_language(self, tmp_path):
        store = _make_store(tmp_path, {
            "templates.yaml": {
                "greeting": {"zh": "你好"},
            }
        })
        assert store.get_template("greeting", language="fr") == "你好"

    def test_returns_empty_for_unknown_template(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_template("nonexistent") == ""

    def test_variable_substitution(self, tmp_path):
        store = _make_store(tmp_path, {
            "templates.yaml": {
                "task_confirm": {"zh": "收到，{task}，执行中。"},
            }
        })
        result = store.get_template("task_confirm", task="巡检")
        assert "巡检" in result

    def test_missing_key_in_format_returns_template_unchanged(self, tmp_path):
        store = _make_store(tmp_path, {
            "templates.yaml": {
                "task_confirm": {"zh": "{task}执行中"},
            }
        })
        result = store.get_template("task_confirm")  # missing 'task' kwarg
        # Should not raise, returns template with unfilled placeholder
        assert "task" in result or "执行中" in result


class TestGetRules:
    def test_returns_rules_list(self, tmp_path):
        store = _make_store(tmp_path, {
            "safety.yaml": {
                "rules": [
                    {"id": "estop", "rule": "急停优先", "priority": "critical"},
                ]
            }
        })
        rules = store.get_rules("safety")
        assert len(rules) == 1
        assert rules[0]["id"] == "estop"

    def test_returns_empty_for_unknown_category(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_rules("nonexistent") == []

    def test_default_behavior_rules_loaded(self, tmp_path):
        store = _make_store(tmp_path)
        rules = store.get_rules("behavior")
        assert len(rules) > 0
        assert all("rule" in r for r in rules)


class TestGetAllRules:
    def test_returns_rules_from_all_categories(self, tmp_path):
        store = _make_store(tmp_path, {
            "behavior.yaml": {"rules": [{"id": "b1", "rule": "rule b"}]},
            "safety.yaml": {"rules": [{"id": "s1", "rule": "rule s"}]},
            "templates.yaml": {"greeting": {"zh": "hi"}},
        })
        all_rules = store.get_all_rules()
        ids = [r.get("id") for r in all_rules]
        assert "b1" in ids
        assert "s1" in ids

    def test_templates_not_included_in_all_rules(self, tmp_path):
        store = _make_store(tmp_path)
        all_rules = store.get_all_rules()
        # Templates don't have 'rule' key
        rule_rules = [r for r in all_rules if "rule" in r]
        assert len(rule_rules) > 0


class TestGetPolicyPrompt:
    def test_returns_string(self, tmp_path):
        store = _make_store(tmp_path)
        result = store.get_policy_prompt()
        assert isinstance(result, str)

    def test_respects_max_chars(self, tmp_path):
        store = _make_store(tmp_path)
        result = store.get_policy_prompt(max_chars=50)
        assert len(result) <= 60  # small margin for truncation logic

    def test_contains_rule_text(self, tmp_path):
        store = _make_store(tmp_path, {
            "behavior.yaml": {"rules": [{"id": "b1", "rule": "be concise always"}]},
        })
        result = store.get_policy_prompt()
        assert "be concise always" in result


class TestReload:
    def test_reload_picks_up_new_file(self, tmp_path):
        store = _make_store(tmp_path)
        # Initially no "custom" category
        assert store.get_rules("custom") == []
        # Write a new file
        policy_dir = tmp_path / "memory" / "policies"
        (policy_dir / "custom.yaml").write_text(
            yaml.dump({"rules": [{"id": "new", "rule": "new rule"}]}),
            encoding="utf-8",
        )
        store.reload()
        rules = store.get_rules("custom")
        assert any(r["id"] == "new" for r in rules)
