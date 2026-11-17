"""Pronunciation table: parse, longest match, restore, and streaming."""

from __future__ import annotations

import pytest

from askme.voice.speech.correct_words import (
    DEFAULT_CORRECT_WORDS,
    StreamCorrector,
    apply_pronunciation,
    correct_stream_chunk,
    parse_correct_words,
    resolve_correct_words,
    restore_display_text,
)

pytestmark = pytest.mark.unit


def test_empty_list_is_identity() -> None:
    table = parse_correct_words([])

    assert table.pattern is None
    assert apply_pronunciation("小算", table) == "小算"
    assert restore_display_text("小蒜", table) == "小蒜"
    assert table.skipped == ()


def test_none_and_blank_inputs_do_not_crash() -> None:
    table = parse_correct_words(None)

    assert apply_pronunciation("", table) == ""
    assert restore_display_text("", table) == ""


def test_malformed_lines_are_skipped_and_valid_lines_remain() -> None:
    table = parse_correct_words(
        [
            "no-separator",
            "|只有读法",
            "只有原文|",
            "   |   ",
            12,
            None,
            "  小算  |  小蒜  ",
            "C++|西加加",
        ]
    )

    assert table.mapping == {"小算": "小蒜", "C++": "西加加"}
    assert "no-separator" in table.skipped
    assert any("12" in item for item in table.skipped)
    assert apply_pronunciation("请说C++和小算", table) == "请说西加加和小蒜"


def test_later_duplicate_key_wins_and_reverse_collision_keeps_the_later_original() -> None:
    table = parse_correct_words(["小算|小蒜", "小算|晓算", "大算|小蒜"])

    assert table.mapping["小算"] == "晓算"
    assert table.reverse["小蒜"] == "大算"
    assert restore_display_text("小蒜", table) == "大算"


def test_longest_key_wins_and_regex_metacharacters_are_literal() -> None:
    table = parse_correct_words(
        [
            "小算|小蒜",
            "小算机器人|机器狗",
            "a.b|点",
            "(小算)|名字",
        ]
    )

    spoken = apply_pronunciation("小算机器人来了，小算好，a.b，axb，(小算)", table)

    assert spoken == "机器狗来了，小蒜好，点，axb，名字"
    assert "a.b" not in spoken
    assert "axb" in spoken


def test_restore_round_trip_for_a_complete_string() -> None:
    table = parse_correct_words(["小算|小蒜", "大算|大蒜"])
    spoken = apply_pronunciation("小算来自大算", table)

    assert spoken == "小蒜来自大蒜"
    assert restore_display_text(spoken, table) == "小算来自大算"


def test_replacement_is_not_scanned_again() -> None:
    table = parse_correct_words(["小算|小蒜", "小蒜|小算"])

    assert apply_pronunciation("小算", table) == "小蒜"
    assert apply_pronunciation("小蒜", table) == "小算"


def test_defaults_cover_product_names_without_reading_the_environment() -> None:
    table = resolve_correct_words(None, env="")

    assert table.mapping["小算"] == "小蒜"
    assert table.mapping["大算"] == "大蒜"
    assert "聚龙科创e谷" in table.mapping
    assert "巨龙科创e谷" in table.mapping
    assert set(table.mapping) == {item.split("|", 1)[0] for item in DEFAULT_CORRECT_WORDS}


def test_resolve_reads_the_process_environment_and_logs_bad_lines(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("ASKME_TTS_CORRECT_WORDS", "方总|方总")

    with caplog.at_level("WARNING"):
        from_env = resolve_correct_words([], use_defaults_when_missing=False)
        logged = resolve_correct_words(["bad-line", "小算|小蒜"], env="")

    assert from_env.mapping == {"方总": "方总"}
    assert resolve_correct_words(None, env="", use_defaults_when_missing=False).mapping == {}
    assert logged.mapping == {"小算": "小蒜"}
    assert "bad-line" in caplog.text


def test_empty_config_disables_defaults_and_env_extends_yaml() -> None:
    disabled = resolve_correct_words([], env="")
    extended = resolve_correct_words(
        ["小算|小蒜"],
        env=" 方总|方总 \nAskMe|阿斯克米 ",
    )
    from_string = resolve_correct_words("大算|大蒜;小算|小蒜", env="")

    assert disabled.mapping == {}
    assert extended.mapping == {"小算": "小蒜", "方总": "方总", "AskMe": "阿斯克米"}
    assert from_string.mapping == {"大算": "大蒜", "小算": "小蒜"}
    assert resolve_correct_words(3, env="").mapping == {}


def test_stream_chunk_holds_a_prefix_and_then_replaces() -> None:
    table = parse_correct_words(["小算|小蒜"])

    first, pending = correct_stream_chunk("你好小", table)
    second, pending = correct_stream_chunk("算到了", table, pending)

    assert first == "你好"
    assert second == "小蒜到了"
    assert pending == ""


def test_stream_flushes_a_rejected_prefix_literally() -> None:
    table = parse_correct_words(["小算|小蒜"])
    corrector = StreamCorrector(table)

    assert corrector.feed("小") == ""
    assert corrector.pending == "小"
    assert corrector.feed("好") == "小好"
    assert corrector.flush() == ""
    corrector.feed("小")
    assert corrector.flush() == "小"
    corrector.reset()
    assert corrector.pending == ""


def test_stream_prefers_waiting_for_a_longer_key() -> None:
    """Official sliding window waits while the buffer is a prefix of a longer key.

    The complete-string API still replaces the shorter key when the longer one
    does not occur. The stream helper matches the official window instead.
    """
    table = parse_correct_words(["小算|小蒜", "小算机器人|机器狗"])

    assert apply_pronunciation("小算到了", table) == "小蒜到了"
    ready, pending = correct_stream_chunk("小算到了", table)
    assert ready == "小算到了"
    assert pending == ""

    robot, pending = correct_stream_chunk("小算机器人", table)
    assert robot == "机器狗"
    assert pending == ""


def test_stream_keeps_pending_on_an_empty_chunk_and_restarts_a_new_key() -> None:
    table = parse_correct_words(["小算|小蒜", "大算|大蒜"])

    held, still_pending = correct_stream_chunk("", table, "小")
    restarted, pending = correct_stream_chunk("大算", table, still_pending)

    assert held == ""
    assert still_pending == "小"
    assert restarted == "小大蒜"
    assert pending == ""


def test_stream_without_a_table_returns_the_fragment() -> None:
    table = parse_correct_words([])

    ready, pending = correct_stream_chunk("小算", table, pending="遗留")

    assert ready == "遗留小算"
    assert pending == ""
