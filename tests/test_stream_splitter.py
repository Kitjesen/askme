"""Tests for StreamSplitter — sentence splitting for real-time TTS."""

from askme.voice.stream_splitter import StreamSplitter


def test_strong_punctuation_splits():
    """Strong punctuation always triggers a split."""
    s = StreamSplitter()
    assert s.feed("你好") == []
    result = s.feed("！")
    assert len(result) == 1
    assert result[0] == "你好！"


def test_question_mark_splits():
    """Question mark triggers split."""
    s = StreamSplitter()
    s.feed("你在干嘛")
    result = s.feed("？")
    assert result == ["你在干嘛？"]


def test_newline_splits():
    """Newline triggers split."""
    s = StreamSplitter()
    s.feed("第一行")
    result = s.feed("\n")
    assert result == ["第一行"]


def test_medium_punctuation_splits():
    """Semicolons and colons split after 8+ chars."""
    s = StreamSplitter()
    s.feed("这是一个较长的句子")
    result = s.feed("；")
    assert len(result) == 1
    assert "较长的句子" in result[0]


def test_medium_punctuation_no_split_short():
    """Semicolons don't split very short buffers."""
    s = StreamSplitter()
    s.feed("短")
    result = s.feed("；")
    assert result == []  # too short


def test_first_sentence_aggressive():
    """First sentence splits on comma after 5 chars when total < 40."""
    s = StreamSplitter()
    s.feed("你好你好你好")
    result = s.feed("，")
    assert len(result) == 1


def test_normal_comma_split():
    """Normal comma split after 15 chars (no comma in first chunk)."""
    s = StreamSplitter()
    s._total_chars = 50  # past first-sentence threshold
    s.feed("这是一段比较长的文本内容需要十五")  # 15 chars, no comma
    result = s.feed("，")
    assert len(result) == 1


def test_chinese_enumeration_comma():
    """Chinese enumeration comma triggers split when buffer is long enough."""
    s = StreamSplitter()
    s._total_chars = 50  # past first threshold
    s.feed("苹果和香蕉还有橘子西瓜各种水果")  # 15 chars, no comma
    result = s.feed("、")
    assert len(result) == 1


def test_emergency_split():
    """Emergency split triggers after 60 chars without punctuation."""
    s = StreamSplitter()
    # Use ASCII to avoid encoding issues; just needs > 60 chars
    long_text = "a" * 70
    assert len(long_text) > 60
    result = s.feed(long_text)
    assert len(result) >= 1


def test_emergency_split_keeps_remainder():
    """Emergency split keeps remainder in buffer for next chunk."""
    s = StreamSplitter()
    long_text = "a" * 80
    result = s.feed(long_text)
    assert len(result) == 1
    # Head should be shorter than the full text (split at ~2/3)
    assert len(result[0]) < 80
    # Remainder should be in buffer
    remainder = s.flush()
    assert remainder is not None
    assert len(result[0]) + len(remainder) == 80


def test_emergency_split_prefers_space():
    """Emergency split prefers splitting at a space boundary."""
    s = StreamSplitter()
    # Put space near the 2/3 mark (40 chars in, total 75)
    text = "x" * 40 + " word " + "y" * 29
    assert len(text) == 75
    result = s.feed(text)
    assert len(result) == 1
    # Should split at the space, not mid-word
    assert result[0].endswith("word")
    remainder = s.flush()
    assert remainder is not None
    assert remainder.startswith("y")


def test_flush_returns_remainder():
    """flush() returns buffered text."""
    s = StreamSplitter()
    s.feed("剩余文本")
    assert s.flush() == "剩余文本"


def test_flush_empty_returns_none():
    """flush() returns None when empty."""
    s = StreamSplitter()
    assert s.flush() is None


def test_reset_clears_state():
    """reset() clears buffer and counters."""
    s = StreamSplitter()
    s.feed("一些文本")
    s.reset()
    assert s.flush() is None


def test_streaming_sequence():
    """Simulate realistic LLM token stream."""
    s = StreamSplitter()
    all_sentences = []

    tokens = ["你好", "！", "我是", "Thunder", "，", "你的", "助手", "。"]
    for token in tokens:
        sentences = s.feed(token)
        all_sentences.extend(sentences)

    remainder = s.flush()
    if remainder:
        all_sentences.append(remainder)

    assert len(all_sentences) >= 2
    assert "你好！" in all_sentences[0]


def test_whitespace_only_not_emitted():
    """Whitespace-only buffers produce stripped result."""
    s = StreamSplitter()
    s.feed("  ")
    # Period after whitespace: buffer is "  。" → stripped is "。" → emits
    result = s.feed("。")
    assert result == ["。"]

    # Pure whitespace flushed returns None
    s.reset()
    s.feed("   ")
    assert s.flush() is None
