from askme.pipeline.external_turns import record_external_turn

from askme.conversation import VoiceTurnLedger


class _Conversation:
    def __init__(self) -> None:
        self.user_messages: list[str] = []
        self.assistant_messages: list[str] = []

    def add_user_message(self, content: str) -> None:
        self.user_messages.append(content)

    def add_assistant_message(self, content: str) -> None:
        self.assistant_messages.append(content)


class _Episodic:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    def log(self, kind: str, content: str) -> None:
        self.entries.append((kind, content))

    def should_reflect(self) -> bool:
        return False


class _Pipeline:
    def __init__(self) -> None:
        self._conversation = _Conversation()
        self._episodic = _Episodic()


def test_record_external_turn_updates_conversation_and_episodic() -> None:
    pipeline = _Pipeline()

    record_external_turn(pipeline, "当前状态", "当前没有进行中的任务。", source="runtime")

    assert pipeline._conversation.user_messages == ["当前状态"]
    assert pipeline._conversation.assistant_messages == ["当前没有进行中的任务。"]
    assert pipeline._episodic.entries[0][0] == "command"
    assert pipeline._episodic.entries[1][0] == "outcome"


def test_record_external_turn_accepts_text_channel(tmp_path) -> None:
    pipeline = _Pipeline()
    pipeline._turn_ledger = VoiceTurnLedger(tmp_path / "external-text.jsonl")

    record_external_turn(
        pipeline,
        "hello",
        "hi",
        source="text",
        channel="text",
        conversation_session_id="text-thread",
        turn_id="text-turn-1",
    )

    thread = pipeline._turn_ledger.get_thread("text-thread")
    turns = pipeline._turn_ledger.list_turns(thread_id="text-thread")
    assert thread.channel == "text"
    assert [turn.source for turn in turns] == ["text"]
    assert pipeline._conversation.user_messages == ["hello"]
    assert pipeline._conversation.assistant_messages == ["hi"]
