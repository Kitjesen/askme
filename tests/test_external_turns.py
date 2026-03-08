from askme.pipeline.external_turns import record_external_turn


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
