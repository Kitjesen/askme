from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from askme.runtime.modules.voice_module import VoiceModule


@pytest.mark.asyncio
async def test_voice_module_uses_audio_input_lifecycle() -> None:
    started = asyncio.Event()

    async def _run() -> None:
        started.set()
        await asyncio.Event().wait()

    mod = VoiceModule()
    mod._audio = MagicMock()
    mod._voice_loop = MagicMock()
    mod._voice_loop.run = _run
    mod._task = None

    await mod.start()
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await mod.stop()

    mod._audio.start_input.assert_called_once_with()
    mod._audio.stop_input.assert_called_once_with()
    mod._audio.shutdown.assert_called_once_with()
