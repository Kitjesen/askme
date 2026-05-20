from __future__ import annotations

import importlib
import sys


def test_state_led_bridge_facade_does_not_load_provider_on_module_import() -> None:
    sys.modules.pop("askme.pipeline.reactions.state_led_bridge", None)
    sys.modules.pop("askme.providers", None)

    importlib.import_module("askme.pipeline.reactions.state_led_bridge")

    assert "askme.providers" not in sys.modules
