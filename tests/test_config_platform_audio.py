from __future__ import annotations

from askme import config as config_mod


def test_windows_platform_audio_overrides_alsa_usb_settings(monkeypatch):
    monkeypatch.setattr(config_mod.os, "name", "nt", raising=False)
    monkeypatch.delenv("ASKME_DISABLE_PLATFORM_AUDIO_OVERRIDES", raising=False)
    cfg = {
        "voice": {
            "input_device": "hw:1,0",
            "input_transport": "usb_direct",
            "tts": {
                "output_device": "plughw:1,0",
                "output_transport": "usb_direct",
            },
        }
    }

    config_mod._apply_platform_audio_overrides(cfg)

    assert cfg["voice"]["input_device"] is None
    assert cfg["voice"]["input_transport"] == "sounddevice"
    assert cfg["voice"]["tts"]["output_device"] is None
    assert cfg["voice"]["tts"]["output_transport"] == "sounddevice"
    assert cfg["_platform_audio_overrides"]


def test_platform_audio_overrides_can_be_disabled(monkeypatch):
    monkeypatch.setattr(config_mod.os, "name", "nt", raising=False)
    monkeypatch.setenv("ASKME_DISABLE_PLATFORM_AUDIO_OVERRIDES", "1")
    cfg = {
        "voice": {
            "tts": {
                "output_device": "plughw:1,0",
                "output_transport": "usb_direct",
            },
        }
    }

    config_mod._apply_platform_audio_overrides(cfg)

    assert cfg["voice"]["tts"]["output_device"] == "plughw:1,0"
    assert cfg["voice"]["tts"]["output_transport"] == "usb_direct"
    assert "_platform_audio_overrides" not in cfg
