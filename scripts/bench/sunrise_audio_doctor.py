#!/usr/bin/env python3
"""Run the Sunrise MCP01 audio doctor from a checkout."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from askme.voice.sunrise_audio_doctor import main

if __name__ == "__main__":
    raise SystemExit(main())
