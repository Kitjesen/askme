#!/usr/bin/env python3
"""Collect S100P readiness evidence from a checkout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _main() -> int:
    from askme.voice.s100p_readiness_bundle import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
