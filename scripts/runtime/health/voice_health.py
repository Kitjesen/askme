#!/usr/bin/env python3
"""Run the askme offline voice health check."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _main() -> int:
    from askme.voice.health_check import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
