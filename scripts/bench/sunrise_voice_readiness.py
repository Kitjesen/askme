#!/usr/bin/env python3
"""Run the Sunrise voice readiness gate from a checkout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _main() -> int:
    from askme.voice.sunrise_readiness import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
