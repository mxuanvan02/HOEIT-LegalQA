"""Pytest bootstrap: make repo modules importable without installation.

The pipeline keeps helpers in `src/` and research helpers in
`scripts/research/`. `src/quality_signals.py` inserts those paths itself at
import time, but the test modules import the helpers directly, so the paths
are registered here once for the whole session.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

for _relative in ("src", "scripts/research"):
    _path = str(_REPO_ROOT / _relative)
    if _path not in sys.path:
        sys.path.insert(0, _path)
