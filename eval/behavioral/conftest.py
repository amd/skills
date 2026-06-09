"""pytest wiring for the behavioral harness.

Adds this directory to ``sys.path`` so tests can ``from harness import ...``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
