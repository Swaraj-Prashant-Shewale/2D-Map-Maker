"""Make ``import kart_map2d`` work from the tools/ directory (no install needed).

Every tool starts with ``import _bootstrap`` (this file lives next to them).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_PARENT = PROJECT_ROOT / "ros2_ws" / "src" / "kart_map2d"
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))
