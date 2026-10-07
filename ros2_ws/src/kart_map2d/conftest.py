"""pytest bootstrap: make the kart_map2d package importable from any rootdir.

Colcon runs pytest from the package dir; a bare ``pytest`` from anywhere
works too because this conftest prepends the package parent to sys.path.
"""

import sys
from pathlib import Path

_pkg_parent = str(Path(__file__).resolve().parent)
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)
