"""kart_map2d: the go-kart's standalone 2D mapping + driving stack.

Ported from the proven Maveric research workspace (Virtual map maker),
2D/flat only, camera-agnostic, Orin-only (ROS 2 Humble, Python 3.10,
numpy < 2). No module here imports rclpy or torch at import time.

BLAS threads: the live node runs numpy on several threads at once (the
executor, per-camera VO workers, a save). With OpenBLAS's default of one
thread per core, every concurrent matmul spins up a full thread pool and
they fight over the cores (measured 4x slower grid inserts in the Humble
container). Importing this package first caps the pools at one thread; an
explicit OPENBLAS_NUM_THREADS / MKL_NUM_THREADS in the environment wins.
"""

import os as _os

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_name, "1")

__version__ = "1.0.0"
