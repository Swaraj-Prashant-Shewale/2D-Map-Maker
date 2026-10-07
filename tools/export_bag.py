"""Export a rosbag2 recording into compact replay arrays (laptop tool).

    .venv/Scripts/python.exe tools/export_bag.py inputs/<recording>

Wraps :mod:`kart_map2d.io.bag_export` (pure Python via `rosbags`; no ROS
install needed). The export lands in inputs/<recording>/export/ - NOTE:
inputs/ is a junction into the research workspace's recordings; exports for
the September bags already exist there, so this tool is for NEW recordings.
"""

import _bootstrap  # noqa: F401

from kart_map2d.io.bag_export import main

if __name__ == "__main__":
    main()
