"""ament_python packaging for kart_map2d.

Build:  colcon build --symlink-install   (in kart2d:humble or on the Orin)
Test:   colcon test --packages-select kart_map2d && colcon test-result --verbose
"""

from pathlib import Path

from setuptools import find_packages, setup

package_name = "kart_map2d"
here = Path(__file__).parent

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=("test", "test.*")),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", [str(p.relative_to(here).as_posix()) for p in sorted((here / "launch").glob("*.launch.py"))]),
        (f"share/{package_name}/config", [str(p.relative_to(here).as_posix()) for p in sorted((here / "config").glob("*")) if p.is_file()]),
        # The Fast DDS large-SHM profile ships with the package (9 MB frames).
        (f"share/{package_name}", ["kart_map2d/ros2/fastdds_large_shm.xml"]),
    ],
    # Non-.py files that must install WITH the package (not only to share/):
    # the recording-rig data file and the Fast DDS profile source copy.
    package_data={package_name: ["io/recorded_rigs.json", "ros2/fastdds_large_shm.xml"]},
    include_package_data=True,
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Maveric kart team",
    maintainer_email="26alladat@gmail.com",
    description="Standalone 2D mapping + drive stack for the Maveric go-kart.",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mapping_node = kart_map2d.ros2.node:main",
            "drive_node = kart_map2d.ros2.drive_node:main",
            "check_cfg = kart_map2d.config.check_cfg:main",
        ],
    },
)
