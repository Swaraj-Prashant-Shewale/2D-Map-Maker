"""The autonomous drive stack: follow a saved map package.

A map is built ONCE (offline pipeline or the live mapping node's
``~/save_map``) with one of the methods (flat / lidar3d); the kart then
drives it with this package, which only LOADS those results - it never runs
segmentation networks, loop closure, cloud building or racing-line
optimisation:

* :mod:`kart_map2d.drive.localization` - map-relative localization: the
  mapping stack's FusionEngine (IMU / visual / wheel odometry) plus
  low-rate absolute pose fixes against the map's grid (LiDAR scan-to-map
  registration when a LiDAR is present, camera ground-evidence patch
  correlation otherwise), global relocalization at start, every fix gated
  by the EKF and tracked by the health monitor like any sensor.
* :mod:`kart_map2d.drive.controller` - pure pursuit on the racing line with a
  kart-feasible speed profile.
* :mod:`kart_map2d.drive.safety` - the last word on every command: speed cap,
  e-stop latch, deadman, watchdogs (pose age, fusion mode, localization) and
  obstacle stops along the path ahead.
* :mod:`kart_map2d.drive.sim` - kinematic kart simulator for closed-loop
  tests (not deployed).

The ROS 2 executable is :mod:`kart_map2d.ros2.drive_node` (separate from the
mapping node, so the kart runs only this while driving). Importing this
package imports nothing heavy; each module documents its frames and units.
"""
