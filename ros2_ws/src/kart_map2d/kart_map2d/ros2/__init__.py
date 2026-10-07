"""Live ROS 2 wrappers (mapping node + drive node) around the verified stack.

Importable without ROS: ``converters`` is pure NumPy over duck-typed message
objects, and both nodes defer every ``rclpy`` / ``*_msgs`` import until a
node is actually constructed, so the whole test suite runs on machines
without a ROS installation.

BLAS thread capping happens in the top-level ``kart_map2d`` package (see its
__init__), which is imported before numpy on every entry point.
"""
