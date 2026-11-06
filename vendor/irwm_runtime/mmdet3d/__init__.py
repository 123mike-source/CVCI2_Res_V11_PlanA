"""Small compatibility surface for the frozen camera-only IR-WM path.

The full mmdetection3d package is not installed in wmmpc.  CVCI3 only needs
the detector/head builder and the BEVFormer base class, so these adapters keep
the environment untouched while reusing the installed MMDetection APIs.
"""

__version__ = "1.0.0"
