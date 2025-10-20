import os
import sys


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)


__all__ = [
    "arguments",
    "data",
    "models",
    "scheduler",
    "trainer",
    "utils",
]
