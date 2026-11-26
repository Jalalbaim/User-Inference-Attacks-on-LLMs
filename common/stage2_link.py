"""Load Stage2/analysis.py under the name 's2_analysis' (every stage folder has its own analysis.py,
so a plain 'import analysis' would pick up the caller's own module)."""
import importlib.util
import os
import sys

_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Stage2", "analysis.py")


def stage2():
    if "s2_analysis" not in sys.modules:
        spec = importlib.util.spec_from_file_location("s2_analysis", _P)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["s2_analysis"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["s2_analysis"]
