import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "third_party", "harness_audit")):
    if path not in sys.path:
        sys.path.insert(0, path)
