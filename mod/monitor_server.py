#!/usr/bin/env python3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MONITOR_DIR = BASE_DIR.parent / "monitor"
if MONITOR_DIR.exists():
    sys.path.insert(0, str(MONITOR_DIR))

import monitoring  # type: ignore

if __name__ == "__main__":
    monitoring.run_health_server("mod", base_dir=BASE_DIR)
