"""
I/O helper utilities.
=====================

JSON, CSV, and .mat file reading/writing helpers.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Any


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save dict to JSON file."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def default_converter(o: Any) -> Any:
        """Serialize numpy scalars and arrays to native Python types for JSON."""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=default_converter)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load dict from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
