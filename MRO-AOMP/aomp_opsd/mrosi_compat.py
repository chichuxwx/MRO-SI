from __future__ import annotations

import os
import sys
from pathlib import Path


def mrosi_root() -> Path:
    configured = os.environ.get("MROSI_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    if (root / "mro_si").exists():
        return root
    return root / "MRO-SI"


def ensure_mrosi_on_path() -> Path:
    root = mrosi_root()
    if not root.exists():
        raise FileNotFoundError(
            f"MRO-SI root not found at {root}. Set MROSI_ROOT=/path/to/MRO-SI before running AOMP-OPSD."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


ensure_mrosi_on_path()

from mro_si.data_collator import MROSIDataCollator  # noqa: E402
from mro_si.trainer import MROSITrainer  # noqa: E402
from mro_si.verifier_utils import batch_verify_answer, extract_boxed_answer, grade_answer  # noqa: E402

__all__ = [
    "MROSIDataCollator",
    "MROSITrainer",
    "batch_verify_answer",
    "ensure_mrosi_on_path",
    "extract_boxed_answer",
    "grade_answer",
    "mrosi_root",
]
