"""Dataset utilities for the M5Product tri‑modal corpus."""

import os
from pathlib import Path
from typing import Dict

DATA_ROOT = Path(os.getenv("RNF_DATA_ROOT", "data"))

def ensure_m5product(download: bool = True) -> Dict[str, Path]:
    """Download the M5Product dataset to `DATA_ROOT` if necessary.

    Returns a dict with sub‑folders for image, text and tabular modalities."""
    image_dir  = DATA_ROOT / "m5product/images"
    text_file  = DATA_ROOT / "m5product/text.jsonl"
    table_file = DATA_ROOT / "m5product/tabular.parquet"

    if download and not image_dir.exists():
        print("[RNF] ⬇️  Fetching M5Product…")
        # NOTE: Replace with real URL when available
        raise NotImplementedError("Please provide download URL for M5Product.")

    return { "images": image_dir, "text": text_file, "table": table_file }
