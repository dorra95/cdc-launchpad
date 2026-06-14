"""Pull the latest startup dataset from a configured source and overwrite
Startups_Tunisia_Master_v4.xlsx. The Streamlit app's existing loader picks up
the new file automatically on next rerun.

Set these GitHub Actions secrets to wire a real source:
  CDC_DATASET_SOURCE_URL    direct download URL (Google Sheets export, Airtable
                            API endpoint, S3 presigned URL, ...)
  CDC_DATASET_SOURCE_FORMAT 'xlsx' (default) or 'csv'

When the URL is empty, the script is a no-op so the workflow stays green.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "Startups_Tunisia_Master_v4.xlsx"


def main() -> int:
    url = os.environ.get("CDC_DATASET_SOURCE_URL", "").strip()
    fmt = os.environ.get("CDC_DATASET_SOURCE_FORMAT", "xlsx").strip().lower()
    if not url:
        print("[sync_dataset] CDC_DATASET_SOURCE_URL not configured - skipping")
        return 0

    print(f"[sync_dataset] fetching {url}")
    try:
        resp = requests.get(url, timeout=60, allow_redirects=True)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[sync_dataset] download failed: {exc!r}")
        return 1

    blob = resp.content
    if fmt == "csv":
        try:
            import pandas as pd
            df = pd.read_csv(io.BytesIO(blob))
            buf = io.BytesIO()
            df.to_excel(buf, index=False)
            blob = buf.getvalue()
        except Exception as exc:
            print(f"[sync_dataset] CSV -> XLSX conversion failed: {exc!r}")
            return 1

    # Quick sanity - make sure the new file is actually openable.
    try:
        import pandas as pd
        pd.read_excel(io.BytesIO(blob), sheet_name=0, nrows=5)
    except Exception as exc:
        print(f"[sync_dataset] downloaded file is not a valid xlsx: {exc!r}")
        return 1

    TARGET.write_bytes(blob)
    print(f"[sync_dataset] wrote {len(blob):,} bytes to {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
