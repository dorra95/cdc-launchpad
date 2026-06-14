"""Re-run the selection model on the latest dataset + learning store and
write the resulting metrics to data/model_metrics.json. The Streamlit app
reads this file to show the freshness of the model in the Capitalize tab.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "model_metrics.json"

# Import the existing engine - no duplication, no parallel implementation.
sys.path.insert(0, str(ROOT))
import app as cdc  # noqa: E402


def main() -> int:
    try:
        df = cdc.load_base()
    except Exception as exc:
        print(f"[retrain] dataset unreadable: {exc!r}")
        return 1

    n_store = 0
    if cdc.STORE_FILE and Path(cdc.STORE_FILE).exists():
        try:
            import pandas as pd
            n_store = int(len(pd.read_csv(cdc.STORE_FILE)))
        except Exception:
            n_store = 0

    bundle = cdc.train_selection(df)
    m = bundle.metrics

    payload = {
        "trained_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "engine": m.get("engine", "unknown"),
        "roc_auc": m.get("roc_auc"),
        "f1": m.get("f1"),
        "accuracy": m.get("accuracy"),
        "n_rows": int(m.get("n_rows", 0)),
        "n_funded": int(m.get("n_funded", 0)),
        "store_rows": n_store,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[retrain] engine={payload['engine']} "
        f"roc={payload['roc_auc']} f1={payload['f1']} acc={payload['accuracy']} "
        f"n={payload['n_rows']} funded={payload['n_funded']} store={n_store}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
