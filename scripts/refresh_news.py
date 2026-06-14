"""Pull the latest Tunisia / MENA ecosystem articles from public RSS feeds and
write data/news_cache.json. The Streamlit Veille tab reads this file with a
graceful fallback to baked-in articles when the file is missing or empty.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import feedparser

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "news_cache.json"
LOOKBACK_DAYS = int(os.environ.get("CDC_NEWS_LOOKBACK_DAYS", "90"))
MAX_ITEMS = int(os.environ.get("CDC_NEWS_MAX_ITEMS", "12"))

# Public RSS / Atom feeds for Tunisia + MENA ecosystem coverage. Add or
# replace via repo edit; the schema below stays the same.
FEEDS: list[dict[str, str]] = [
    {"source": "wamda.com", "tag": "MENA",
     "url": "https://www.wamda.com/feed",
     "color": "navy"},
    {"source": "theafricareport.com", "tag": "Africa",
     "url": "https://www.theafricareport.com/feed/",
     "color": "red"},
    {"source": "menabytes.com", "tag": "MENA",
     "url": "https://www.menabytes.com/feed/",
     "color": "navy"},
    {"source": "smartcapital.tn", "tag": "Programme",
     "url": "https://smartcapital.tn/feed/",
     "color": "red"},
]

# Cheap relevance filter - only keep articles mentioning Tunisia / MENA /
# fintech / greentech / startup. Anything else is dropped.
RELEVANT = re.compile(
    r"\b(tunisia|tunisie|tunisien|tunis|maghreb|mena|startup|fintech|greentech|"
    r"venture|edtech|deeptech|africa|africain)\b",
    re.IGNORECASE,
)


def parse_date(entry: Any) -> str:
    for key in ("published_parsed", "updated_parsed"):
        val = entry.get(key)
        if val:
            try:
                return dt.date(val.tm_year, val.tm_mon, val.tm_mday).isoformat()
            except Exception:
                continue
    return dt.date.today().isoformat()


def truncate(text: str, n: int) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[: n - 1] + "..." if len(text) > n else text


def fetch_all() -> list[dict[str, Any]]:
    cutoff = dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)
    out: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for feed_def in FEEDS:
        url = feed_def["url"]
        try:
            parsed = feedparser.parse(url)
        except Exception as exc:
            print(f"[refresh_news] feed {url} failed: {exc!r}")
            continue
        for entry in parsed.entries[:25]:
            title = str(entry.get("title", "")).strip()
            link = str(entry.get("link", "")).strip()
            if not title or not link or link in seen_urls:
                continue
            blob = f"{title} {entry.get('summary', '')}"
            if not RELEVANT.search(blob):
                continue
            date_iso = parse_date(entry)
            try:
                if dt.date.fromisoformat(date_iso) < cutoff:
                    continue
            except Exception:
                pass
            seen_urls.add(link)
            summary = truncate(entry.get("summary", ""), 260)
            out.append({
                "title_en": title,
                "title_fr": title,        # same headline - translation is left to readers
                "source": feed_def["source"],
                "date": date_iso,
                "tag": feed_def["tag"],
                "url": link,
                "summary_en": summary,
                "summary_fr": summary,
                "color": feed_def["color"],
            })
    out.sort(key=lambda a: a["date"], reverse=True)
    return out[:MAX_ITEMS]


def main() -> int:
    items = fetch_all()
    if not items:
        print("[refresh_news] no items found - leaving previous cache untouched")
        # Make sure the file exists with empty array so the app side is happy.
        if not TARGET.exists():
            TARGET.parent.mkdir(parents=True, exist_ok=True)
            TARGET.write_text(json.dumps({"items": [], "updated_at": ""}), encoding="utf-8")
        return 0

    payload = {
        "items": items,
        "updated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "lookback_days": LOOKBACK_DAYS,
        "count": len(items),
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[refresh_news] wrote {len(items)} items to {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
