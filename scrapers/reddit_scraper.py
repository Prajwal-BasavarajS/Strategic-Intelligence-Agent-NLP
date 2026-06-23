"""
Reddit scraper for NVIDIA CEO Agent (RSS + rate-limit handling).
Source #1 of 3. Uses public .rss feeds. Handles Reddit's 429 rate limiting
with a longer inter-request delay and one retry with backoff.

Outputs: data/raw/reddit.json
"""

import json
import time
import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
import requests

# --- Config -----------------------------------------------------------------
SUBREDDITS = ["nvidia", "wallstreetbets","ArtificialInteligence"]
OUTPUT_PATH = "data/raw/reddit.json"
DELAY_SECONDS = 8          # polite gap between feeds; Reddit RSS limits bursts
MAX_RETRIES = 2            # retry a 429 this many extra times

HEADERS = {
    "User-Agent": "nvidia-ceo-agent/0.1 (academic project; contact: student)"
}


def strip_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    return re.sub(r"\s+", " ", text).strip()


def parse_date(entry) -> str:
    """Return an ISO-8601 UTC string from a feed entry, or empty if missing.

    feedparser exposes Atom dates two ways:
      - entry.published / entry.updated         (raw string)
      - entry.published_parsed / .updated_parsed (time.struct_time, UTC)
    The struct_time form is the reliable one for Reddit's Atom feeds, so we
    try it first, then fall back to parsing the raw string.
    """
    import calendar
    import time as _time

    # 1. Preferred: the pre-parsed struct_time (already UTC)
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            ts = calendar.timegm(st)          # struct_time (UTC) -> unix ts
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    # 2. Fallback: raw RFC-822/ISO string
    raw = entry.get("published") or entry.get("updated")
    if not raw:
        return ""
    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc).isoformat()
    except Exception:
        return ""


def fetch_with_retry(url: str) -> requests.Response:
    """GET with retry/backoff on 429. Raises on final failure."""
    delay = DELAY_SECONDS
    for attempt in range(MAX_RETRIES + 1):
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 429 and attempt < MAX_RETRIES:
            print(f"    429 received, backing off {delay}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2          # exponential backoff
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()     # unreachable, but satisfies type checkers
    return resp


def scrape_subreddit(sub: str) -> list[dict]:
    url = f"https://www.reddit.com/r/{sub}/.rss"
    resp = fetch_with_retry(url)
    feed = feedparser.parse(resp.content)

    docs = []
    for e in feed.entries:
        title = (e.get("title") or "").strip()
        summary = strip_html(e.get("summary", ""))
        docs.append({
            "id": f"reddit_{e.get('id', e.get('link', title))[-32:]}",
            "source": "reddit",
            "source_detail": f"r/{sub}",
            "title": title,
            "text": summary if summary else title,
            "url": e.get("link", ""),
            "date": parse_date(e),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        })
    return docs


def main():
    os.makedirs("data/raw", exist_ok=True)
    all_docs = []

    for i, sub in enumerate(SUBREDDITS):
        try:
            docs = scrape_subreddit(sub)
            print(f"  r/{sub}: {len(docs)} posts")
            all_docs.extend(docs)
        except Exception as e:
            print(f"  r/{sub}: FAILED ({type(e).__name__}: {e})")
        # wait between feeds, but not after the last one
        if i < len(SUBREDDITS) - 1:
            time.sleep(DELAY_SECONDS)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_docs, f, indent=2)

    print(f"\nSaved {len(all_docs)} documents -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()