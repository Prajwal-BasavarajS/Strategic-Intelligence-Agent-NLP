"""
NVIDIA Newsroom scraper for the CEO Agent.
Source #3 of 3. Pulls official NVIDIA press releases / announcements from
the company newsroom RSS feed. This is the 'corporate' viewpoint that makes
the 3-source independence story airtight (community / press / corporate).

"""

import json
import time
import os
import re
import hashlib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
import requests

# --- Config -----------------------------------------------------------------
# Official NVIDIA newsroom feed. Investor press-release feed is an alternative.
FEEDS = [
    ("nvidia_newsroom", "https://nvidianews.nvidia.com/rss.xml"),
]
OUTPUT_PATH = "data/raw/nvidia_ir.json"

HEADERS = {
    "User-Agent": "nvidia-ceo-agent/0.1 (academic project; contact: student)"
}


def strip_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    return re.sub(r"\s+", " ", text).strip()


def parse_date(entry) -> str:
    import calendar
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            ts = calendar.timegm(st)
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    raw = entry.get("published") or entry.get("updated")
    if not raw:
        return ""
    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc).isoformat()
    except Exception:
        return ""


def make_id(label: str, link: str, title: str) -> str:
    basis = link or title
    h = hashlib.md5(basis.encode("utf-8")).hexdigest()[:12]
    return f"ir_{label}_{h}"


def scrape_feed(label: str, url: str) -> list[dict]:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)

    docs = []
    for e in feed.entries:
        title = (e.get("title") or "").strip()
        # IR feeds often carry a fuller summary/description than Google News
        summary = strip_html(e.get("summary", "")) or strip_html(e.get("description", ""))
        docs.append({
            "id": make_id(label, e.get("link", ""), title),
            "source": "nvidia_ir",
            "source_detail": label,
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

    for label, url in FEEDS:
        try:
            docs = scrape_feed(label, url)
            print(f"  {label}: {len(docs)} items")
            all_docs.extend(docs)
        except Exception as e:
            print(f"  {label}: FAILED ({type(e).__name__}: {e})")
        time.sleep(2)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_docs, f, indent=2)

    print(f"\nSaved {len(all_docs)} documents -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()