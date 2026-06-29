"""
Scores every clean document with VADER, then produces:
  - per-document compound score + label
  - aggregate sentiment by source (news vs public/reddit vs corporate)
  - a daily sentiment trend (avg compound per day)

Note: VADER is tuned for social/short text. It reads Reddit well; news
headlines are often written neutrally, so news skews toward neutral. That's
why we report sentiment SEPARATELY by source - public sentiment is the more
expressive signal, news sentiment is more neutral by nature.

"""

import json
import os
from collections import defaultdict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CLEAN_PATH = "data/clean/docs.json"
SENTIMENT_PATH = "data/sentiment.json"


def label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def main():
    with open(CLEAN_PATH) as f:
        docs = json.load(f)

    analyzer = SentimentIntensityAnalyzer()

    scored = []
    by_source = defaultdict(list)          # source -> [compound, ...]
    by_day = defaultdict(list)             # 'YYYY-MM-DD' -> [compound, ...]

    for d in docs:
        text = f"{d['title']} {d['text']}".strip()
        compound = analyzer.polarity_scores(text)["compound"]
        lab = label(compound)

        scored.append({
            "id": d["id"],
            "source": d["source"],
            "title": d["title"],
            "compound": round(compound, 4),
            "label": lab,
            "date": d.get("date", ""),
        })
        by_source[d["source"]].append(compound)
        day = (d.get("date") or "")[:10]   # 'YYYY-MM-DD'
        if day:
            by_day[day].append(compound)

    # Aggregate by source: average compound + label distribution.
    source_summary = {}
    for src, scores in by_source.items():
        avg = sum(scores) / len(scores)
        dist = {"positive": 0, "neutral": 0, "negative": 0}
        for s in scores:
            dist[label(s)] += 1
        source_summary[src] = {
            "avg_compound": round(avg, 4),
            "label": label(avg),
            "count": len(scores),
            "distribution": dist,
        }

    # Daily trend, sorted by date.
    trend = [
        {"date": day, "avg_compound": round(sum(v) / len(v), 4), "count": len(v)}
        for day, v in sorted(by_day.items())
    ]

    # Overall distribution across all docs.
    overall = {"positive": 0, "neutral": 0, "negative": 0}
    for s in scored:
        overall[s["label"]] += 1

    result = {
        "overall_distribution": overall,
        "by_source": source_summary,
        "trend": trend,
        "documents": scored,
    }

    os.makedirs(os.path.dirname(SENTIMENT_PATH), exist_ok=True)
    with open(SENTIMENT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Scored {len(scored)} documents -> {SENTIMENT_PATH}\n")
    print("Overall:", overall)
    print("\nBy source:")
    for src, summ in source_summary.items():
        print(f"  {src:<12} avg={summ['avg_compound']:+.3f} "
              f"({summ['label']}) n={summ['count']} {summ['distribution']}")
    print(f"\nTrend points: {len(trend)} days")


if __name__ == "__main__":
    main()