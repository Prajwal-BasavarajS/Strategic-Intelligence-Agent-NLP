"""
Executive Intelligence Dashboard for the NVIDIA CEO Agent (Phase 8).

Reads the cached JSON artifacts (no live LLM calls, so the demo never hangs)
and renders the 7 required sections:
  1. Company Overview
  2. Market Intelligence
  3. Opportunity Monitor
  4. Risk Monitor
  5. Sentiment Analysis
  6. Strategic Recommendations
  7. CEO Briefing

Run:  streamlit run app.py
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# --- Data loading -----------------------------------------------------------
CLEAN_PATH = "data/clean/docs.json"
ANALYSIS_PATH = "data/analysis.json"
RECS_PATH = "data/recommendations.json"
SENTIMENT_PATH = "data/sentiment.json"
BRIEFING_PATH = "data/briefing.json"   # produced in Phase 9; optional here

COMPANY = "NVIDIA"
INDUSTRY = "Semiconductors / AI Computing"


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


docs = load_json(CLEAN_PATH, [])
analysis = load_json(ANALYSIS_PATH, {})
recs = load_json(RECS_PATH, {"recommendations": []})
sentiment = load_json(SENTIMENT_PATH, {})
briefing = load_json(BRIEFING_PATH, None)

# --- Page config ------------------------------------------------------------
st.set_page_config(page_title=f"{COMPANY} CEO Agent", layout="wide")

IMPACT_COLOR = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
PRIORITY_COLOR = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}


# ============================================================================
# SECTION 1 — Company Overview
# ============================================================================
st.title(f"🧠 {COMPANY} — Strategic Intelligence Agent")
st.caption("AI CEO Agent · live-collected intelligence · evidence-based recommendations")

source_set = sorted({d["source"] for d in docs})
last_update = "—"
if docs:
    dates = [d.get("scraped_at", "") for d in docs if d.get("scraped_at")]
    if dates:
        last_update = max(dates)[:19].replace("T", " ") + " UTC"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Company", COMPANY)
c2.metric("Documents", len(docs))
c3.metric("Sources", len(source_set))
c4.metric("Last Update", last_update.split(" ")[0] if last_update != "—" else "—")
st.caption(f"Industry: {INDUSTRY}  ·  Sources: {', '.join(source_set)}  ·  Updated: {last_update}")

st.divider()

# ============================================================================
# SECTION 2 — Market Intelligence (recent news / announcements)
# ============================================================================
st.header("📡 Market Intelligence")
st.caption("Most recent items collected across sources")

def is_junk(d):
    t = d["title"]
    return t.count("$") > 3 or "$$" in t   # ticker-spam pattern

recent = sorted(
    [d for d in docs if d.get("date") and d["source"] in ("news", "nvidia_ir")
     and not is_junk(d)],
    key=lambda d: d["date"], reverse=True
)[:12]

if recent:
    for d in recent:
        day = d["date"][:10]
        st.markdown(f"**{d['title']}**  \n"
                    f"<span style='color:gray'>{d['source']} · {day}</span>  ·  "
                    f"[link]({d['url']})", unsafe_allow_html=True)
else:
    st.info("No dated documents available.")


st.divider()

# ============================================================================
# SECTION 3 / 4 — Opportunity & Risk Monitors
# ============================================================================
def render_findings(title, icon, findings):
    st.header(f"{icon} {title}")
    if not findings:
        st.info("No findings available.")
        return
    for f in findings:
        impact = f.get("impact", "Medium")
        conf = f.get("confidence", 0.0)
        with st.container(border=True):
            st.markdown(f"**{f['title']}**  {IMPACT_COLOR.get(impact, '⚪')} {impact}")
            st.write(f.get("detail", ""))
            st.caption(f"Confidence: {conf:.0%}")
            ev = f.get("evidence", [])
            if ev:
                st.markdown("**Evidence:**")
                for e in ev:
                    st.markdown(f"- [{e['title']}]({e['url']})")


col_o, col_r = st.columns(2)
with col_o:
    render_findings("Opportunity Monitor", "🚀",
                    analysis.get("opportunities", {}).get("findings", []))
with col_r:
    render_findings("Risk Monitor", "⚠️",
                    analysis.get("risks", {}).get("findings", []))

st.divider()

# ============================================================================
# SECTION 5 — Sentiment Analysis
# ============================================================================
st.header("📊 Sentiment Analysis")

if sentiment:
    overall = sentiment.get("overall_distribution", {})
    by_source = sentiment.get("by_source", {})
    trend = sentiment.get("trend", [])

    s1, s2 = st.columns([1, 2])
    with s1:
        st.subheader("Overall")
        if overall:
            odf = pd.DataFrame(
                {"sentiment": list(overall.keys()),
                 "count": list(overall.values())}
            ).set_index("sentiment")
            st.bar_chart(odf)

    with s2:
        st.subheader("By Source")
        if by_source:
            rows = []
            for src, summ in by_source.items():
                rows.append({
                    "source": src,
                    "avg_compound": summ["avg_compound"],
                    "label": summ["label"],
                    "positive": summ["distribution"]["positive"],
                    "neutral": summ["distribution"]["neutral"],
                    "negative": summ["distribution"]["negative"],
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)

    st.subheader("Sentiment Trend")
    if trend:
        tdf = pd.DataFrame(trend)
        tdf["date"] = pd.to_datetime(tdf["date"])
        tdf = tdf.set_index("date")[["avg_compound"]]
        st.line_chart(tdf)
else:
    st.info("Run sentiment.py to populate this section.")

st.divider()

# ============================================================================
# SECTION 6 — Strategic Recommendations
# ============================================================================
st.header("🎯 Strategic Recommendations")

rec_list = recs.get("recommendations", [])
if not rec_list:
    st.info("No recommendations available.")
else:
    for i, r in enumerate(rec_list, 1):
        prio = r.get("priority", "Medium")
        with st.container(border=True):
            st.markdown(f"### {i}. {r['recommendation']}")
            st.markdown(f"**Priority:** {PRIORITY_COLOR.get(prio, '⚪')} {prio}")
            st.write(r.get("rationale", ""))
            st.markdown(f"**Expected impact:** {r.get('expected_impact', '—')}")

            ra = r.get("risk_assessment", {})
            if ra:
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Financial risk", ra.get("financial", "—"))
                rc2.metric("Operational risk", ra.get("operational", "—"))
                rc3.metric("Strategic risk", ra.get("strategic", "—"))

            ev = r.get("evidence", [])
            if ev:
                st.markdown("**Supporting evidence:**")
                for e in ev:
                    st.markdown(f"- [{e['title']}]({e['url']})")

st.divider()

# ============================================================================
# SECTION 7 — CEO Briefing
# ============================================================================
st.header("📝 CEO Briefing")

if briefing:
    st.subheader("What happened?")
    st.write(briefing.get("what_happened", "—"))
    st.subheader("Why does it matter?")
    st.write(briefing.get("why_it_matters", "—"))
    st.subheader("What should management do next?")
    st.write(briefing.get("what_next", "—"))
else:
    st.info("CEO Briefing not yet generated. Run briefing.py (Phase 9) to "
            "populate this section.")