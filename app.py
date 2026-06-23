# """
# Executive Intelligence Dashboard for the NVIDIA CEO Agent (Phase 8, polished).

# Reads cached JSON artifacts (no live LLM calls) and renders the 7 required
# sections. This version adds light visual polish: colored badges, compact risk
# pills, card spacing, and a header accent — no structural/data changes.

# Run:  streamlit run app.py
# """

# import json
# import os
# from datetime import datetime, timezone

# import pandas as pd
# import streamlit as st

# # --- Data loading -----------------------------------------------------------
# CLEAN_PATH = "data/clean/docs.json"
# ANALYSIS_PATH = "data/analysis.json"
# RECS_PATH = "data/recommendations.json"
# SENTIMENT_PATH = "data/sentiment.json"
# BRIEFING_PATH = "data/briefing.json"

# COMPANY = "NVIDIA"
# INDUSTRY = "Semiconductors / AI Computing"


# def load_json(path, default=None):
#     if not os.path.exists(path):
#         return default
#     with open(path) as f:
#         return json.load(f)


# docs = load_json(CLEAN_PATH, [])
# analysis = load_json(ANALYSIS_PATH, {})
# recs = load_json(RECS_PATH, {"recommendations": []})
# sentiment = load_json(SENTIMENT_PATH, {})
# briefing = load_json(BRIEFING_PATH, None)

# # --- Page config + CSS ------------------------------------------------------
# st.set_page_config(page_title=f"{COMPANY} CEO Agent", layout="wide",
#                    page_icon="🧠")

# st.markdown("""
# <style>
#     /* Accent header bar */
#     .hero {
#         background: linear-gradient(90deg, #76b900 0%, #1a1a1a 100%);
#         padding: 1.4rem 1.6rem; border-radius: 12px; margin-bottom: 0.5rem;
#     }
#     .hero h1 { color: #ffffff; margin: 0; font-size: 1.9rem; }
#     .hero p  { color: #d7f0b0; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

#     /* Badges */
#     .badge {
#         display: inline-block; padding: 2px 10px; border-radius: 999px;
#         font-size: 0.78rem; font-weight: 600; color: #fff; margin-left: 6px;
#         vertical-align: middle;
#     }
#     .b-high { background: #d62828; }
#     .b-med  { background: #f08c00; }
#     .b-low  { background: #2f9e44; }

#     /* Risk pills row */
#     .riskwrap { display: flex; gap: 10px; margin: 8px 0; flex-wrap: wrap; }
#     .riskpill {
#         flex: 1; min-width: 150px; border: 1px solid #e3e3e3;
#         border-radius: 8px; padding: 8px 12px; background: #fafafa;
#     }
#     .riskpill .lbl { font-size: 0.72rem; color: #888; text-transform: uppercase;
#                      letter-spacing: 0.04em; }
#     .riskpill .val { font-size: 1.0rem; font-weight: 600; }
#     .v-high { color: #d62828; } .v-med { color: #f08c00; } .v-low { color: #2f9e44; }

#     .conf { font-size: 0.78rem; color: #999; }
# </style>
# """, unsafe_allow_html=True)

# LEVEL_CLASS = {"High": "high", "Medium": "med", "Low": "low"}


# def badge(level: str) -> str:
#     c = LEVEL_CLASS.get(level, "med")
#     return f"<span class='badge b-{c}'>{level}</span>"


# def risk_pill(label: str, value: str) -> str:
#     c = LEVEL_CLASS.get(value, "med")
#     return (f"<div class='riskpill'><div class='lbl'>{label}</div>"
#             f"<div class='val v-{c}'>{value}</div></div>")


# # ============================================================================
# # SECTION 1 — Company Overview
# # ============================================================================
# st.markdown(
#     f"<div class='hero'><h1>🧠 {COMPANY} — Strategic Intelligence Agent</h1>"
#     f"<p>AI CEO Agent · live-collected intelligence · evidence-based recommendations</p>"
#     f"</div>", unsafe_allow_html=True)

# source_set = sorted({d["source"] for d in docs})
# last_update = "—"
# if docs:
#     dates = [d.get("scraped_at", "") for d in docs if d.get("scraped_at")]
#     if dates:
#         last_update = max(dates)[:19].replace("T", " ") + " UTC"

# c1, c2, c3, c4 = st.columns(4)
# c1.metric("Company", COMPANY)
# c2.metric("Documents", len(docs))
# c3.metric("Sources", len(source_set))
# c4.metric("Last Update", last_update.split(" ")[0] if last_update != "—" else "—")
# st.caption(f"Industry: {INDUSTRY}  ·  Sources: {', '.join(source_set)}  ·  Updated: {last_update}")

# st.divider()

# # ============================================================================
# # SECTION 2 — Market Intelligence
# # ============================================================================
# st.header("📡 Market Intelligence")
# st.caption("Most recent news and announcements")


# def is_junk(d):
#     t = d.get("title", "")
#     return t.count("$") > 3 or "$$" in t


# recent = sorted(
#     [d for d in docs if d.get("date") and d["source"] in ("news", "nvidia_ir")
#      and not is_junk(d)],
#     key=lambda d: d["date"], reverse=True
# )[:12]

# if recent:
#     for d in recent:
#         day = d["date"][:10]
#         st.markdown(f"**{d['title']}**  \n"
#                     f"<span style='color:gray'>{d['source']} · {day}</span>  ·  "
#                     f"[link]({d['url']})", unsafe_allow_html=True)
# else:
#     st.info("No dated documents available.")

# st.divider()

# # ============================================================================
# # SECTION 3 / 4 — Opportunity & Risk Monitors
# # ============================================================================
# def render_findings(title, icon, findings):
#     st.header(f"{icon} {title}")
#     if not findings:
#         st.info("No findings available.")
#         return
#     for f in findings:
#         impact = f.get("impact", "Medium")
#         conf = f.get("confidence", 0.0)
#         with st.container(border=True):
#             st.markdown(f"**{f['title']}** {badge(impact)}",
#                         unsafe_allow_html=True)
#             st.write(f.get("detail", ""))
#             st.markdown(f"<span class='conf'>Confidence: {conf:.0%}</span>",
#                         unsafe_allow_html=True)
#             ev = f.get("evidence", [])
#             if ev:
#                 st.markdown("**Evidence:**")
#                 for e in ev:
#                     st.markdown(f"- [{e['title']}]({e['url']})")


# col_o, col_r = st.columns(2)
# with col_o:
#     render_findings("Opportunity Monitor", "🚀",
#                     analysis.get("opportunities", {}).get("findings", []))
# with col_r:
#     render_findings("Risk Monitor", "⚠️",
#                     analysis.get("risks", {}).get("findings", []))

# st.divider()

# # ============================================================================
# # SECTION 5 — Sentiment Analysis
# # ============================================================================
# st.header("📊 Sentiment Analysis")

# if sentiment:
#     overall = sentiment.get("overall_distribution", {})
#     by_source = sentiment.get("by_source", {})
#     trend = sentiment.get("trend", [])

#     s1, s2 = st.columns([1, 2])
#     with s1:
#         st.subheader("Overall")
#         if overall:
#             odf = pd.DataFrame(
#                 {"sentiment": list(overall.keys()),
#                  "count": list(overall.values())}
#             ).set_index("sentiment")
#             st.bar_chart(odf)
#     with s2:
#         st.subheader("By Source")
#         if by_source:
#             rows = []
#             for src, summ in by_source.items():
#                 rows.append({
#                     "source": src,
#                     "avg_compound": summ["avg_compound"],
#                     "label": summ["label"],
#                     "positive": summ["distribution"]["positive"],
#                     "neutral": summ["distribution"]["neutral"],
#                     "negative": summ["distribution"]["negative"],
#                 })
#             st.dataframe(pd.DataFrame(rows), hide_index=True,
#                          use_container_width=True)

#     st.subheader("Sentiment Trend")
#     if trend:
#         tdf = pd.DataFrame(trend)
#         tdf["date"] = pd.to_datetime(tdf["date"])
#         tdf = tdf.set_index("date")[["avg_compound"]]
#         st.line_chart(tdf)
# else:
#     st.info("Run sentiment.py to populate this section.")

# st.divider()

# # ============================================================================
# # SECTION 6 — Strategic Recommendations
# # ============================================================================
# st.header("🎯 Strategic Recommendations")

# rec_list = recs.get("recommendations", [])
# if not rec_list:
#     st.info("No recommendations available.")
# else:
#     for i, r in enumerate(rec_list, 1):
#         prio = r.get("priority", "Medium")
#         with st.container(border=True):
#             st.markdown(f"### {i}. {r['recommendation']} {badge(prio)}",
#                         unsafe_allow_html=True)
#             st.write(r.get("rationale", ""))
#             st.markdown(f"**Expected impact:** {r.get('expected_impact', '—')}")

#             ra = r.get("risk_assessment", {})
#             if ra:
#                 st.markdown(
#                     "<div class='riskwrap'>"
#                     + risk_pill("Financial risk", ra.get("financial", "—"))
#                     + risk_pill("Operational risk", ra.get("operational", "—"))
#                     + risk_pill("Strategic risk", ra.get("strategic", "—"))
#                     + "</div>", unsafe_allow_html=True)

#             ev = r.get("evidence", [])
#             if ev:
#                 st.markdown("**Supporting evidence:**")
#                 for e in ev:
#                     st.markdown(f"- [{e['title']}]({e['url']})")

# st.divider()

# # ============================================================================
# # SECTION 7 — CEO Briefing
# # ============================================================================
# st.header("📝 CEO Briefing")

# if briefing:
#     st.subheader("What happened?")
#     st.write(briefing.get("what_happened", "—"))
#     st.subheader("Why does it matter?")
#     st.write(briefing.get("why_it_matters", "—"))
#     st.subheader("What should management do next?")
#     st.write(briefing.get("what_next", "—"))
# else:
#     st.info("CEO Briefing not yet generated. Run briefing.py to populate this section.")



"""
Executive Intelligence Dashboard for the NVIDIA CEO Agent (styled).

Reads cached JSON artifacts (no live LLM calls) and renders the 7 required
sections with a full visual style pass: color system, KPI tiles, styled
section headers, and cards. No data/logic changes from the working version.

Run:  streamlit run app.py
"""

import json
import os

import pandas as pd
import streamlit as st

CLEAN_PATH = "data/clean/docs.json"
ANALYSIS_PATH = "data/analysis.json"
RECS_PATH = "data/recommendations.json"
SENTIMENT_PATH = "data/sentiment.json"
BRIEFING_PATH = "data/briefing.json"

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

st.set_page_config(page_title=f"{COMPANY} CEO Agent", layout="wide",
                   page_icon="brain")

st.markdown("""
<style>
    :root {
        --nv: #76b900; --nv-dark: #5a8f00;
        --ink: #14110f; --muted: #6b7280;
        --red: #d62828; --orange: #e8870b; --green: #2f9e44;
        --card: #ffffff; --line: #ececec; --bg-soft: #f7f8f5;
    }
    .block-container { padding-top: 1.5rem; max-width: 1300px; }
    .hero {
        background: radial-gradient(120% 140% at 0% 0%, #1f2937 0%, #0b0f14 60%);
        border-left: 6px solid var(--nv);
        padding: 1.5rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem;
    }
    .hero h1 { color:#fff; margin:0; font-size:1.95rem; letter-spacing:-0.01em; }
    .hero p  { color:#9fce5a; margin:.35rem 0 0; font-size:.95rem; }
    .kpis { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:.4rem; }
    .kpi {
        flex:1; min-width:180px; background:var(--card);
        border:1px solid var(--line); border-radius:12px; padding:14px 18px;
        box-shadow:0 1px 2px rgba(0,0,0,.04);
    }
    .kpi .k-label { font-size:.74rem; text-transform:uppercase;
        letter-spacing:.06em; color:var(--muted); }
    .kpi .k-value { font-size:1.7rem; font-weight:700; color:var(--ink);
        line-height:1.1; margin-top:2px; }
    .kpi .k-accent { color:var(--nv-dark); }
    .sec {
        display:flex; align-items:center; gap:10px;
        margin:1.4rem 0 .4rem; padding-bottom:.5rem;
        border-bottom:2px solid var(--bg-soft);
    }
    .sec .s-icon {
        width:34px; height:34px; border-radius:9px; background:var(--bg-soft);
        display:flex; align-items:center; justify-content:center; font-size:1.1rem;
    }
    .sec .s-title { font-size:1.3rem; font-weight:700; color:var(--ink); }
    .sec .s-sub { font-size:.82rem; color:var(--muted); margin-left:auto; }
    .card {
        background:var(--card); border:1px solid var(--line);
        border-radius:12px; padding:16px 18px; margin-bottom:12px;
        box-shadow:0 1px 2px rgba(0,0,0,.04);
    }
    .card.opp { border-left:4px solid var(--green); }
    .card.risk { border-left:4px solid var(--red); }
    .card.rec { border-left:4px solid var(--nv); }
    .card h4 { margin:0 0 .3rem; font-size:1.02rem; color:var(--ink); }
    .card p { margin:.2rem 0; color:#374151; font-size:.92rem; }
    .badge { display:inline-block; padding:2px 11px; border-radius:999px;
        font-size:.74rem; font-weight:700; color:#fff; vertical-align:middle; }
    .b-high{background:var(--red);} .b-med{background:var(--orange);}
    .b-low{background:var(--green);}
    .riskwrap{ display:flex; gap:10px; margin:10px 0; flex-wrap:wrap; }
    .riskpill{ flex:1; min-width:140px; border:1px solid var(--line);
        border-radius:9px; padding:8px 12px; background:var(--bg-soft); }
    .riskpill .lbl{ font-size:.68rem; color:var(--muted); text-transform:uppercase;
        letter-spacing:.05em; }
    .riskpill .val{ font-size:1rem; font-weight:700; }
    .v-high{color:var(--red);} .v-med{color:var(--orange);} .v-low{color:var(--green);}
    .conf{ font-size:.76rem; color:var(--muted); }
    .ev a{ font-size:.86rem; }
    .newsrow{ padding:8px 0; border-bottom:1px solid var(--bg-soft); }
    .newsrow .t{ font-weight:600; color:var(--ink); font-size:.94rem; }
    .newsrow .m{ font-size:.78rem; color:var(--muted); }
    .brief{ background:var(--bg-soft); border-radius:12px; padding:16px 20px;
        margin-bottom:10px; border-left:4px solid var(--nv); }
    .brief .q{ font-weight:700; color:var(--nv-dark); margin-bottom:4px; }
</style>
""", unsafe_allow_html=True)

LVL = {"High": "high", "Medium": "med", "Low": "low"}


def badge(level):
    return f"<span class='badge b-{LVL.get(level,'med')}'>{level}</span>"


def risk_pill(label, value):
    return (f"<div class='riskpill'><div class='lbl'>{label}</div>"
            f"<div class='val v-{LVL.get(value,'med')}'>{value}</div></div>")


def section(icon, title, sub=""):
    sub_html = f"<div class='s-sub'>{sub}</div>" if sub else ""
    st.markdown(
        f"<div class='sec'><div class='s-icon'>{icon}</div>"
        f"<div class='s-title'>{title}</div>{sub_html}</div>",
        unsafe_allow_html=True)


st.markdown(
    f"<div class='hero'><h1>{COMPANY} &mdash; Strategic Intelligence Agent</h1>"
    f"<p>AI CEO Agent &middot; live-collected intelligence &middot; evidence-based recommendations</p>"
    f"</div>", unsafe_allow_html=True)

source_set = sorted({d["source"] for d in docs})
last_update = "-"
if docs:
    dates = [d.get("scraped_at", "") for d in docs if d.get("scraped_at")]
    if dates:
        last_update = max(dates)[:10]

st.markdown(
    "<div class='kpis'>"
    f"<div class='kpi'><div class='k-label'>Company</div>"
    f"<div class='k-value k-accent'>{COMPANY}</div></div>"
    f"<div class='kpi'><div class='k-label'>Documents</div>"
    f"<div class='k-value'>{len(docs)}</div></div>"
    f"<div class='kpi'><div class='k-label'>Sources</div>"
    f"<div class='k-value'>{len(source_set)}</div></div>"
    f"<div class='kpi'><div class='k-label'>Last Update</div>"
    f"<div class='k-value'>{last_update}</div></div>"
    "</div>", unsafe_allow_html=True)
st.caption(f"Industry: {INDUSTRY}  -  Sources: {', '.join(source_set)}")

section("News", "Market Intelligence", "Most recent news & announcements")


def is_junk(d):
    t = d.get("title", "")
    return t.count("$") > 3 or "$$" in t


recent = sorted(
    [d for d in docs if d.get("date") and d["source"] in ("news", "nvidia_ir")
     and not is_junk(d)],
    key=lambda d: d["date"], reverse=True)[:10]

if recent:
    for d in recent:
        st.markdown(
            f"<div class='newsrow'><div class='t'>{d['title']}</div>"
            f"<div class='m'>{d['source']} &middot; {d['date'][:10]} &middot; "
            f"<a href='{d['url']}' target='_blank'>open</a></div></div>",
            unsafe_allow_html=True)
else:
    st.info("No dated documents available.")


def render_cards(findings, kind):
    if not findings:
        st.info("No findings available.")
        return
    for f in findings:
        impact = f.get("impact", "Medium")
        conf = f.get("confidence", 0.0)
        ev_html = ""
        if f.get("evidence"):
            links = "".join(
                f"<div class='ev'>- <a href='{e['url']}' target='_blank'>{e['title']}</a></div>"
                for e in f["evidence"])
            ev_html = f"<div style='margin-top:6px'><b style='font-size:.82rem'>Evidence</b>{links}</div>"
        st.markdown(
            f"<div class='card {kind}'><h4>{f['title']} {badge(impact)}</h4>"
            f"<p>{f.get('detail','')}</p>"
            f"<div class='conf'>Confidence: {conf:.0%}</div>{ev_html}</div>",
            unsafe_allow_html=True)


col_o, col_r = st.columns(2)
with col_o:
    section("Opp", "Opportunity Monitor")
    render_cards(analysis.get("opportunities", {}).get("findings", []), "opp")
with col_r:
    section("Risk", "Risk Monitor")
    render_cards(analysis.get("risks", {}).get("findings", []), "risk")

section("Data", "Sentiment Analysis", "VADER - scored per source")

if sentiment:
    overall = sentiment.get("overall_distribution", {})
    by_source = sentiment.get("by_source", {})
    trend = sentiment.get("trend", [])

    s1, s2 = st.columns([1, 2])
    with s1:
        st.markdown("**Overall distribution**")
        if overall:
            odf = pd.DataFrame(
                {"sentiment": list(overall.keys()),
                 "count": list(overall.values())}).set_index("sentiment")
            st.bar_chart(odf, color="#76b900")
    with s2:
        st.markdown("**By source**")
        if by_source:
            rows = [{
                "source": src, "avg": summ["avg_compound"],
                "label": summ["label"],
                "pos": summ["distribution"]["positive"],
                "neu": summ["distribution"]["neutral"],
                "neg": summ["distribution"]["negative"],
            } for src, summ in by_source.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)

    st.markdown("**Sentiment trend**")
    if trend:
        tdf = pd.DataFrame(trend)
        tdf["date"] = pd.to_datetime(tdf["date"])
        st.line_chart(tdf.set_index("date")[["avg_compound"]], color="#76b900")
else:
    st.info("Run sentiment.py to populate this section.")

section("Rec", "Strategic Recommendations", "prioritized - evidence-based")

rec_list = recs.get("recommendations", [])
if not rec_list:
    st.info("No recommendations available.")
else:
    for i, r in enumerate(rec_list, 1):
        prio = r.get("priority", "Medium")
        ra = r.get("risk_assessment", {})
        risk_html = ""
        if ra:
            risk_html = ("<div class='riskwrap'>"
                + risk_pill("Financial", ra.get("financial", "-"))
                + risk_pill("Operational", ra.get("operational", "-"))
                + risk_pill("Strategic", ra.get("strategic", "-"))
                + "</div>")
        ev_html = ""
        if r.get("evidence"):
            links = "".join(
                f"<div class='ev'>- <a href='{e['url']}' target='_blank'>{e['title']}</a></div>"
                for e in r["evidence"])
            ev_html = f"<div style='margin-top:6px'><b style='font-size:.82rem'>Supporting evidence</b>{links}</div>"
        st.markdown(
            f"<div class='card rec'><h4>{i}. {r['recommendation']} {badge(prio)}</h4>"
            f"<p>{r.get('rationale','')}</p>"
            f"<p><b>Expected impact:</b> {r.get('expected_impact','-')}</p>"
            f"{risk_html}{ev_html}</div>", unsafe_allow_html=True)

section("Brief", "CEO Briefing", "executive summary")

if briefing:
    for q, key in [("What happened?", "what_happened"),
                   ("Why does it matter?", "why_it_matters"),
                   ("What should management do next?", "what_next")]:
        st.markdown(
            f"<div class='brief'><div class='q'>{q}</div>"
            f"{briefing.get(key,'-')}</div>", unsafe_allow_html=True)
else:
    st.info("CEO Briefing not yet generated. Run briefing.py to populate this section.")