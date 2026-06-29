"""
Executive Intelligence Dashboard for the NVIDIA CEO Agent (styled).

Reads cached JSON artifacts (no live LLM calls) and renders the 7 required
sections with a full visual style pass: color system, KPI tiles, styled
section headers, and cards. No data/logic changes from the working version.

"""

import json
import os

import pandas as pd
import streamlit as st

CLEAN_PATH = "data/clean/docs.json"
ANALYSIS_PATH = "data/analysis.json"
RECS_PATH = "data/recommendations.json"
RECS_VALIDATED_PATH = "data/recommendations_validated.json"
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
# Prefer the VALIDATED recommendations (post "Validate" stage); fall back to
# the raw recommendations if validation has not been run yet.
recs = load_json(RECS_VALIDATED_PATH, None) or load_json(
    RECS_PATH, {"recommendations": []})
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


def render_cards(findings, kind, low_confidence=False, lc_reason=""):
    if low_confidence:
        st.markdown(
            "<div style='background:#fff4e5;border-left:4px solid #e8870b;"
            "border-radius:8px;padding:8px 12px;margin-bottom:10px;"
            "font-size:.84rem;color:#92400e;'>"
            "&#9888; <b>Provisional &mdash; limited evidence.</b> "
            f"{lc_reason or 'The agent could not fully verify these findings.'}"
            "</div>", unsafe_allow_html=True)
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
    opp_block = analysis.get("opportunities", {})
    render_cards(opp_block.get("findings", []), "opp",
                 opp_block.get("low_confidence", False),
                 opp_block.get("low_confidence_reason", ""))
with col_r:
    section("Risk", "Risk Monitor")
    risk_block = analysis.get("risks", {})
    render_cards(risk_block.get("findings", []), "risk",
                 risk_block.get("low_confidence", False),
                 risk_block.get("low_confidence_reason", ""))

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

section("Rec", "Strategic Recommendations", "prioritized - evidence-based - validated")

rec_list = recs.get("recommendations", [])
if not rec_list:
    st.info("No recommendations available.")
else:
    for i, r in enumerate(rec_list, 1):
        prio = r.get("priority", "Medium")

        # Validation badge from the Validate stage.
        val = r.get("validation")
        val_badge = ""
        val_note = ""
        if val is not None:
            if val.get("passed"):
                val_badge = ("<span style='background:#2f9e44;color:#fff;"
                             "padding:2px 9px;border-radius:999px;font-size:.72rem;"
                             "font-weight:700;margin-left:6px;'>&#10003; Validated</span>")
            else:
                val_badge = ("<span style='background:#e8870b;color:#fff;"
                             "padding:2px 9px;border-radius:999px;font-size:.72rem;"
                             "font-weight:700;margin-left:6px;'>&#9888; Not validated</span>")
                reason = (val.get("llm_reason")
                          or "; ".join(val.get("problems", []))
                          or "did not pass validation")
                val_note = (f"<div style='background:#fff4e5;border-left:3px solid "
                            f"#e8870b;border-radius:6px;padding:6px 10px;margin:6px 0;"
                            f"font-size:.8rem;color:#92400e;'>"
                            f"<b>Validation flag:</b> {reason}</div>")

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
            f"<div class='card rec'><h4>{i}. {r['recommendation']} {badge(prio)}{val_badge}</h4>"
            f"{val_note}"
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