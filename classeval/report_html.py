"""HTML rendering for the live pipeline status and the explainable report."""
from __future__ import annotations

from html import escape

from . import config

CLASS_COLORS = {"Excellent": "#16a34a", "Good": "#2563eb", "Average": "#d97706", "Poor": "#dc2626"}

CSS = """
.ce-wrap{font-family:var(--font);color:var(--body-text-color);}
.ce-card{background:var(--block-background-fill);border:1px solid var(--border-color-primary);border-radius:14px;padding:16px 18px;margin:0 0 12px;}
.ce-muted{color:var(--body-text-color-subdued);font-size:.88em;}
.ce-row{display:flex;gap:14px;flex-wrap:wrap;align-items:stretch;}
.ce-grow{flex:1 1 260px;min-width:0;}
.ce-h{font-weight:700;font-size:1.05em;margin:0 0 8px;}
.ce-bar{height:8px;border-radius:99px;background:var(--border-color-primary);overflow:hidden;}
.ce-bar>span{display:block;height:100%;border-radius:99px;transition:width .4s;}
.ce-stage{display:grid;grid-template-columns:22px minmax(0,1fr) 64px;gap:8px;align-items:center;padding:7px 0;border-bottom:1px dashed var(--border-color-primary);}
.ce-stage:last-child{border-bottom:none;}
.ce-stage .lbl{font-weight:600;font-size:.93em;}
.ce-stage .desc{font-size:.82em;color:var(--body-text-color-subdued);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.ce-note{font-size:.8em;margin-top:4px;padding:4px 8px;border-radius:8px;background:rgba(217,119,6,.12);color:#b45309;}
.ce-stage .t{text-align:right;font-variant-numeric:tabular-nums;font-size:.85em;color:var(--body-text-color-subdued);}
.ce-ico{width:18px;height:18px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:11px;color:#fff;}
.ce-spin{border:2px solid var(--border-color-primary);border-top-color:#2563eb;background:none;animation:cespin 0.9s linear infinite;}
@keyframes cespin{to{transform:rotate(360deg)}}
.ce-hero{display:flex;gap:22px;align-items:center;flex-wrap:wrap;}
.ce-badge{font-size:1.9em;font-weight:800;padding:10px 22px;border-radius:14px;color:#fff;letter-spacing:.5px;}
.ce-score{font-size:2.6em;font-weight:800;line-height:1;}
.ce-pill{display:inline-block;white-space:nowrap;padding:2px 9px;border-radius:99px;font-size:.78em;font-weight:600;border:1px solid var(--border-color-primary);margin-right:4px;}
.ce-crit{border-left:4px solid var(--border-color-primary);}
.ce-ev{display:flex;gap:8px;align-items:flex-start;margin:6px 0;font-size:.9em;}
.ce-ts{font-family:var(--font-mono);font-size:.82em;padding:1px 7px;border-radius:6px;background:var(--background-fill-secondary);border:1px solid var(--border-color-primary);white-space:nowrap;}
.ce-q{font-style:italic;}
.ce-tr{max-height:520px;overflow:auto;font-size:.9em;}
.ce-tr div{padding:3px 0;}
.ce-T{color:#2563eb;font-weight:700;}
.ce-S{color:#9333ea;font-weight:600;}
.ce-grid2{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;}
.ce-kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:.9em;}
.ce-thr{display:flex;height:10px;margin-top:6px;position:relative;}
.ce-thr span{display:block;height:100%;}
.ce-marker{position:absolute;top:-4px;width:3px;height:18px;background:var(--body-text-color);border-radius:2px;}
ul.ce-list{margin:4px 0 0 18px;padding:0;}
"""


def _color(score: float) -> str:
    return "#16a34a" if score >= 80 else "#2563eb" if score >= 65 else "#d97706" if score >= 45 else "#dc2626"


def _bar(score: float, color: str | None = None) -> str:
    s = max(0.0, min(100.0, score))
    return f'<div class="ce-bar"><span style="width:{s:.0f}%;background:{color or _color(s)}"></span></div>'


def _fmt_secs(s: float | None) -> str:
    if s is None:
        return ""
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m {int(s % 60):02d}s"


# ---- Live status -----------------------------------------------------------------
def status(stages: list[dict], overall: float, total_elapsed: float, error: str | None = None) -> str:
    icons = {
        "pending": '<span class="ce-ico" style="background:var(--border-color-primary)"></span>',
        "running": '<span class="ce-ico ce-spin"></span>',
        "done": '<span class="ce-ico" style="background:#16a34a">✓</span>',
        "cached": '<span class="ce-ico" style="background:#0891b2" title="from cache">⚡</span>',
        "skipped": '<span class="ce-ico" style="background:#94a3b8">–</span>',
        "error": '<span class="ce-ico" style="background:#dc2626">!</span>',
    }
    rows = []
    for s in stages:
        desc = s["desc"] or {"pending": "waiting", "cached": "loaded from cache", "skipped": "skipped"}.get(s["status"], "")
        bar = _bar(s["frac"] * 100, "#2563eb") if s["status"] == "running" else ""
        if s["key"] == "transcript" and s["status"] in ("pending", "running"):
            bar += ('<div class="ce-note">⏳ This step can take a while for long recordings '
                    '(roughly 5–6 min for a 1-hour class). Please keep this tab open.</div>')
        rows.append(
            f'<div class="ce-stage">{icons.get(s["status"], "")}'
            f'<div><div class="lbl">{escape(s["label"])}</div><div class="desc" title="{escape(desc)}">{escape(desc)}</div>{bar}</div>'
            f'<div class="t">{_fmt_secs(s["elapsed"])}</div></div>'
        )
    err = f'<div class="ce-card" style="border-color:#dc2626;color:#dc2626;margin-top:10px"><b>Error:</b> {escape(error)}</div>' if error else ""
    return (
        f'<div class="ce-wrap"><div class="ce-card"><div style="display:flex;justify-content:space-between;align-items:baseline">'
        f'<div class="ce-h">Pipeline progress</div><div class="ce-muted">{overall:.0%} · {_fmt_secs(total_elapsed)}</div></div>'
        f'{_bar(overall * 100, "#2563eb")}<div style="margin-top:8px">{"".join(rows)}</div></div>{err}</div>'
    )


# ---- Report ----------------------------------------------------------------------------
def hero(rep: dict) -> str:
    f = rep["final"]
    label, score = f["label"], f["score"]
    col = CLASS_COLORS[label]
    pillars = [("Teaching quality", "teaching", rep["teaching"]["score"]),
               ("Audio quality", "audio", rep["audio"]["score"] if rep.get("audio") else None),
               ("Video quality", "video", rep["video"]["score"] if rep.get("video") else None)]
    prow = []
    for name, key, sc in pillars:
        if sc is None:
            prow.append(f'<div class="ce-grow"><b>{name}</b> <span class="ce-muted">not available</span></div>')
            continue
        w = f["weights"].get(key, 0)
        prow.append(
            f'<div class="ce-grow"><div style="display:flex;justify-content:space-between"><b>{name}</b>'
            f'<span><b>{sc:.0f}</b>/100 <span class="ce-muted">× {w:.0%} = {f["contributions"][key]:.1f} pts</span></span></div>{_bar(sc)}</div>'
        )
    # threshold strip
    segs, prev = [], 100.0
    for name, lo in config.CLASS_THRESHOLDS:
        segs.append(f'<span title="{name} ≥ {lo:.0f}" style="width:{prev - lo}%;background:{CLASS_COLORS[name]};opacity:.75"></span>')
        prev = lo
    strip = (f'<div class="ce-thr" style="flex-direction:row-reverse">{"".join(segs)}'
             f'<div class="ce-marker" style="left:calc({score:.1f}% - 1px)"></div></div>'
             '<div class="ce-muted" style="position:relative;height:16px;font-size:.78em">'
             + "".join(f'<span style="position:absolute;left:{lo:.0f}%;transform:translateX(-50%)">{n} ≥ {lo:.0f}</span>'
                       for n, lo in config.CLASS_THRESHOLDS[:-1]) + "</div>")
    notes = "".join(f'<div class="ce-muted" style="margin-top:6px">⚠ {escape(n)}</div>' for n in f["notes"])
    topic = escape(rep["llm"].get("lesson_topic") or "")
    d = rep["transcript"]["dynamics"]
    return f"""<div class="ce-wrap"><div class="ce-card">
<div class="ce-hero"><div class="ce-badge" style="background:{col}">{label}</div>
<div><div class="ce-score">{score:.1f}<span class="ce-muted" style="font-size:.4em"> / 100</span></div>
<div class="ce-muted">Final weighted score{f' · topic: <b>{topic}</b>' if topic else ''} · {d['duration_min']} min class</div></div></div>
<div style="margin-top:14px">{strip}</div>
<div class="ce-row" style="margin-top:14px">{''.join(prow)}</div>{notes}
<div class="ce-muted" style="margin-top:10px">{escape(rep["llm"].get("summary") or "")}</div>
</div></div>"""


def _evidence(evs: list[dict]) -> str:
    out = []
    for e in evs:
        tick = {"verified": '<span title="quote found in transcript" style="color:#16a34a">✓</span>',
                "corrected": '<span title="quote found; timestamp corrected" style="color:#0891b2">✓*</span>'}.get(
            e.get("status"), '<span title="quote could not be matched to the transcript" style="color:#d97706">?</span>')
        pol = "#16a34a" if e.get("polarity") == "positive" else "#dc2626"
        out.append(
            f'<div class="ce-ev"><span class="ce-ts" style="border-left:3px solid {pol}">{escape(e.get("timestamp", ""))}</span>'
            f'<div><span class="ce-q">“{escape(e.get("quote", ""))}”</span> {tick}<br>'
            f'<span class="ce-muted">{escape(e.get("observation", ""))}</span></div></div>'
        )
    return "".join(out)


def teaching(rep: dict) -> str:
    t = rep["teaching"]
    llm = rep["llm"]
    lst = lambda xs: "".join(f"<li>{escape(x)}</li>" for x in (xs or []))  # noqa: E731
    head = f"""<div class="ce-row"><div class="ce-card ce-grow"><div class="ce-h">💪 Strengths</div><ul class="ce-list">{lst(llm.get('strengths'))}</ul></div>
<div class="ce-card ce-grow"><div class="ce-h">🎯 Improvements</div><ul class="ce-list">{lst(llm.get('improvements'))}</ul></div></div>"""
    how = f"""<div class="ce-card"><div class="ce-h">How this score was computed</div><div class="ce-muted">
Weighted mean of applicable criteria = <b>{t['base_score']:.1f}</b> − incident penalty <b>{t['penalty']:.1f}</b> = <b>{t['score']:.1f}</b>/100.
Core criteria always count; conditional ones only when the situation arose; bonus ones only when observed.
Evidence: {llm.get('evidence_check', {}).get('verified', 0)}/{llm.get('evidence_check', {}).get('checked', 0)} quotes matched to the transcript (✓ verified, ? unmatched).</div></div>"""

    inc_html = ""
    if t["incidents"]:
        items = []
        for i in t["incidents"]:
            sev = {"minor": "#d97706", "moderate": "#ea580c", "severe": "#dc2626"}.get(i["severity"], "#d97706")
            ver = "" if i.get("verified") else ' <span class="ce-muted">(quote not matched – half penalty)</span>'
            items.append(
                f'<div class="ce-ev"><span class="ce-ts" style="border-left:3px solid {sev}">{escape(i["timestamp"])}</span>'
                f'<div><span class="ce-pill" style="color:{sev}">{escape(i["severity"])}</span><span class="ce-pill">{escape(i["category"].replace("_", " "))}</span>'
                f'<b>−{i.get("penalty", 0):.1f}</b>{ver}<br><span class="ce-q">“{escape(i["quote"])}”</span><br>'
                f'<span class="ce-muted">{escape(i["explanation"])}</span></div></div>'
            )
        inc_html = f'<div class="ce-card" style="border-color:#dc2626"><div class="ce-h">⚠ Negative incidents (teacher)</div>{"".join(items)}</div>'
    else:
        inc_html = '<div class="ce-card"><div class="ce-h">✅ Negative incidents</div><div class="ce-muted">No frustration, shouting or harsh language detected from the teacher.</div></div>'

    cards = []
    for c in sorted(t["criteria"], key=lambda r: (not r["applicable"], -r["weight"])):
        s100 = c["score"] * 10
        kind = {"core": "core", "cond": "conditional", "bonus": "bonus"}[c["kind"]]
        if c["applicable"]:
            badge = f'<b style="font-size:1.25em;color:{_color(s100)}">{c["score"]}</b><span class="ce-muted">/10</span>'
            bar = _bar(s100)
        else:
            badge, bar = '<span class="ce-pill">not applicable</span>', ""
        cards.append(
            f'<div class="ce-card ce-crit" style="border-left-color:{_color(s100) if c["applicable"] else "var(--border-color-primary)"}">'
            f'<div style="display:flex;justify-content:space-between;gap:8px"><div class="ce-h">{escape(c["name"])}</div><div>{badge}</div></div>'
            f'{bar}<div style="margin:6px 0 2px"><span class="ce-pill">{kind}</span><span class="ce-pill">weight {c["weight"]:g}</span></div>'
            f'<div style="margin:6px 0">{escape(c.get("rationale", ""))}</div>{_evidence(c.get("evidence", []))}</div>'
        )
    return f'<div class="ce-wrap"><div class="ce-grid2" style="margin-bottom:12px">{"".join(cards)}</div>{head}{inc_html}{how}</div>'


def metrics(section: dict | None, title: str) -> str:
    if not section:
        return f'<div class="ce-wrap"><div class="ce-card">{title} was not evaluated (no stream).</div></div>'
    cards = []
    for m in sorted(section["metrics"].values(), key=lambda m: -m.get("weight", 0)):
        cards.append(
            f'<div class="ce-card"><div style="display:flex;justify-content:space-between;gap:8px"><div class="ce-h">{escape(m["name"])}</div>'
            f'<div><b style="color:{_color(m["score"])}">{m["score"]:.0f}</b><span class="ce-muted">/100</span></div></div>{_bar(m["score"])}'
            f'<div style="margin:6px 0 2px"><span class="ce-pill">{escape(str(m["value"]))}</span><span class="ce-pill">weight {m.get("weight", 0)}</span></div>'
            f'<div style="margin-top:6px">{escape(m["explanation"])}</div></div>'
        )
    info = section.get("info") or {}
    kv = "".join(f"<div class='ce-muted'>{escape(str(k).replace('_', ' '))}</div><div>{escape(str(v))}</div>" for k, v in info.items() if v not in (None, ""))
    return (f'<div class="ce-wrap"><div class="ce-card"><div class="ce-h">{title}: {section["score"]:.1f}/100</div>{_bar(section["score"])}'
            f'<div class="ce-muted" style="margin-top:6px">Weighted mean of the sub-scores below (no AI involved - pure signal analysis).</div>'
            f'{f"<div class=ce-kv style=margin-top:8px>{kv}</div>" if kv else ""}</div><div class="ce-grid2">{"".join(cards)}</div></div>')


def transcript(rep: dict) -> str:
    t = rep["transcript"]
    d = t["dynamics"]
    conf_col = {"high": "#16a34a", "medium": "#d97706", "low": "#dc2626"}[t["mapping_confidence"]]
    stats = "".join(
        f"<div class='ce-muted'>{escape(k.replace('_', ' '))}</div><div>{escape(str(v))}</div>" for k, v in d.items())
    from .transcript import fmt_ts

    lines = "".join(
        f'<div><span class="ce-ts">{fmt_ts(s["start"])}</span> '
        f'<span class="{"ce-T" if s["role"] == "TEACHER" else "ce-S"}">{"TEACHER" if s["role"] == "TEACHER" else "STUDENT"}</span>'
        f'{"" if s["role"] == "TEACHER" or t["source"] != "vtt" else f" <span class=ce-muted>({escape(s["speaker"])})</span>"}: '
        f'{escape(s["text"])}</div>'
        for s in t["segments"]
    )
    src = "uploaded Zoom transcript (.vtt)" if t["source"] == "vtt" else f"OpenAI {escape(rep['config']['transcribe_model'])}"
    return f"""<div class="ce-wrap"><div class="ce-row">
<div class="ce-card ce-grow"><div class="ce-h">Speaker mapping</div>
<div>Source: {src}</div><div>Teacher: <b>{escape(t['teacher_speaker'])}</b> <span class="ce-pill" style="color:{conf_col}">{t['mapping_confidence']} confidence</span></div>
<div class="ce-muted" style="margin-top:6px">{escape(t['mapping_note'])}</div></div>
<div class="ce-card ce-grow"><div class="ce-h">Class dynamics</div><div class="ce-kv">{stats}</div></div></div>
<div class="ce-card"><div class="ce-h">Transcript</div><div class="ce-tr">{lines}</div></div></div>"""
