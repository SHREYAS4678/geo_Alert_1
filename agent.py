"""
GeoAlert - Agentic AI Alert Layer
===================================
This is what makes GeoAlert "agentic" rather than just "a classifier with a
threshold": the ensemble in ensemble.py produces a risk class + confidence,
but it is this agent that decides what to DO about it. Given the same
"Warning" reading, it might stay quiet (transient rain spike, trend already
falling) or escalate hard (soil saturation has been climbing for 3 hours
straight) — because it can call tools to look at context before deciding,
the same way a human on-call engineer would check recent history before
paging anyone.

Uses Google's Gemini API (free tier, no credit card required — see README
for how to get a key at aistudio.google.com). The SDK is `google-genai`,
Google's current unified Python SDK (the older `google-generativeai`
package is deprecated). Tools are plain Python functions with docstrings;
the SDK reads the docstring to build the tool schema and executes the calls
for us automatically ("automatic function calling") — https://ai.google.dev/gemini-api/docs/generate-content/function-calling

If GEMINI_API_KEY isn't set, the agent falls back to a deterministic
rule-based narrative so the rest of the pipeline (backend + dashboard)
still runs end-to-end without any API key configured — useful for a first
demo before you've grabbed a free key.
"""
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from datetime import datetime, timezone

import database as db

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ALERT_EMAIL_TO = os.environ.get("ALERT_EMAIL_TO", "")
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")


# --------------------------------------------------------------------------
# Tools the agent may call. Docstrings ARE the schema the model sees, so
# they're written for the model's benefit as much as for humans reading
# this file.
# --------------------------------------------------------------------------

def make_tools(node_id: str, send_log: list):
    """Bind the two tools to a specific node_id + a mutable send_log list
    (so app.py can find out afterwards whether an alert actually went out,
    without parsing the model's free-text response)."""

    def get_recent_trend(hours: float) -> dict:
        """Look up this monitoring node's recent sensor history and return
        summary statistics, so you can judge whether conditions are
        stable, rising, or already falling before deciding how urgently
        to respond.

        Args:
            hours: how many hours of history to summarise (e.g. 3.0 for the
                last 3 hours, 24.0 for the full day).
        """
        rows = db.recent_readings(node_id, hours=hours, limit=200)
        if not rows:
            return {"available": False, "note": "No prior history for this node yet."}
        soil = [r["soil_moisture_pct"] for r in rows if r["soil_moisture_pct"] is not None]
        tilt = [r["tilt_deg"] for r in rows if r["tilt_deg"] is not None]
        rain = [r["rainfall_1h_mm"] for r in rows if r["rainfall_1h_mm"] is not None]
        vib = [r["vibration_events_10min"] for r in rows if r["vibration_events_10min"] is not None]
        return {
            "available": True,
            "readings_count": len(rows),
            "soil_moisture_pct": {"first": soil[-1] if soil else None, "latest": soil[0] if soil else None,
                                   "max": max(soil) if soil else None},
            "tilt_deg": {"first": tilt[-1] if tilt else None, "latest": tilt[0] if tilt else None,
                         "max": max(tilt) if tilt else None},
            "rainfall_1h_mm_total": round(sum(rain), 2) if rain else 0,
            "vibration_events_total": sum(vib) if vib else 0,
            "trend": "rising" if (soil and soil[0] > soil[-1]) else "falling_or_flat",
        }

    def send_alert(severity: str, headline: str, message: str) -> dict:
        """Send a real alert email to the on-call recipient and log it.
        Only call this for 'Warning' or 'Critical' severity, or for
        'Watch' if get_recent_trend shows conditions are actively rising.
        Do not call this for routine 'Safe' readings.

        Args:
            severity: one of "Watch", "Warning", "Critical".
            headline: a short one-line summary suitable as an email subject
                (e.g. "Rising soil saturation + tilt drift at Node-01").
            message: the full alert body — plain language, 3-5 sentences,
                explaining what the sensors show and what a human should
                check or do next. No markdown formatting.
        """
        sent = _send_email(f"[GeoAlert · {severity}] {headline}", message)
        send_log.append({"severity": severity, "headline": headline, "message": message, "sent": sent})
        return {"status": "sent" if sent else "simulated_no_smtp_configured", "severity": severity}

    return [get_recent_trend, send_alert]


def _send_email(subject: str, body: str) -> bool:
    if not (ALERT_EMAIL_TO and SMTP_USER and SMTP_PASS):
        print(f"[agent] SMTP not configured — simulating send.\n  Subject: {subject}\n  Body: {body}")
        return False
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = ALERT_EMAIL_TO
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [ALERT_EMAIL_TO], msg.as_string())
        return True
    except Exception as e:
        print(f"[agent] Email send failed: {e}")
        return False


SYSTEM_INSTRUCTION = """You are GeoAlert's on-call AI agent for a landslide early-warning
network. You receive one sensor reading at a time plus a machine-learning
ensemble's risk classification (Safe / Watch / Warning / Critical) with its
confidence and the four individual models' opinions.

Your job is NOT to re-run the classification — trust the ensemble's label.
Your job is to decide the RESPONSE: whether this reading warrants sending a
real alert email right now, and if so, what it should say in plain language
a village official or site engineer can act on immediately.

Before deciding on anything above 'Safe', call get_recent_trend to check
whether conditions are rising, flat, or already improving — a 'Warning'
reading after 3 hours of steadily worsening soil saturation and tilt is far
more urgent than the same label after a brief rain spike that's already
passing. Use that context in your reasoning.

Only call send_alert for Warning or Critical, or for Watch when the trend
is clearly rising. For Safe, or a Watch that is stable/improving, do not
send an alert — just explain your reasoning briefly.

Always end your reply with a short plain-language summary of your decision
and why, even when you decide not to alert.
"""


def run_agent(node_id: str, features: dict, risk_label: int, risk_name: str,
              confidence: float, per_model: dict) -> dict:
    """Main entry point called by app.py after every ensemble prediction.
    Returns a dict with the agent's narrative and whether an alert was sent
    — safe to call even with GEMINI_API_KEY unset (falls back to a
    rule-based narrative so the rest of the system keeps working)."""
    send_log = []

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return _fallback_agent(node_id, features, risk_name, confidence, send_log)

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return _fallback_agent(node_id, features, risk_name, confidence, send_log,
                                note="google-genai not installed (pip install google-genai) — used rule-based fallback.")

    client = genai.Client(api_key=api_key)
    tools = make_tools(node_id, send_log)

    prompt = f"""Node: {node_id}
Timestamp: {datetime.now(timezone.utc).isoformat()}

Current reading:
{features}

Ensemble classification: {risk_name} (confidence {confidence:.0%})
Per-model breakdown: {per_model['ensemble']}

Decide the response."""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=tools,
                temperature=0.3,
            ),
        )
        narrative = response.text or "(agent returned no text)"
    except Exception as e:
        return _fallback_agent(node_id, features, risk_name, confidence, send_log,
                                note=f"Gemini call failed ({e}) — used rule-based fallback.")

    alert_sent = any(s["sent"] for s in send_log) or bool(send_log)
    return {
        "narrative": narrative,
        "alert_triggered": bool(send_log),
        "alert_sent": any(s["sent"] for s in send_log),
        "send_log": send_log,
        "mode": "gemini_agent",
    }


def _fallback_agent(node_id, features, risk_name, confidence, send_log, note=""):
    """Deterministic stand-in used only when no Gemini key is configured,
    so `python app.py` still works out of the box. It applies the same
    escalation rule the system prompt describes, just without an LLM
    writing the explanation."""
    should_alert = risk_name in ("Warning", "Critical")
    narrative = (
        f"[Rule-based fallback — no GEMINI_API_KEY set{': ' + note if note else ''}] "
        f"Node {node_id} classified as {risk_name} ({confidence:.0%} confidence). "
        f"{'Escalating: this severity crosses the alert threshold.' if should_alert else 'No alert sent: below the Warning threshold.'}"
    )
    if should_alert:
        headline = f"{risk_name} risk level at {node_id}"
        body = (f"Automated rule-based alert (Gemini agent unavailable): node {node_id} reads "
                f"{risk_name} with {confidence:.0%} ensemble confidence. Soil moisture "
                f"{features.get('soil_moisture_pct')}%, tilt {features.get('tilt_deg')}°, "
                f"rainfall (1h) {features.get('rainfall_1h_mm')}mm. Please verify on-site.")
        sent = _send_email(f"[GeoAlert · {risk_name}] {headline}", body)
        send_log.append({"severity": risk_name, "headline": headline, "message": body, "sent": sent})

    return {
        "narrative": narrative,
        "alert_triggered": bool(send_log),
        "alert_sent": any(s["sent"] for s in send_log),
        "send_log": send_log,
        "mode": "rule_based_fallback",
    }
