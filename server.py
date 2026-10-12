"""
Jev vs LLM race - a tiny demo server (Python standard library only, no pip install).

Run:  python server.py   then open http://localhost:8000
Keys are read from environment variables, or from a .env / config.env file next to this script.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


def load_env(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env(os.path.join(HERE, ".env"))
load_env(os.path.join(HERE, "config.env"))

JEV_API_KEY = os.environ.get("JEV_API_KEY", "")
JEV_BASE_URL = os.environ.get("JEV_BASE_URL", "https://kayaai.ir/api").rstrip("/")
JEV_MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-1.13")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.avalai.ir/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
PORT = int(os.environ.get("PORT", "8000"))

# ---------------------------------------------------------------------------
# The multi-step task: triage a customer support ticket.
# Every step is a yes/no decision that normally needs an LLM to "think".
# ---------------------------------------------------------------------------
STEPS = [
    {
        "id": "is_bug",
        "title": "آیا باگ نرم‌افزاری است؟",
        "instructions": "Is the customer reporting a software defect?",
        "criteria": {
            "true": "The customer describes broken or unexpected product behavior.",
            "false": "The customer is asking a question or requesting a feature.",
        },
    },
    {
        "id": "is_urgent",
        "title": "آیا فوری است؟",
        "instructions": "Does this ticket need urgent handling?",
        "criteria": {
            "true": "The issue blocks payment, causes data or money loss, a security risk, or the customer cannot use the product at all.",
            "false": "The issue is minor, cosmetic, a general question, or can wait for normal handling.",
        },
    },
    {
        "id": "is_angry",
        "title": "آیا مشتری ناراحت است؟",
        "instructions": "Is the customer frustrated, angry, or threatening to leave?",
        "criteria": {
            "true": "The message shows frustration, anger, complaints about repeated problems, or threats to cancel.",
            "false": "The message is calm, neutral, or polite.",
        },
    },
    {
        "id": "wants_refund",
        "title": "آیا درخواست بازگشت پول دارد؟",
        "instructions": "Is the customer asking for a refund, chargeback, or compensation?",
        "criteria": {
            "true": "The customer explicitly or implicitly asks to get money back or be compensated.",
            "false": "The customer does not ask for any money back or compensation.",
        },
    },
    {
        "id": "needs_human",
        "title": "ارجاع به کارشناس انسانی؟",
        "instructions": "Should this ticket be escalated to a senior human agent instead of an automated reply?",
        "criteria": {
            "true": "The case involves money, legal or security risk, an upset customer with an urgent problem, or something an automated reply cannot solve.",
            "false": "A clear automated reply with instructions is enough.",
        },
    },
]


def route_team(d):
    """Plain code (not AI) turns the decisions into a route."""
    if d.get("wants_refund"):
        return "Billing"
    if d.get("is_bug"):
        return "Engineering"
    return "Customer Success"


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
def post_json(url, key, payload, timeout=90):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            status = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        status = e.code
    ms = (time.perf_counter() - t0) * 1000
    try:
        data = json.loads(raw)
    except ValueError:
        data = {"raw_text": raw}
    if status >= 400:
        raise RuntimeError("HTTP %s from %s: %s" % (status, url, raw[:500]))
    return data, ms


def to_bool(v):
    """Normalise whatever shape an answer comes back in to True/False/None."""
    if isinstance(v, dict):
        if isinstance(v.get("noul"), (int, float)):  # Jev returns a probability
            return v["noul"] >= 0.5
        for k in ("value", "answer", "result", "decision"):
            if k in v:
                return to_bool(v[k])
        return None
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "yes", "1", "y"):
        return True
    if s in ("false", "no", "0", "n"):
        return False
    return None


def usage_of(data):
    u = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(u, dict):
        return None
    return {
        "prompt": u.get("prompt_tokens", u.get("input_tokens", 0)) or 0,
        "completion": u.get("completion_tokens", u.get("output_tokens", 0)) or 0,
        "total": u.get("total_tokens")
        or (u.get("prompt_tokens", u.get("input_tokens", 0)) or 0)
        + (u.get("completion_tokens", u.get("output_tokens", 0)) or 0),
    }


# ---------------------------------------------------------------------------
# Decision makers
# ---------------------------------------------------------------------------
def jev_decide(ticket, step):
    payload = {
        "model": JEV_MODEL,
        "state": {"ticket": ticket},
        "questions": {
            step["id"]: {
                "type": "noul",
                "instructions": step["instructions"],
                "criteria": step["criteria"],
            }
        },
    }
    data, ms = post_json(JEV_BASE_URL + "/decisions", JEV_API_KEY, payload)
    answers = data.get("answers", data) if isinstance(data, dict) else {}
    raw_answer = answers.get(step["id"]) if isinstance(answers, dict) else None
    return {
        "decision": to_bool(raw_answer),
        "confidence": raw_answer.get("noul") if isinstance(raw_answer, dict) else None,
        "reasoning": None,
        "ms": ms,
        "usage": usage_of(data),
        "request": payload,
        "response": data,
    }


def jev_batch(ticket):
    """All five decisions in a single Jev request (the questions map takes many keys)."""
    payload = {
        "model": JEV_MODEL,
        "state": {"ticket": ticket},
        "questions": {
            s["id"]: {"type": "noul", "instructions": s["instructions"], "criteria": s["criteria"]}
            for s in STEPS
        },
    }
    data, ms = post_json(JEV_BASE_URL + "/decisions", JEV_API_KEY, payload)
    answers = data.get("answers", {}) if isinstance(data, dict) else {}
    steps = {}
    for s in STEPS:
        a = answers.get(s["id"])
        steps[s["id"]] = {
            "decision": to_bool(a),
            "confidence": a.get("noul") if isinstance(a, dict) else None,
        }
    return {"steps": steps, "ms": ms, "usage": usage_of(data), "reasoning": None,
            "request": payload, "response": data}


def llm_chat(messages, max_tokens=400, temperature=0):
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data, ms = post_json(LLM_BASE_URL + "/chat/completions", LLM_API_KEY, payload)
    text = data["choices"][0]["message"]["content"] or ""
    return text, ms, usage_of(data), payload, data


def llm_decide(ticket, step):
    system = (
        "You are a support-ticket triage engine. Think step by step about the ticket, "
        "then give a final yes/no decision. Reply ONLY with JSON: "
        '{"reasoning": "<short reasoning>", "answer": true|false}'
    )
    user = (
        "Ticket:\n%s\n\nQuestion: %s\nAnswer true if: %s\nAnswer false if: %s"
        % (ticket, step["instructions"], step["criteria"]["true"], step["criteria"]["false"])
    )
    text, ms, usage, payload, data = llm_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    decision, reasoning = None, text
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            j = json.loads(m.group(0))
            decision = to_bool(j.get("answer"))
            reasoning = j.get("reasoning", text)
        except ValueError:
            pass
    return {
        "decision": decision,
        "reasoning": reasoning,
        "ms": ms,
        "usage": usage,
        "request": payload,
        "response": data,
    }


def write_reply(ticket, decisions, team):
    """Final step for both paths: only here do we need text generation."""
    facts = "\n".join("- %s: %s" % (k, v) for k, v in decisions.items())
    system = (
        "You write short, friendly customer-support replies (max 80 words) "
        "in the same language as the ticket. Follow the triage facts exactly."
    )
    user = "Ticket:\n%s\n\nTriage facts:\n%s\n- routed_team: %s\n\nWrite the reply." % (
        ticket,
        facts,
        team,
    )
    text, ms, usage, payload, data = llm_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=250,
        temperature=0.3,
    )
    return {"text": text, "ms": ms, "usage": usage, "request": payload, "response": data}


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send(self, code, obj, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self.send(200, f.read(), "text/html")
        if self.path.startswith("/fonts/") and self.path.endswith(".woff2"):
            name = os.path.basename(self.path)
            fp = os.path.join(HERE, "fonts", name)
            if os.path.exists(fp):
                with open(fp, "rb") as f:
                    return self.send(200, f.read(), "font/woff2")
        if self.path == "/api/config":
            return self.send(
                200,
                {
                    "steps": [{"id": s["id"], "title": s["title"]} for s in STEPS],
                    "jev_model": JEV_MODEL,
                    "llm_model": LLM_MODEL,
                    "keys_ok": bool(JEV_API_KEY and LLM_API_KEY),
                },
            )
        self.send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        try:
            if self.path == "/api/step":
                step = next(s for s in STEPS if s["id"] == body["step"])
                fn = jev_decide if body["path"] == "jev" else llm_decide
                return self.send(200, fn(body["ticket"], step))
            if self.path == "/api/jev_batch":
                return self.send(200, jev_batch(body["ticket"]))
            if self.path == "/api/reply":
                decisions = body.get("decisions", {})
                team = route_team(decisions)
                out = write_reply(body["ticket"], decisions, team)
                out["team"] = team
                return self.send(200, out)
            self.send(404, {"error": "not found"})
        except Exception as e:  # show any API error in the UI trace
            self.send(500, {"error": str(e)})


if __name__ == "__main__":
    if not (JEV_API_KEY and LLM_API_KEY):
        print("WARNING: JEV_API_KEY / LLM_API_KEY missing. Put them in .env (see .env.example).")
    print("Jev model: %s | LLM model: %s" % (JEV_MODEL, LLM_MODEL))
    print("Open http://localhost:%d" % PORT)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
