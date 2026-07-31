"""
FastAPI service exposing the email assistant.

Run:
    cd email_assistant
    uvicorn api:app --reload --port 8000

Then open http://localhost:8000/  for the demo UI, or
     http://localhost:8000/docs for the auto-generated Swagger page.

Endpoints
    GET  /              demo web UI
    GET  /health        service + model status
    GET  /samples       the labelled sample inbox (for the UI dropdown)
    POST /draft         one email  -> one reply draft
    POST /batch         many emails -> drafts sorted by priority
    POST /webhook/email integration stub: accepts a generic inbound-email
                        webhook payload (Gmail relay, Zapier, Twilio/WhatsApp
                        Business API bridge) and returns the draft
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from assistant import EmailAssistant
from config import COMPANY, SAMPLE_EMAILS

app = FastAPI(
    title="SafeX AI Email Assistant",
    description="Drafts reply suggestions for common customer emails.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

assistant = EmailAssistant()
BASE_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------------ schemas
class EmailIn(BaseModel):
    subject: str = Field(..., examples=["Website is down since this morning"])
    body: str = Field(..., examples=["We are getting a 502 error since 8am. This is urgent."])
    sender_name: str | None = Field(None, examples=["Priya Nair"])
    email_id: str | None = None


class BatchIn(BaseModel):
    emails: list[EmailIn]


class WebhookIn(BaseModel):
    """Loose shape matching common inbound-email webhook providers."""
    from_: str | None = Field(None, alias="from")
    subject: str = ""
    text: str = ""

    class Config:
        populate_by_name = True


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {
        "status": "ok",
        "company": COMPANY["name"],
        "llm_enabled": assistant.use_llm,
        "generator": "llm" if assistant.use_llm else "template",
        "intents": sorted(assistant.classifier.ml_classes_),
    }


@app.get("/samples")
def samples():
    df = pd.read_csv(SAMPLE_EMAILS)
    return {"count": len(df), "emails": df.to_dict(orient="records")}


@app.post("/draft")
def draft(email: EmailIn):
    if not email.subject.strip() and not email.body.strip():
        raise HTTPException(status_code=400, detail="subject or body is required")
    return assistant.draft_reply(
        subject=email.subject,
        body=email.body,
        sender_name=email.sender_name,
        email_id=email.email_id,
    ).to_dict()


@app.post("/batch")
def batch(payload: BatchIn):
    if not payload.emails:
        raise HTTPException(status_code=400, detail="emails list is empty")
    results = assistant.draft_batch([e.model_dump() for e in payload.emails])
    return {
        "count": len(results),
        "needs_review": sum(r.needs_human_review for r in results),
        "results": [r.to_dict() for r in results],
    }


@app.post("/webhook/email")
def webhook(payload: WebhookIn):
    """Integration stub.

    Point an inbound-email webhook (or a Twilio / WhatsApp Business API
    message relay) here. Returns the draft plus a routing decision so the
    caller knows whether it may auto-send or must queue for an agent.
    """
    sender = (payload.from_ or "").split("<")[0].strip() or None
    result = assistant.draft_reply(payload.subject, payload.text, sender_name=sender)
    return {
        "action": "queue_for_review" if result.needs_human_review else "ready_to_send",
        "priority": result.priority,
        "reply": {"subject": result.subject, "body": result.draft},
        "meta": {
            "intent": result.intent,
            "confidence": result.confidence,
            "generator": result.generator,
        },
    }


@app.get("/", response_class=HTMLResponse)
def index():
    html = BASE_DIR / "demo.html"
    if not html.exists():
        return HTMLResponse("<h1>SafeX AI Email Assistant</h1><p>See /docs</p>")
    return HTMLResponse(html.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
