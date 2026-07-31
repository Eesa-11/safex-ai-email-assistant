"""
EmailAssistant - the orchestrator that turns an inbound customer email into
a suggested reply draft.

Pipeline
--------
    raw email
       |
       v
  [1] extract()            regex + optional spaCy -> names, invoices, dates,
       |                   services, urgency, sentiment
       v
  [2] HybridIntentClassifier.predict()   rules + TF-IDF LogReg -> intent + confidence
       |
       v
  [3] templates.render()   deterministic, fact-safe draft skeleton
       |
       v
  [4] LLM polish (Groq)    rewrites the skeleton into natural prose, bound to
       |                   the facts in the skeleton. Skipped if no API key,
       |                   and any failure falls back to the skeleton.
       v
  [5] routing              priority score + human-review flag

Design decision worth calling out: the LLM is a *rewriter*, not an author.
It is only ever given facts that already appear in the template, so it cannot
quote a price or promise a date that SafeX has not agreed to. That is the
difference between a demo and something you could actually put in front of
customers.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import requests

import templates
from config import (
    CONFIDENCE_THRESHOLD,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_URL,
    HUMAN_REVIEW_INTENTS,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
)
from extractor import extract
from intent_classifier import HybridIntentClassifier

PRIORITY_BASE = {
    "security_incident": 95,
    "technical_support": 80,
    "complaint": 75,
    "billing": 60,
    "follow_up": 58,
    "quote_request": 55,
    "meeting_request": 50,
    "project_status": 48,
    "pricing": 45,
    "service_inquiry": 42,
    "partnership": 35,
    "general_info": 25,
    "careers": 20,
}

SYSTEM_PROMPT = """You are a customer support writer for {company}, a digital services and cybersecurity company based in Islamabad, Pakistan.

You will be given: a customer's email, the detected intent, a tone instruction, and an APPROVED DRAFT written from a company template.

Rewrite the approved draft so it reads like a thoughtful human wrote it specifically for this customer.

Hard rules:
- Use ONLY facts that appear in the approved draft or the customer's own email. Never invent prices, discounts, dates, names, staff members, guarantees, or capabilities.
- Keep every commitment in the approved draft (timeframes, next steps, contact details) intact and unchanged.
- Keep it under 180 words. Short paragraphs. No bullet lists unless the approved draft has them.
- Do not use placeholder text such as [Name] or [Company].
- Start with the greeting and end with the sign-off. Output the email body ONLY - no subject line, no preamble, no commentary.
"""


@dataclass
class DraftResult:
    email_id: str | None
    intent: str
    confidence: float
    top_intents: list[tuple[str, float]]
    priority: int
    urgency: str
    sentiment: str
    needs_human_review: bool
    review_reasons: list[str]
    subject: str
    draft: str
    generator: str                     # "llm" | "template"
    features: dict = field(default_factory=dict)
    latency_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class EmailAssistant:
    def __init__(self, use_llm: bool = True, rule_weight: float = 0.6):
        self.classifier = HybridIntentClassifier(rule_weight=rule_weight).fit()
        self.use_llm = use_llm and bool(GROQ_API_KEY)
        self.llm_available = bool(GROQ_API_KEY)

    # -------------------------------------------------------------- LLM
    def _polish(self, customer_email: str, intent: str, skeleton: str) -> str:
        from config import COMPANY

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(company=COMPANY["name"])},
                {"role": "user", "content": (
                    f"CUSTOMER EMAIL:\n{customer_email}\n\n"
                    f"DETECTED INTENT: {intent}\n"
                    f"TONE: {templates.TONE_GUIDE.get(intent, 'Professional and helpful.')}\n\n"
                    f"APPROVED DRAFT:\n{skeleton}\n\n"
                    "Rewrite the approved draft following the rules."
                )},
            ],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=LLM_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    # ----------------------------------------------------------- routing
    @staticmethod
    def _priority(intent: str, urgency: str, sentiment: str) -> int:
        score = PRIORITY_BASE.get(intent, 40)
        score += {"high": 15, "normal": 0, "low": -5}.get(urgency, 0)
        if sentiment == "negative":
            score += 10
        return max(1, min(100, score))

    # -------------------------------------------------------------- main
    def draft_reply(
        self,
        subject: str,
        body: str,
        sender_name: str | None = None,
        email_id: str | None = None,
    ) -> DraftResult:
        started = time.time()

        feats = extract(subject, body, sender_name)
        pred = self.classifier.predict(subject, body)

        skeleton = templates.render(pred.intent, feats)
        draft, generator = skeleton, "template"

        if self.use_llm:
            try:
                polished = self._polish(f"Subject: {subject}\n\n{body}", pred.intent, skeleton)
                if polished and len(polished.split()) > 20:
                    draft, generator = polished, "llm"
            except Exception:
                # Deliberate: a flaky/rate-limited API must never cost the
                # agent their draft. The template result stands.
                pass

        reasons: list[str] = []
        if pred.intent in HUMAN_REVIEW_INTENTS:
            reasons.append(f"'{pred.intent}' is a sensitive intent - always human-reviewed")
        if pred.confidence < CONFIDENCE_THRESHOLD:
            reasons.append(f"low classifier confidence ({pred.confidence:.2f})")
        if feats.sentiment == "negative":
            reasons.append("negative customer sentiment detected")
        if feats.urgency == "high":
            reasons.append("urgent language detected")

        return DraftResult(
            email_id=email_id,
            intent=pred.intent,
            confidence=pred.confidence,
            top_intents=pred.top_n(3),
            priority=self._priority(pred.intent, feats.urgency, feats.sentiment),
            urgency=feats.urgency,
            sentiment=feats.sentiment,
            needs_human_review=bool(reasons),
            review_reasons=reasons,
            subject=templates.subject_line(pred.intent, subject),
            draft=draft,
            generator=generator,
            features=feats.to_dict(),
            latency_ms=int((time.time() - started) * 1000),
        )

    def draft_batch(self, emails: list[dict]) -> list[DraftResult]:
        """Process an inbox. Results come back sorted by priority, highest first."""
        results = [
            self.draft_reply(
                subject=e.get("subject", ""),
                body=e.get("body", ""),
                sender_name=e.get("sender_name"),
                email_id=e.get("email_id"),
            )
            for e in emails
        ]
        return sorted(results, key=lambda r: r.priority, reverse=True)


if __name__ == "__main__":
    a = EmailAssistant()
    print(f"LLM enabled: {a.use_llm}\n" + "=" * 70)
    r = a.draft_reply(
        subject="Website is down since this morning",
        body=("Our website that your team built has been showing a 502 error since "
              "about 8am today. This is urgent as we are getting customer complaints."),
        sender_name="Priya Nair",
    )
    print(f"Intent: {r.intent} ({r.confidence:.2f}) | priority {r.priority} | "
          f"{r.urgency} urgency | {r.sentiment} | via {r.generator}")
    print(f"Review needed: {r.needs_human_review} {r.review_reasons}")
    print("-" * 70)
    print(f"Subject: {r.subject}\n\n{r.draft}")
