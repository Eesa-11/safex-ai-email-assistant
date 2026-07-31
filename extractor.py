"""
Lightweight information extraction from raw email text.

Pulls out the details a reply actually needs to reference: who is writing,
what they mentioned (invoice numbers, dates, budgets, services), how urgent
it is, and how happy they sound.

spaCy is used for person/organisation/date entities *if it is installed*
(`pip install spacy && python -m spacy download en_core_web_sm`). If it is
not available the module falls back to regex + a small lexicon, so the
prototype runs anywhere with zero downloads.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from config import NEGATIVE_KEYWORDS, URGENCY_KEYWORDS

# Optional spaCy ------------------------------------------------------------
_NLP = None
_SPACY_AVAILABLE = False
try:  # pragma: no cover - environment dependent
    import spacy

    try:
        _NLP = spacy.load("en_core_web_sm")
        _SPACY_AVAILABLE = True
    except OSError:
        _NLP = None
except ImportError:
    spacy = None  # type: ignore

# Matched on word boundaries, not substrings: "social media" must not trigger
# cybersecurity via "soc", and "web application" must not trigger mobile app.
SERVICE_LEXICON = {
    "web development": ["website", "web site", "web development", "corporate site",
                        "landing page", "web application", "web app"],
    "ecommerce": ["ecommerce", "e-commerce", "online store", "online shop", "payment integration"],
    "mobile app": ["mobile app", "mobile application", "ios", "android", "app store",
                   "delivery tracking app", "iphone app"],
    "cybersecurity": ["cybersecurity", "cyber security", "security audit", "security review",
                      "penetration testing", "pentest", "vulnerability", "phishing",
                      "ransomware", "network security", "security testing"],
    "digital marketing": ["seo", "ppc", "google ads", "social media", "digital marketing",
                          "marketing", "campaign", "ad spend", "rankings"],
    "creative media": ["branding", "photography", "videography", "logo", "creative media"],
    "consulting": ["it consulting", "consulting", "advisory"],
    "custom platform": ["saas", "custom platform", "client portal", "booking system",
                        "scheduling system", "dashboards", "inventory management"],
}

# Ordered so the most specific service wins when a phrase overlaps two entries.
_SERVICE_PATTERNS = {
    svc: [re.compile(rf"\b{re.escape(t)}\b", re.I) for t in terms]
    for svc, terms in SERVICE_LEXICON.items()
}

DOCUMENT_TERMS = {
    "company profile": [r"\bcompany profile\b", r"\bcompany overview\b"],
    "portfolio": [r"\bportfolio\b", r"\bprevious work\b", r"\bcase stud(y|ies)\b"],
    "contract": [r"\bcontract\b", r"\bagreement template\b"],
    "NDA": [r"\bnda\b", r"\bnon-?disclosure\b"],
}
_DOC_PATTERNS = {
    name: [re.compile(p, re.I) for p in pats] for name, pats in DOCUMENT_TERMS.items()
}

REFUND_RX = re.compile(
    r"\brefund\b|\bcharged twice\b|\bdouble charged\b|\bincorrect charge\b|"
    r"\boverchar(g|ge)d\b|\bstill (billed|charged)\b|\bpayment left our account\b|"
    r"\bcharged again\b", re.I)

_PLURALS = {"page": "pages", "endpoint": "endpoints", "server": "servers",
            "user": "users", "month": "months", "week": "weeks", "day": "days",
            "workstation": "workstations"}

MONEY_RX = re.compile(r"(?:[$£€]|PKR|USD|AED|Rs\.?)\s?\d[\d,]*(?:\.\d+)?(?:\s?[kKmM])?", re.I)
INVOICE_RX = re.compile(r"\binvoice\s*#?\s*(\d{3,})\b", re.I)
NUMBER_RX = re.compile(
    r"\b(\d+)\s+(pages?|endpoints?|servers?|users?|months?|weeks?|days?|workstations?)\b", re.I)
# A specific weekday ("next Monday") is a stronger scheduling signal than a vague
# one ("this week"), so weekdays are listed first and matches keep their order of
# appearance rather than being sorted alphabetically.
DATE_RX = re.compile(
    r"\b(?:(?:next|this|coming)\s+)?(?:(?:mon|tues|wednes|thurs|fri|satur|sun)day)\b|"
    r"\b(?:tomorrow|today|next week|this week)\b|"
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|"
    r"\bthe \d{1,2}(?:st|nd|rd|th)\b", re.I)
# am/pm is case-insensitive, but the trailing timezone must stay upper-case so
# "11am on Tuesday" does not capture "on" as a timezone.
TIME_RX = re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?i:am|pm)\b(?:\s+[A-Z]{2,4}\b)?")


@dataclass
class EmailFeatures:
    sender_first_name: str = "there"
    services_mentioned: list[str] = field(default_factory=list)
    documents_requested: list[str] = field(default_factory=list)
    invoice_numbers: list[str] = field(default_factory=list)
    is_refund_request: bool = False
    money_mentions: list[str] = field(default_factory=list)
    quantities: list[str] = field(default_factory=list)
    dates_mentioned: list[str] = field(default_factory=list)
    times_mentioned: list[str] = field(default_factory=list)
    organisations: list[str] = field(default_factory=list)
    urgency: str = "normal"          # low | normal | high
    sentiment: str = "neutral"       # negative | neutral | positive
    word_count: int = 0
    spacy_used: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _dedupe(items) -> list[str]:
    """Order-preserving de-duplication (case-insensitive)."""
    seen, out = set(), []
    for i in items:
        key = i.lower()
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out


def _quantity(number: str, unit: str) -> str:
    """'12 month' -> '12 months'; leaves already-plural units alone."""
    u = unit.lower()
    if number != "1" and not u.endswith("s"):
        u = _PLURALS.get(u, u + "s")
    return f"{number} {u}"


def _first_name(sender_name: str | None, body: str) -> str:
    if sender_name and sender_name.strip():
        return sender_name.strip().split()[0]
    # Fall back to a sign-off on the last line: "Thanks, Daniel"
    tail = body.strip().splitlines()[-1] if body.strip() else ""
    m = re.search(r"(?:thanks|regards|best|cheers|sincerely)[,!]?\s+([A-Z][a-z]+)", tail, re.I)
    return m.group(1) if m else "there"


def _urgency(text: str) -> str:
    hits = sum(1 for k in URGENCY_KEYWORDS if k in text)
    if hits >= 2 or re.search(r"\burgent(ly)?\b|\basap\b|\bimmediately\b", text):
        return "high"
    return "normal" if hits else "low"


def _sentiment(text: str) -> str:
    neg = sum(1 for k in NEGATIVE_KEYWORDS if k in text)
    pos = sum(1 for k in ("thank you", "thanks", "great", "appreciate", "impressed",
                          "love", "excellent") if k in text)
    if neg >= 2 or (neg and re.search(r"unacceptable|disappointed|very poor", text)):
        return "negative"
    if neg > pos:
        return "negative"
    return "positive" if pos > neg else "neutral"


def extract(subject: str, body: str, sender_name: str | None = None) -> EmailFeatures:
    text = f"{subject}\n{body}"
    low = text.lower()

    services = [
        svc for svc, patterns in _SERVICE_PATTERNS.items()
        if any(rx.search(text) for rx in patterns)
    ]
    documents = [
        name for name, patterns in _DOC_PATTERNS.items()
        if any(rx.search(text) for rx in patterns)
    ]

    feats = EmailFeatures(
        sender_first_name=_first_name(sender_name, body),
        services_mentioned=services,
        documents_requested=documents,
        invoice_numbers=_dedupe(INVOICE_RX.findall(text)),
        is_refund_request=bool(REFUND_RX.search(text)),
        money_mentions=_dedupe(m.strip() for m in MONEY_RX.findall(text)),
        quantities=_dedupe(_quantity(n, u) for n, u in NUMBER_RX.findall(text)),
        dates_mentioned=_dedupe(d.strip().lower() for d in DATE_RX.findall(text)),
        times_mentioned=_dedupe(t.strip() for t in TIME_RX.findall(text)),
        urgency=_urgency(low),
        sentiment=_sentiment(low),
        word_count=len(body.split()),
        spacy_used=_SPACY_AVAILABLE,
    )

    if _SPACY_AVAILABLE and _NLP is not None:  # pragma: no cover
        doc = _NLP(text)
        feats.organisations = sorted({e.text for e in doc.ents if e.label_ == "ORG"})
        spacy_dates = [e.text.lower() for e in doc.ents if e.label_ in ("DATE", "TIME")]
        feats.dates_mentioned = _dedupe(feats.dates_mentioned + spacy_dates)
        if sender_name is None:
            people = [e.text for e in doc.ents if e.label_ == "PERSON"]
            if people:
                feats.sender_first_name = people[-1].split()[0]

    return feats


if __name__ == "__main__":
    f = extract(
        "Invoice 4471 - double charged",
        "Hi, invoice 4471 was charged twice to our card on the 3rd for $1,200. "
        "Please refund urgently. Thanks, Lucia",
        "Lucia Moretti",
    )
    for k, v in f.to_dict().items():
        print(f"{k:22}: {v}")
