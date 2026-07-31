"""
Central configuration for the SafeX AI Email Assistant.

Everything tunable lives here so the rest of the codebase stays clean and
the module can be re-pointed at a different company / LLM without edits.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_EMAILS = DATA_DIR / "sample_emails.csv"

# ---------------------------------------------------------------- company
COMPANY = {
    "name": "SafeX Solutions",
    "agent_name": "Customer Success Team",
    "email": "contact@safexsolutions.com",
    "phone": "+92 327 5781580",
    "website": "safexsolutions.com",
    "hours": "Mon-Fri, 9am-6pm PKT",
}

# ------------------------------------------------------------------- LLM
# Groq free tier (no credit card). Same provider used by the Week 1 chatbot,
# so one key powers both modules. If unset, the assistant silently falls back
# to deterministic templates and still produces usable drafts.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_TIMEOUT = 15
LLM_TEMPERATURE = 0.4
LLM_MAX_TOKENS = 400

# --------------------------------------------------------------- routing
# Intents a human must approve before anything is sent. Drafts for these are
# still generated, but flagged for review in the API response and the UI.
HUMAN_REVIEW_INTENTS = {"complaint", "security_incident", "billing"}

# Below this classifier confidence the draft is flagged as low-confidence.
CONFIDENCE_THRESHOLD = 0.35

# Words that push an email up the priority queue regardless of intent.
URGENCY_KEYWORDS = [
    "urgent", "urgently", "asap", "immediately", "critical", "emergency",
    "down", "outage", "breach", "ransomware", "hacked", "not working",
    "escalate", "deadline", "today",
]

NEGATIVE_KEYWORDS = [
    "disappointed", "unacceptable", "poor", "unhappy", "frustrated", "angry",
    "complaint", "refund", "cancel", "delay", "delayed", "no response",
    "still waiting", "third time", "not acceptable", "barely",
]
