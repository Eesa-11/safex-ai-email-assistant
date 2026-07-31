# SafeX Solutions — AI Email Assistant Prototype

**Business Automation Research · Week 2 individual contribution**

A prototype that reads an inbound customer email and produces a suggested reply
draft, a priority score, and a routing decision (send vs. queue for a human).

It builds directly on the group's Week 1 FAQ chatbot: same company, same free
Groq LLM tier, same "works offline, LLM optional" design. Where the chatbot
answers a visitor on the website, this module handles the inbox.

---

## What it does

| Step | Component | What happens |
|---|---|---|
| 1 | `extractor.py` | Pulls sender name, services mentioned, invoice numbers, money, quantities, dates/times, plus urgency and sentiment signals |
| 2 | `intent_classifier.py` | Classifies the email into one of **13 intents** using weighted regex rules blended with a TF-IDF → Logistic Regression model |
| 3 | `templates.py` | Renders an approved reply template for that intent, filled with the extracted details. A few intents have variants — "what payment methods do you take" and "you charged me twice" are both `billing` but need opposite replies |
| 4 | `assistant.py` | Optionally sends the template to an LLM to be **rewritten** into natural prose, then scores priority and decides whether a human must review |
| 5 | `api.py` | Serves all of it over FastAPI, with a browser demo UI |

```
raw email → extract → classify → template → LLM polish → priority + routing
```

### The design decision that matters

**The LLM is a rewriter, not an author.** It only ever receives facts that already
exist in the approved template. It cannot quote a price, promise a delivery date,
or invent a capability, because none of those are in what it was handed. That is
the difference between a demo and something that could sit in front of real
customers.

Three further rails:

- **Always-review intents** — complaints, billing and security incidents are never
  auto-sendable, regardless of confidence.
- **Confidence / sentiment / urgency flags** — low classifier confidence, negative
  sentiment, or urgent language each independently flag an email for a human.
- **Graceful degradation** — no API key, a rate limit, or a network failure returns
  the template draft rather than an error. The prototype is never a blank screen.

---

## Intents covered

`pricing` · `quote_request` · `service_inquiry` · `technical_support` · `complaint` ·
`billing` · `meeting_request` · `project_status` · `careers` · `partnership` ·
`follow_up` · `general_info` · `security_incident`

---

## Results

| Measure | Score | What it means |
|---|---|---|
| Labelled corpus (40 emails) | **100%** | Optimistic — the rules were written against this data |
| Leave-one-out CV | **100%** | ML layer never sees the email it scores, but the rule layer still has seen the corpus |
| **Holdout, first run** | **64.3%** (9/14) | The honest out-of-sample number |
| Holdout, after rule tuning | 100% | Fair measure of today's rules, but no longer a clean holdout |

`data/holdout_emails.csv` holds 14 emails written after the classifier was built and
deliberately paraphrased ("Ballpark figure needed", "Coffee or a call?", "Nothing
loads on our portal"). The first run scored 64.3%; all five failures were rule
*coverage* gaps, not wrong logic — nothing matched "still on schedule", "pretending
to be our director", "second message", "bring you in", or "training programme".
Those patterns were added and the set now passes, but since the errors were
inspected first, that 100% is no longer clean. Real-world performance on new
phrasing sits somewhere between the two figures, and the fix for that is a real
corpus, not more hand-written regex.

Throughput: **1-2 ms per email** in template mode (40 emails processed in well under
a second); ~1–2 s per email with LLM polish enabled, bounded by the Groq API.

Of the 40 sample emails, **11 (28%)** are flagged for human review and the rest are
draft-ready — which is the actual business case: an agent reviews a third of the
inbox instead of writing all of it.

---

## Running it

New to this / setting up from scratch? **[SETUP.md](SETUP.md)** has the full
step-by-step: virtual environment, installing requirements, running each piece,
and a troubleshooting table. The short version:

```bash
cd email_assistant
pip install -r requirements.txt

# Optional: natural-language polish via Groq's free tier (no credit card)
#   get a key at https://console.groq.com
export GROQ_API_KEY="your_key_here"        # Windows: set GROQ_API_KEY=...
```

**Console walkthrough** (best for the screen recording — exercises everything):

```bash
python run_demo.py --save        # also writes demo_output.txt
```

**Notebook**:

```bash
jupyter notebook notebooks/email_assistant_demo.ipynb
```

**API + browser UI**:

```bash
uvicorn api:app --port 8000
```

- UI: <http://localhost:8000/>
- Swagger docs: <http://localhost:8000/docs>

```bash
curl -X POST http://localhost:8000/draft \
  -H "Content-Type: application/json" \
  -d '{"subject":"Website is down","body":"502 error since 8am, this is urgent.","sender_name":"Priya Nair"}'
```

Individual modules also run standalone for quick checks:

```bash
python extractor.py          # extraction on one email
python intent_classifier.py  # classification probes
python assistant.py          # a single end-to-end draft
```

---

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Demo web UI |
| `GET` | `/health` | Service status, LLM on/off, intent list |
| `GET` | `/samples` | The labelled sample inbox |
| `POST` | `/draft` | One email → one draft + metadata |
| `POST` | `/batch` | Many emails → drafts sorted by priority |
| `POST` | `/webhook/email` | Inbound-webhook stub (Gmail relay / Zapier / Twilio–WhatsApp bridge); returns `ready_to_send` or `queue_for_review` |

`/draft` response (actual output for the curl above):

```json
{
  "email_id": null,
  "intent": "technical_support",
  "confidence": 0.7298,
  "top_intents": [["technical_support", 0.7298], ["service_inquiry", 0.0352], ["pricing", 0.0346]],
  "priority": 95,
  "urgency": "high",
  "sentiment": "neutral",
  "needs_human_review": true,
  "review_reasons": ["urgent language detected"],
  "subject": "Re: Website is down",
  "draft": "Hi Priya,\n\nThanks for flagging this, and ...",
  "generator": "template",
  "features": { "...": "..." },
  "latency_ms": 1
}
```

---

## Files

```
email_assistant/
├── config.py              company details, LLM settings, routing thresholds
├── extractor.py           regex + optional spaCy feature extraction
├── intent_classifier.py   hybrid rules + TF-IDF/LogReg classifier
├── templates.py           14 approved reply templates + per-intent tone guides
├── assistant.py           orchestrator: the full pipeline
├── api.py                 FastAPI service
├── demo.html              browser demo UI
├── run_demo.py            console walkthrough / smoke test
├── requirements.txt
├── SETUP.md               step-by-step install & run guide
├── data/
│   ├── sample_emails.csv    40 labelled customer emails
│   └── holdout_emails.csv   14 unseen paraphrased emails
├── demo_output.txt        captured output of the last run_demo.py run
├── notebooks/
│   └── email_assistant_demo.ipynb   executed, outputs and charts saved
└── screenshots/
    └── SCREENSHOT_GUIDE.md  what to capture, and the video outline
```

---

## Priority scoring

`priority = intent base + urgency bonus + negative-sentiment bonus`, capped at 100.

| Intent | Base | | Modifier | Value |
|---|---|---|---|---|
| security_incident | 95 | | high urgency | +15 |
| technical_support | 80 | | low urgency | −5 |
| complaint | 75 | | negative sentiment | +10 |
| billing | 60 | | | |
| follow_up | 58 | | | |
| quote_request | 55 | | | |
| meeting_request | 50 | | | |
| project_status | 48 | | | |
| pricing | 45 | | | |
| service_inquiry | 42 | | | |
| partnership | 35 | | | |
| general_info | 25 | | | |
| careers | 20 | | | |

An agent works the queue top-down rather than oldest-first: ransomware before
CV submissions.

---

## Known limits

- **40 synthetic emails is a proof of pipeline, not of accuracy.** The numbers above
  should be read as "the plumbing works", not "this is 100% accurate".
- **Single-message only.** No thread history, no attachment reading.
- **English only.** The corpus includes non-English greetings (*Assalam o alaikum*,
  *Bonjour*), but classification and drafting are English-only.
- **spaCy is optional and off by default** so the prototype installs anywhere; with
  `en_core_web_sm` installed, organisation/person/date extraction improves automatically.

## Next steps

- Log every agent edit to a draft and retrain on the corrections — daily support work
  becomes free training data.
- Thread awareness, so replies acknowledge what was already said.
- Swap the transport layer to serve WhatsApp Business API / Twilio SMS from the same
  classify-then-draft pipeline; only the length constraint in the templates changes.
- Share `faq_data.json` with the Week 1 chatbot so email and web chat answer factual
  questions identically.
