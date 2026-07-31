# Week 2 Progress Report — AI Email Assistant Prototype

**Project:** Business Automation Research
**Organisation:** SafeX Solutions
**Module:** AI Email Assistant Prototype (individual contribution)
**Week:** 2

---

## 1. Objective

Week 1 delivered a group FAQ chatbot for the SafeX website. Week 2's task was to
take one component of the automation research forward individually: a prototype
that **drafts reply suggestions for common customer emails**.

The website chatbot handles a visitor who is already on the site. The inbox is the
other half of the same problem — and the more expensive half, because a person
writes every reply by hand.

---

## 2. What was built

A complete, runnable module (`email_assistant/`) with five parts:

1. **Feature extraction** (`extractor.py`) — regex + a service lexicon pull sender
   name, services mentioned, invoice numbers, amounts, quantities, dates and times
   out of raw email text, plus urgency and sentiment signals. spaCy NER is used
   automatically if installed, but is not required.

2. **Hybrid intent classifier** (`intent_classifier.py`) — 128 weighted regex
   patterns across 13 intents, blended 0.6/0.4 with a TF-IDF (word 1–2 grams +
   char 3–5 grams) → Logistic Regression model. Rules give explainable, dependable
   behaviour on the intents where a misroute is expensive; the ML layer generalises
   to phrasings no rule anticipated. Classification is fully local — no API call.

3. **Template engine** (`templates.py`) — 14 approved reply templates covering the
   13 intents, each filled with the specifics extracted from the customer's own
   email. `billing` splits into two variants because "what payment methods do you
   accept" and "you charged me twice" are the same intent but demand opposite
   replies; the extractor's refund signal picks between them.

4. **Orchestrator** (`assistant.py`) — runs the pipeline, optionally sends the
   template to Groq's free Llama 3.3 70B tier to be rewritten into natural prose,
   then computes a priority score and a routing decision.

5. **Service layer** (`api.py`, `demo.html`) — FastAPI with `/draft`, `/batch`,
   `/samples`, `/health` and a `/webhook/email` integration stub, plus a browser UI
   that processes the whole sample inbox and shows the triage queue.

Supporting deliverables: a labelled 40-email dataset, a 14-email holdout set, an
executed Jupyter notebook, and a console demo script.

---

## 3. Key design decision

**The LLM rewrites; it does not author.**

The obvious build is "send the email to an LLM, ask for a reply". That was rejected.
An LLM writing customer email unsupervised will eventually quote a price, promise a
delivery date, or invent a capability — and a company is bound by what its emails say.

Instead the template is produced first, deterministically, from facts SafeX has
approved. The LLM receives that template and is instructed to rewrite it naturally
while preserving every commitment, using only facts already present. If the API is
down, rate-limited, or unconfigured, the template draft is returned as-is.

The practical consequence: **the module cannot say anything SafeX has not already
agreed to say**, and it never fails to produce a draft.

---

## 4. Results

| Measure | Result |
|---|---|
| Intents supported | 13 |
| Labelled corpus | 40 emails |
| Accuracy on labelled corpus | 100% (optimistic — rules written against it) |
| Leave-one-out cross-validation | 100% |
| **Holdout set, first run** | **64.3%** (9/14) — the honest out-of-sample figure |
| Holdout after rule tuning | 100% (fair for today's rules, no longer a clean holdout) |
| Throughput (template mode) | 1-2 ms per email |
| Throughput (LLM polish) | ~1–2 s per email |
| Sample inbox flagged for human review | 11 of 40 (28%) |

The holdout result is the number worth reporting honestly. Fourteen emails were
written after the classifier was finished, deliberately paraphrased so they shared
little vocabulary with the training data. The first run scored 64.3%. Every failure
was a rule *coverage* gap rather than faulty logic — no pattern existed for "still
on schedule", "pretending to be our director", "second message", "bring you in", or
"training programme". Those patterns were added and the set now passes, but because
the errors were inspected first, the tuned score cannot be treated as clean. Real
performance on genuinely new phrasing sits between the two, and the fix is a real
corpus of tickets, not more hand-written regex.

---

## 5. How this is useful to SafeX

- **The 72% that is draft-ready.** Routine enquiries — pricing, quotes, service
  questions, meetings, careers — come back as complete drafts an agent skims and
  sends, rather than writes.
- **Triage by priority, not by timestamp.** A ransomware report sits at 100 and an
  internship CV at 20. Working the queue top-down means the expensive emails are
  seen first.
- **Consistency.** Every complaint gets the same structured, accountable response
  regardless of who is on shift.
- **Reuse.** The same classify-then-draft pipeline serves WhatsApp Business API or
  Twilio SMS by swapping the transport layer.

---

## 6. Challenges and how they were handled

| Challenge | Resolution |
|---|---|
| LLMs inventing prices and dates in customer email | Template-first architecture; the LLM only rewrites approved content |
| Small dataset (40 emails) makes ML alone unreliable | Hybrid — rules carry the high-stakes intents, ML generalises the rest |
| Demo must work without an API key or network | Full template fallback; every LLM failure path returns a usable draft |
| Hand-written rules overfit the data they were written against | Built a separate holdout set and reported the pre-tuning number honestly |
| Some intents genuinely need a person | Explicit always-review list plus confidence/sentiment/urgency flags |

---

## 7. Skills applied

- **Applied AI/ML** — hybrid rule + TF-IDF/Logistic Regression classification,
  leave-one-out evaluation, confusion-matrix analysis
- **Prompt and model design** — constrained system prompt where the model rewrites
  rather than authors; per-intent tone guidance
- **Data handling** — dataset design, labelling, pandas/NumPy analysis, holdout
  methodology
- **API integration** — Groq LLM API with timeout and failure fallback; FastAPI
  service with a webhook stub for inbound-email and messaging platforms
- **Technical documentation** — README, this report, an executed notebook, and a
  demo script

---

## 8. Status and next steps

**Status: code, dataset, notebook and documentation complete and runnable.** All of
it is in `email_assistant/`. Screenshots and the explanation video are the remaining
submission items — `screenshots/SCREENSHOT_GUIDE.md` lists the six shots to capture
and a video outline.

Next, in priority order:

1. **Feedback loop** — log every agent edit to a draft and retrain on the
   corrections, turning daily support work into training data.
2. **Real corpus** — a few thousand anonymised tickets would replace the synthetic
   accuracy figures with trustworthy ones.
3. **Thread awareness** — read conversation history, not just the latest message.
4. **Shared knowledge base** — merge `faq_data.json` from the Week 1 chatbot so
   email replies and the website chatbot answer factual questions identically.
5. **Channel expansion** — WhatsApp Business API / Twilio, reusing the same pipeline.
