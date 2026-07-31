"""
One-command demo / smoke test for the SafeX AI Email Assistant.

    cd email_assistant
    python run_demo.py            # full walkthrough, prints to console
    python run_demo.py --save     # also writes demo_output.txt

Useful for the screen recording: it exercises every part of the module
(extraction, classification, evaluation, drafting, triage) without needing
Jupyter or a browser.
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from assistant import EmailAssistant
from config import COMPANY, SAMPLE_EMAILS
from extractor import extract
from intent_classifier import HybridIntentClassifier

BAR = "=" * 78


def section(title: str) -> None:
    print(f"\n{BAR}\n  {title}\n{BAR}")


def main() -> None:
    section(f"{COMPANY['name']} - AI Email Assistant  |  Week 2 prototype")

    df = pd.read_csv(SAMPLE_EMAILS)
    print(f"Sample inbox : {len(df)} emails across {df.true_intent.nunique()} intents")
    print(df.true_intent.value_counts().to_string())

    # ---------------------------------------------------------- extraction
    section("1. Information extraction")
    row = df[df.email_id == "E025"].iloc[0]
    print(f"Email  : {row.subject}\n{row.body}\n")
    for k, v in extract(row.subject, row.body, row.sender_name).to_dict().items():
        print(f"  {k:22}: {v}")

    # ------------------------------------------------------- classification
    section("2. Intent classification")
    clf = HybridIntentClassifier().fit()
    preds = [clf.predict(r.subject, r.body) for r in df.itertuples()]
    acc = sum(p.intent == t for p, t in zip(preds, df.true_intent)) / len(df)
    print(f"Accuracy on labelled corpus : {acc:.1%}")

    hold = pd.read_csv(SAMPLE_EMAILS.parent / "holdout_emails.csv")
    hp = [clf.predict(r.subject, r.body) for r in hold.itertuples()]
    hacc = sum(p.intent == t for p, t in zip(hp, hold.true_intent)) / len(hold)
    print(f"Accuracy on holdout set     : {hacc:.1%}  (64.3% before rule tuning)")

    probe = clf.predict("Nothing loads on our portal",
                        "Since last night nobody can open the dashboard, it just spins forever.")
    print(f"\nProbe -> {probe.intent} ({probe.confidence:.1%}); top3 {probe.top_n(3)}")

    # ------------------------------------------------------------- drafting
    section("3. Drafted replies")
    bot = EmailAssistant()
    print(f"LLM polish: {'ON (Groq)' if bot.use_llm else 'OFF - template mode'}\n")

    for eid in ["E005", "E011", "E024"]:
        r_ = df[df.email_id == eid].iloc[0]
        res = bot.draft_reply(r_.subject, r_.body, r_.sender_name, eid)
        print(f"--- IN  [{eid}] {r_.sender_name}: {r_.subject}")
        print(f"    {r_.body[:150]}")
        print(f"--- OUT intent={res.intent} conf={res.confidence:.0%} "
              f"priority={res.priority} review={res.needs_human_review} via={res.generator}")
        print(f"Subject: {res.subject}\n{res.draft}\n")

    # --------------------------------------------------------------- triage
    section("4. Inbox triage (sorted by priority)")
    results = bot.draft_batch(df[["email_id", "sender_name", "subject", "body"]]
                              .to_dict(orient="records"))
    print(f"{len(results)} processed | "
          f"{sum(r.needs_human_review for r in results)} flagged for human review | "
          f"avg {sum(r.latency_ms for r in results)/len(results):.0f} ms per email\n")
    print(f"{'PRIO':>5}  {'ID':<5} {'INTENT':<18} {'REVIEW':<7} SUBJECT")
    for r in results[:12]:
        print(f"{r.priority:>5}  {r.email_id:<5} {r.intent:<18} "
              f"{'YES' if r.needs_human_review else '-':<7} {r.subject[:40]}")

    section("Done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="write demo_output.txt")
    args = ap.parse_args()

    if args.save:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        text = buf.getvalue()
        sys.stdout.write(text)
        # Always beside this script, not in whatever directory it was called from.
        out = Path(__file__).resolve().parent / "demo_output.txt"
        out.write_text(text, encoding="utf-8")
        print(f"\n[saved to {out.name}]")
    else:
        main()
