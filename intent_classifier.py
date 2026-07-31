"""
Hybrid intent classifier for inbound customer emails.

Two signals are combined:

1. Rule layer  - hand-written keyword/phrase patterns per intent. Fast,
   fully explainable, and reliable for the high-stakes intents (security
   incidents, complaints, billing) where a misroute is expensive.
2. ML layer    - TF-IDF (word + char n-grams) over the labelled sample
   corpus feeding a multinomial Logistic Regression. Generalises to
   phrasings the rules never anticipated.

The two probability distributions are blended:

    P(intent) = w_rule * P_rule + (1 - w_rule) * P_ml

This gives explainable behaviour on the cases that matter while still
degrading gracefully on unseen wording. Everything runs locally - no API
call is needed to classify an email.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from config import SAMPLE_EMAILS

# --------------------------------------------------------------------------
# Rule layer: (regex pattern, weight) per intent. Weights let a strong signal
# ("ransomware") outrank a weak, ambiguous one ("cost").
# --------------------------------------------------------------------------
RULES: dict[str, list[tuple[str, float]]] = {
    "security_incident": [
        (r"\bransomware\b", 3.0), (r"\bbreach(ed)?\b", 3.0), (r"\bhack(ed|ing)?\b", 2.5),
        (r"\bphishing\b", 3.0), (r"\bmalware\b", 2.5), (r"\bsuspicious (login|attachment|email)", 3.0),
        (r"\bfailed login\b", 2.5), (r"\bcompromis(ed|e)\b", 2.5), (r"\bunauthori[sz]ed\b", 2.0),
        (r"\bimpersonat(ing|e|ion)\b", 3.0), (r"\bpretending to be\b", 3.0),
        (r"\bgift cards?\b", 2.5), (r"\b(someone|somebody) (is )?(inside|in) our (system|network)\b", 3.0),
        (r"\bawareness training\b", 2.0), (r"\bemail security\b", 2.5),
    ],
    "technical_support": [
        (r"\b(not|isn'?t|stopped) working\b", 2.5), (r"\bsite (is )?down\b", 3.0),
        (r"\b50[0-9] error\b", 3.0), (r"\bbug\b", 2.0), (r"\berror\b", 2.0),
        (r"\bcannot (log ?in|access)\b", 2.5), (r"\bslow(ly)?\b", 1.5),
        (r"\bcrash(ing|es|ed)?\b", 2.5), (r"\bspam folder\b", 2.5), (r"\bfix\b", 1.5),
        (r"\bbroken\b", 2.0),
    ],
    "complaint": [
        (r"\bdisappoint(ed|ing)\b", 3.0), (r"\bunacceptable|not acceptable\b", 3.0),
        (r"\bpoor service\b", 3.0), (r"\bunhappy\b", 2.5), (r"\bfrustrat(ed|ing)\b", 2.5),
        (r"\bnot seeing value\b", 2.5), (r"\bthird time\b", 2.0), (r"\bnobody has\b", 2.0),
        (r"\bcomplain(t|ing)?\b", 2.5), (r"\bexplanation\b", 1.5),
    ],
    "pricing": [
        (r"\bhow much\b", 2.5), (r"\bwhat (do you )?charge\b", 2.5), (r"\bcost\b", 2.0),
        (r"\bpricing\b", 3.0), (r"\bprice\b", 2.0), (r"\bbudget\b", 2.0),
        (r"\bpackages?\b", 1.5), (r"\brates?\b", 1.5), (r"\bballpark\b", 2.0),
    ],
    "quote_request": [
        (r"\bquotation\b", 3.0), (r"\bquote\b", 3.0), (r"\bproposal for\b", 2.0),
        (r"\bformal (quote|quotation)\b", 3.0), (r"\bestimate\b", 2.0),
    ],
    "service_inquiry": [
        (r"\bdo you (offer|provide|do|build|handle)\b", 3.0), (r"\bis this something\b", 2.5),
        (r"\bcan you (build|develop|help with)\b", 2.0), (r"\btell (us|me) more about\b", 2.5),
        (r"\binterested in your\b", 2.5), (r"\bwhat (is|does) included\b", 1.5),
        (r"\bongoing (maintenance|support)\b", 2.0),
    ],
    "meeting_request": [
        (r"\bschedule a (call|meeting)\b", 3.0), (r"\bset up a (call|meeting)\b", 3.0),
        (r"\bdiscovery call\b", 3.0), (r"\bvideo (call|meeting)\b", 2.5),
        (r"\bare you available\b", 2.5), (r"\bwhat times work\b", 2.5),
        (r"\barrange a (call|meeting)\b", 3.0), (r"\bnext (week|monday|tuesday)\b", 1.0),
    ],
    "billing": [
        (r"\binvoice\b", 3.0), (r"\brefund\b", 3.0), (r"\bdouble charged|charged twice\b", 3.0),
        (r"\bpayment method\b", 3.0), (r"\bdeposit\b", 2.0), (r"\bbilled\b", 2.5),
        (r"\bcharge(d)? to (our|my) card\b", 3.0),
    ],
    "project_status": [
        (r"\bstatus of\b", 3.0), (r"\bmilestone\b", 3.0), (r"\bprogress (summary|update)\b", 3.0),
        (r"\bon track\b", 2.5), (r"\bupdated (delivery date|project plan)\b", 2.5),
        (r"\bremaining deliverables\b", 3.0), (r"\bwhere things stand\b", 2.5),
        (r"\bon schedule\b", 3.0), (r"\bstill on track\b", 3.0), (r"\boutstanding\b", 1.5),
        (r"\bphase (one|two|three|\d)\b", 2.0), (r"\bwhen we agreed\b", 2.5),
        (r"\bchecking in on\b", 2.0), (r"\bdelivery date\b", 2.5),
    ],
    "careers": [
        (r"\bintern(ship)?s?\b", 3.0), (r"\bapply(ing)? for\b", 2.5), (r"\bmy cv\b", 3.0),
        (r"\bresume\b", 3.0), (r"\bopen (positions?|roles?)\b", 3.0),
        (r"\bskill development program\b", 2.5), (r"\bdear hr\b", 3.0),
        (r"\btraining program(me)?\b", 3.0), (r"\bfresh graduate\b", 3.0),
        (r"\b(finished|completed) my (degree|studies)\b", 3.0), (r"\bjoin your (team|programme|program)\b", 2.5),
        (r"\bfinal year student\b", 3.0), (r"\bhiring\b", 2.0),
    ],
    "partnership": [
        (r"\bpartnership\b", 3.0), (r"\breferral partner\b", 3.0), (r"\breseller?\b", 3.0),
        (r"\bwhite label\b", 3.0), (r"\bcollaborat(e|ion)\b", 2.0),
        (r"\bworking together\b", 2.5), (r"\bbring you in\b", 3.0),
        (r"\b(some sort of |an )?arrangement\b", 2.0), (r"\bour clients (who|keep)\b", 2.0),
        (r"\bon behalf of our clients\b", 2.0),
    ],
    "follow_up": [
        (r"\bfollow(ing)? up\b", 3.0), (r"\bhave not received (any )?(reply|response)\b", 3.0),
        (r"\bsent an email last\b", 3.0), (r"\bchecking in again\b", 3.0),
        (r"\bmy earlier email\b", 3.0), (r"\bstill waiting\b", 2.5),
        (r"\bsecond (message|email)\b", 3.0), (r"\bnobody has replied\b", 3.0),
        (r"\bstill no (answer|reply|response)\b", 3.0), (r"\bnot received any reply\b", 3.0),
        (r"\bover a week (now|ago)\b", 2.0), (r"\bany update on my\b", 2.5),
    ],
    "general_info": [
        (r"\bcompany profile\b", 3.0), (r"\bportfolio\b", 2.0), (r"\bwhere are you located\b", 3.0),
        (r"\boffice is located\b", 3.0), (r"\bnda\b", 3.0), (r"\bcontract (and|template)\b", 2.5),
        (r"\bmore information about\b", 1.5),
    ],
}

INTENTS = sorted(RULES.keys())


@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    matched_rules: list[str] = field(default_factory=list)

    def top_n(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)[:n]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class HybridIntentClassifier:
    """Rule + TF-IDF logistic regression intent classifier."""

    def __init__(self, rule_weight: float = 0.6, csv_path: Path | str = SAMPLE_EMAILS):
        self.rule_weight = rule_weight
        self.csv_path = Path(csv_path)
        self.pipeline: Pipeline | None = None
        self.ml_classes_: list[str] = []
        self._compiled = {
            intent: [(re.compile(p, re.I), w) for p, w in pats]
            for intent, pats in RULES.items()
        }

    # ------------------------------------------------------------ training
    @staticmethod
    def _build_pipeline() -> Pipeline:
        return Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                         min_df=1, stop_words="english")),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                         sublinear_tf=True, min_df=1)),
            ])),
            ("clf", LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")),
        ])

    def fit(self, texts: list[str] | None = None, labels: list[str] | None = None):
        if texts is None or labels is None:
            df = pd.read_csv(self.csv_path)
            texts = (df["subject"] + ". " + df["body"]).tolist()
            labels = df["true_intent"].tolist()
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(texts, labels)
        self.ml_classes_ = list(self.pipeline.classes_)
        return self

    # ------------------------------------------------------------ scoring
    def _rule_scores(self, text: str) -> tuple[dict[str, float], list[str]]:
        raw, matched = {}, []
        for intent, patterns in self._compiled.items():
            score = 0.0
            for rx, weight in patterns:
                if rx.search(text):
                    score += weight
                    matched.append(f"{intent}:{rx.pattern}")
            raw[intent] = score
        return raw, matched

    def predict(self, subject: str, body: str) -> IntentPrediction:
        text = f"{subject}. {body}"
        raw_rule, matched = self._rule_scores(text)

        # Rules -> probability distribution (uniform when nothing matched).
        total = sum(raw_rule.values())
        if total > 0:
            rule_probs = {i: s / total for i, s in raw_rule.items()}
        else:
            rule_probs = {i: 1 / len(INTENTS) for i in INTENTS}

        # ML -> probability distribution.
        if self.pipeline is not None:
            proba = self.pipeline.predict_proba([text])[0]
            ml_probs = {c: float(p) for c, p in zip(self.ml_classes_, proba)}
        else:
            ml_probs = {i: 1 / len(INTENTS) for i in INTENTS}

        w = self.rule_weight if total > 0 else 0.0
        blended = {
            i: w * rule_probs.get(i, 0.0) + (1 - w) * ml_probs.get(i, 0.0)
            for i in INTENTS
        }
        s = sum(blended.values()) or 1.0
        blended = {i: v / s for i, v in blended.items()}

        best = max(blended, key=blended.get)
        return IntentPrediction(
            intent=best,
            confidence=round(blended[best], 4),
            scores={k: round(v, 4) for k, v in blended.items()},
            matched_rules=matched,
        )


if __name__ == "__main__":
    clf = HybridIntentClassifier().fit()
    demos = [
        ("Site keeps throwing an error", "Every page returns a 500 error since the update, please fix."),
        ("Ballpark figure?", "Roughly what would a 10 page marketing website cost us?"),
        ("Quick chat?", "Would your team be free for a 30 minute call on Thursday?"),
    ]
    for subj, body in demos:
        p = clf.predict(subj, body)
        print(f"{subj!r:40} -> {p.intent:18} ({p.confidence:.2f})  top3={p.top_n()}")
