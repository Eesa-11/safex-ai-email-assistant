"""
Deterministic reply templates, one per intent.

These serve two purposes:

1. **Offline fallback.** With no LLM key set, the assistant still returns a
   complete, sendable draft. The prototype is never a blank screen.
2. **Guardrail for the LLM.** The template is handed to the model as the
   skeleton it must follow, together with the facts it is allowed to state.
   This keeps the model from inventing prices, dates, or promises - the main
   risk of letting an LLM write customer-facing email unsupervised.

Placeholders are filled from `extractor.EmailFeatures` and `config.COMPANY`.
"""

from __future__ import annotations

from config import COMPANY
from extractor import EmailFeatures

# Guidance handed to the LLM per intent: the tone and the must-hit beats.
TONE_GUIDE: dict[str, str] = {
    "pricing": "Helpful and consultative. Never state a specific figure - explain that pricing depends on scope and offer a scoping call.",
    "quote_request": "Businesslike and prompt. Confirm receipt, list what you need from them to quote accurately, give a turnaround time.",
    "service_inquiry": "Warm and informative. Confirm the capability, give one or two concrete specifics, invite a conversation.",
    "technical_support": "Calm, urgent, accountable. Acknowledge impact, state that it is being investigated now, promise a concrete next update.",
    "complaint": "Apologetic without being defensive. Take ownership, avoid excuses, commit to a specific follow-up by a named team.",
    "billing": "Precise and reassuring. Reference the specific invoice or charge, explain the check being run, give a resolution window.",
    "meeting_request": "Friendly and efficient. Accept enthusiastically, propose concrete slots, mention what will be covered.",
    "project_status": "Transparent and organised. Give structure, commit to sending the detail, no vague reassurance.",
    "careers": "Encouraging and clear. Explain the process and next step without promising a role.",
    "partnership": "Interested but measured. Welcome the idea, ask the qualifying questions, propose an intro call.",
    "follow_up": "Apologetic for the delay, then immediately useful. Do not repeat the apology twice.",
    "general_info": "Polite and efficient. Confirm what is being sent and when.",
    "security_incident": "Serious, immediate, containment-first. Give holding actions the client can take right now, and escalate.",
}

# {name} {company} {agent} {email} {phone} {hours} {service_line} {detail_line}
TEMPLATES: dict[str, str] = {
    "pricing": """Hi {name},

Thanks for reaching out to {company} about {service_line}.

Our pricing is scoped per project rather than fixed-price, because the final figure depends on the features, integrations and timeline involved.{detail_line} If you can share a little more about your requirements, we can put together an accurate estimate - usually within two working days.

Would a short call this week work to walk through the scope? You can also reach us any time at {email} or {phone}.

Best regards,
{agent}
{company}""",

    "quote_request": """Hi {name},

Thank you for your quotation request regarding {service_line}. We have logged it and our solutions team is preparing the figures now.{detail_line}

To make the quote as accurate as possible, it would help to know your target launch date, any must-have integrations, and whether you need post-launch support included. We will send the full quotation within two working days of receiving those details.

Best regards,
{agent}
{company}""",

    "service_inquiry": """Hi {name},

Thanks for getting in touch with {company} - yes, we work on {service_line} regularly.{detail_line}

We handle projects end to end: discovery and scoping, design and build, testing, deployment, and ongoing support once you are live. We work with clients across 15+ countries from our base in Islamabad.

Happy to set up a short call to go through your specific requirements. You can reach us at {email} or {phone}.

Best regards,
{agent}
{company}""",

    "technical_support": """Hi {name},

Thanks for flagging this, and apologies for the disruption.{detail_line}

I have passed the details to our technical team and they are investigating now. We will confirm the cause and give you an update within the next few hours; if we need access or logs from your side, we will come back to you straight away.

If the situation changes or becomes more severe, call us directly on {phone} so we can escalate immediately.

Best regards,
{agent}
{company}""",

    "complaint": """Hi {name},

Thank you for taking the time to write, and I am sorry about the experience you have had - it is not the standard we hold ourselves to.{detail_line}

I am escalating this to the account lead responsible for your project so it gets proper attention rather than another generic reply. You will hear from us within one working day with a clear explanation and a plan to put things right.

If you would prefer to talk it through directly, call {phone} and ask for the customer success team.

Sincerely,
{agent}
{company}""",

    # Pre-sales / account questions about how billing works - no dispute implied.
    "billing_query": """Hi {name},

Thanks for checking before we proceed.{detail_line}

We accept bank transfer, card payment and online transfer, and invoices are issued at agreed milestones. For most projects we ask for a deposit up front, with the balance split across delivery milestones - the exact split is set out in the contract before any work begins.

If you let us know which method suits your finance team, we will structure the invoicing around it.

Best regards,
{agent}
{company}""",

    # Disputed or duplicate charges, refund requests.
    "billing_refund": """Hi {name},

Thanks for bringing this to our attention.{detail_line}

Our accounts team is reviewing the transaction record now. If a duplicate or incorrect charge is confirmed, the refund will be processed to the original payment method, and these typically clear within 5-7 working days depending on your bank.

We will write back with confirmation once the check is complete. Apologies for the inconvenience in the meantime.

Best regards,
{agent}
{company}""",

    "meeting_request": """Hi {name},

Thanks for reaching out - a call would be great.{detail_line}

Our team is generally available {hours}. If you can confirm a slot that suits you, we will send a calendar invite with a video link. Allow around 30 minutes so we can cover your requirements, our approach, and rough timelines.

Looking forward to speaking.

Best regards,
{agent}
{company}""",

    "project_status": """Hi {name},

Thanks for checking in.{detail_line}

Here is where things stand: the team is working through the current milestone, and I am pulling together an updated status summary with completed items, work in progress, and revised dates. You will have that in writing by the end of tomorrow.

If anything on your side is blocking us or the priorities have shifted, let us know and we will replan around it.

Best regards,
{agent}
{company}""",

    "careers": """Hi {name},

Thank you for your interest in joining {company}.

We review applications on a rolling basis. Please send your CV and any portfolio or GitHub links to {email} with the role or programme in the subject line. Shortlisted candidates are contacted for a short technical conversation, followed by a task relevant to the role.

We appreciate the interest and will be in touch if there is a good match.

Best regards,
{agent}
{company}""",

    "partnership": """Hi {name},

Thanks for reaching out about a partnership with {company} - this is the kind of arrangement we are open to.{detail_line}

To take it further, it would help to understand the markets and client profile you work with, the volume you would expect, and whether you are looking at referral or white-label delivery. From there we can propose a commercial structure that works for both sides.

Would a 30 minute introductory call suit you this week or next?

Best regards,
{agent}
{company}""",

    "follow_up": """Hi {name},

Apologies for the delay in coming back to you - your earlier message should not have gone unanswered.{detail_line}

I have picked this up personally and am chasing the relevant team now. You will have a substantive response within one working day. If it is time-sensitive, call {phone} and we will handle it live.

Thank you for your patience.

Best regards,
{agent}
{company}""",

    "general_info": """Hi {name},

Thanks for your interest in {company}.{detail_line}

{company} is a digital services and cybersecurity company headquartered in Islamabad, Pakistan, working with clients in 15+ countries across web and mobile development, cybersecurity, digital marketing, creative media, and IT consulting.

I am putting {documents_line} together and will send everything across shortly. Anything else you need in the meantime, just reply here or call {phone}.

Best regards,
{agent}
{company}""",

    "security_incident": """Hi {name},

Thank you for alerting us - we are treating this as a priority.{detail_line}

While our security team reviews the details, please take these immediate containment steps:
1. Disconnect any affected machine from the network, but do not power it off.
2. Reset credentials for the accounts involved, from a known-clean device.
3. Preserve logs and do not delete suspicious emails or files - they are evidence.
4. Do not pay or respond to any ransom or extortion message.

One of our security engineers will contact you directly. For anything time-critical, call {phone} and reference this email.

Regards,
{agent}
{company}""",
}

DEFAULT_TEMPLATE = TEMPLATES["general_info"]


_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _pretty(phrase: str) -> str:
    """Capitalise weekdays so extracted text does not read as sloppy lowercase."""
    return " ".join(
        w.capitalize() if w.lower() in _WEEKDAYS else w for w in phrase.split()
    )


def _detail_line(intent: str, feats: EmailFeatures) -> str:
    """A single sentence echoing back the specifics the customer gave us.

    Echoing concrete details is the cheapest way to make an automated draft
    read as though a human actually read the email.
    """
    if intent == "meeting_request":
        when = feats.dates_mentioned[:1] + feats.times_mentioned[:1]
        if when:
            return f" I have noted {_pretty(' '.join(when))} as your preference."
        return ""

    bits: list[str] = []
    if intent.startswith("billing") and feats.invoice_numbers:
        bits.append(f"invoice {feats.invoice_numbers[0]}")
    if feats.quantities:
        bits.append(feats.quantities[0])
    if feats.money_mentions:
        bits.append(feats.money_mentions[0])
    if not bits:
        return ""
    return f" I can see you mentioned {', '.join(bits)}, which we have noted on the record."


# Article-free noun phrases, so the same string reads correctly in every slot it
# appears in: "about X", "regarding X", "we work on X regularly".
NATURAL_SERVICE = {
    "web development": "web development",
    "ecommerce": "ecommerce development",
    "mobile app": "mobile app development",
    "cybersecurity": "cybersecurity",
    "digital marketing": "digital marketing",
    "creative media": "creative media",
    "consulting": "IT consulting",
    "custom platform": "custom platform development",
}

# Used when no service could be identified. Each slot needs its own wording -
# "we work on your requirements regularly" is not English.
FALLBACK_SERVICE = {
    "service_inquiry": "projects like this",
    "quote_request": "the work you described",
    "pricing": "your requirements",
}


def _join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _service_line(intent: str, feats: EmailFeatures) -> str:
    names = [NATURAL_SERVICE.get(s, s) for s in feats.services_mentioned]
    if not names:
        return FALLBACK_SERVICE.get(intent, "your requirements")
    return _join(names)


def _documents_line(feats: EmailFeatures) -> str:
    """What general_info says it is sending, based on what was actually asked for."""
    if not feats.documents_requested:
        return "the information you asked for"
    docs = [d if d == "NDA" else d for d in feats.documents_requested]
    return "the " + _join(docs) + (" documents" if len(docs) > 1 else "")


def resolve_intent(intent: str, feats: EmailFeatures) -> str:
    """Pick the template variant for intents that have more than one shape.

    The classifier answers 'what is this about'; some of those answers need a
    second question answered before the right template can be chosen. Billing is
    the clear case: 'what payment methods do you take' and 'you charged me twice'
    are the same intent but demand completely different replies.
    """
    if intent == "billing":
        return "billing_refund" if feats.is_refund_request else "billing_query"
    return intent


def render(intent: str, feats: EmailFeatures) -> str:
    """Fill the template for `intent` with the extracted features."""
    key = resolve_intent(intent, feats)
    template = TEMPLATES.get(key, DEFAULT_TEMPLATE)
    return template.format(
        name=feats.sender_first_name,
        company=COMPANY["name"],
        agent=COMPANY["agent_name"],
        email=COMPANY["email"],
        phone=COMPANY["phone"],
        hours=COMPANY["hours"],
        service_line=_service_line(intent, feats),
        documents_line=_documents_line(feats),
        detail_line=_detail_line(key, feats),
    )


def subject_line(intent: str, original_subject: str) -> str:
    prefix = "Re: "
    subj = original_subject.strip()
    return subj if subj.lower().startswith("re:") else prefix + subj
