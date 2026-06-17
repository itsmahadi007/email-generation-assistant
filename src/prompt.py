"""The email-generation prompt.

Advanced prompting technique = THREE techniques combined:

  1. Role-playing  — the model adopts an expert-communications-writer persona,
                     which raises register, structure, and professionalism.
  2. Few-shot      — two worked (input -> email) exemplars anchor the output
                     format and quality bar so the model has a concrete target.
  3. Chain-of-thought with delimited output — the model first reasons (audience,
                     goal, how to weave each fact in, what to leave out) inside a
                     <thinking> block, THEN writes the final email inside an
                     <email> block. The hidden plan improves fact-coverage and
                     tone control; the delimiter lets us parse the email
                     deterministically and discard the reasoning.

We ask for ``Subject:`` + body so the output is a complete, ready-to-send email.
"""
from __future__ import annotations

import re

SYSTEM_PROMPT = """\
You are an expert professional communications writer who ghost-writes email for \
busy executives. You are known for email that is clear, appropriately concise, \
and perfectly calibrated in tone. You never invent facts, you weave every given \
detail in naturally (never as a bulleted dump), and you always include a subject \
line, an appropriate greeting, and a sign-off.

You will be given an email's INTENT, a list of KEY FACTS that must all appear, \
and the desired TONE.

Work in two steps:
1. Inside a <thinking> block, briefly plan: who the reader is, what outcome the \
email should drive, how you will weave in each key fact, and what to leave out \
to keep it tight and on-tone. Keep this under 120 words.
2. Inside an <email> block, write the final email only — starting with a \
"Subject:" line, then the greeting, body, and sign-off. Put nothing else in the \
<email> block. Match the requested tone precisely. Include every key fact. Do \
not add facts, names, dates, or numbers that were not provided.
"""

# Two few-shot exemplars. These are not part of the 10 evaluation scenarios.
FEWSHOT = [
    {
        "intent": "Reschedule a client call",
        "key_facts": [
            "Original call was set for Thursday 3pm",
            "A board meeting now conflicts",
            "Propose Friday 11am or Monday 2pm instead",
        ],
        "tone": "professional and apologetic",
        "email": (
            "<thinking>Reader is a client; goal is to move the call without "
            "friction. Lead with a brief apology, give the reason, then offer "
            "two concrete options to make saying yes easy. Keep it short and "
            "warm.</thinking>\n"
            "<email>\n"
            "Subject: Rescheduling our Thursday call\n\n"
            "Hi Dana,\n\n"
            "I'm sorry for the short notice — a board meeting has come up that "
            "conflicts with our 3pm call on Thursday. Would Friday at 11am or "
            "Monday at 2pm work for you instead? I'm happy to flex around your "
            "schedule.\n\n"
            "Thanks for understanding,\n"
            "Alex\n"
            "</email>"
        ),
    },
    {
        "intent": "Ask a teammate for a status update",
        "key_facts": [
            "Need the migration status by end of day",
            "Leadership review is tomorrow morning",
            "Offer to help if anything is blocked",
        ],
        "tone": "casual and supportive",
        "email": (
            "<thinking>Peer teammate; goal is a quick update without sounding "
            "like I'm nagging. Casual tone, explain the deadline pressure, and "
            "offer help so it reads as supportive.</thinking>\n"
            "<email>\n"
            "Subject: Quick migration check-in\n\n"
            "Hey Sam,\n\n"
            "Could you send me where the migration stands by end of day? I've "
            "got the leadership review first thing tomorrow and want to come in "
            "with an accurate picture. If anything's blocked, ping me — happy to "
            "jump in.\n\n"
            "Cheers,\n"
            "Alex\n"
            "</email>"
        ),
    },
]


def _format_request(intent: str, key_facts: list[str], tone: str) -> str:
    facts = "\n".join(f"- {f}" for f in key_facts)
    return (
        f"INTENT: {intent}\n"
        f"TONE: {tone}\n"
        f"KEY FACTS (every one must appear in the email):\n{facts}"
    )


def build_messages(intent: str, key_facts: list[str], tone: str) -> list[dict]:
    """Build the full message list: system + few-shot turns + the real request."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEWSHOT:
        messages.append(
            {"role": "user", "content": _format_request(ex["intent"], ex["key_facts"], ex["tone"])}
        )
        messages.append({"role": "assistant", "content": ex["email"]})
    messages.append({"role": "user", "content": _format_request(intent, key_facts, tone)})
    return messages


_EMAIL_RE = re.compile(r"<email>(.*?)</email>", re.DOTALL | re.IGNORECASE)


def extract_email(raw: str) -> str:
    """Pull the final email out of the <email> block.

    Falls back to stripping any <thinking> block if the model forgot the
    delimiters, so a slightly malformed response still yields a usable email.
    """
    match = _EMAIL_RE.search(raw)
    if match:
        return match.group(1).strip()
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


# Human-readable description of the technique, embedded verbatim in the report.
PROMPT_TEMPLATE_DOC = f"""\
### Advanced prompting technique

Three techniques are combined:

1. **Role-playing** — the system prompt casts the model as an expert
   executive-communications writer, raising register and structure.
2. **Few-shot examples** — two worked `(intent, key_facts, tone) -> email`
   exemplars anchor the output format and quality bar.
3. **Chain-of-thought with delimited output** — the model plans privately in a
   `<thinking>` block (reader, goal, fact-weaving, what to cut) and then writes
   the email in an `<email>` block. The plan improves fact coverage and tone
   control; the delimiter lets the harness parse the email deterministically and
   discard the reasoning.

#### System prompt

```
{SYSTEM_PROMPT}
```

#### Per-scenario user message format

```
INTENT: <intent>
TONE: <tone>
KEY FACTS (every one must appear in the email):
- <fact 1>
- <fact 2>
...
```
"""
