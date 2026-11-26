"""
news_prompt.py -- the ONE prompt and output schema every text model uses (spec: "Same input
and output for every text model"). The teacher (Gemini), the fine-tuned Llama 3.2 3B and the
zero-shot benchmarks all import from here, so a change made once applies to all of them.

Input per article: ticker, company, headline, summary. No dates and no prices.
Output: {"relevant": bool, "event_type": one of 10, "sentiment": -1..1, "confidence": 0..1}
The teacher also returns a one-sentence "rationale" (hand checks only, never trained on).
"""

import json

EVENT_TYPES = [
    "earnings", "guidance", "analyst_rating", "m_and_a", "product",
    "legal_regulatory", "management", "capital_return", "macro_sector", "other",
]

SYSTEM_PROMPT = """You label financial news for a research project. Each request is ONE news item and ONE company (ticker and name). Judge only from the text given. Do not use anything you may know about what happened to the stock or the company afterwards.

Return these fields:

relevant: true if the item is mainly about this company or directly affects its business. false if the company is only mentioned in passing, is one stock in a list or market roundup ("10 stocks moving", "52-week highs"), or only appears as the source of an opinion about another company. Example: "Wells Fargo raises price target on Alphabet" is NOT relevant to Wells Fargo, but IS relevant to Alphabet.

event_type: the single best category:
- earnings: quarterly or annual results, EPS or revenue beats and misses, earnings previews and dates
- guidance: the company raises, lowers or confirms its own outlook or forecast
- analyst_rating: analyst upgrades, downgrades, initiations, price target changes
- m_and_a: mergers, acquisitions, divestitures, spin-offs, large stake purchases or sales
- product: product launches, approvals, customer contracts, partnerships, recalls
- legal_regulatory: lawsuits, investigations, fines, regulation, antitrust
- management: executive or board changes, insider share purchases or sales
- capital_return: dividends, buybacks, stock splits
- macro_sector: the economy, interest rates, the overall market, or the company's industry as a whole
- other: anything else

sentiment: the likely effect of this news on THIS company's stock price over the next few days, from -1 (very negative) to 1 (very positive). Use 0 for neutral, mixed, or not relevant. Judge what the news means for the company, not the tone of the writing.

confidence: how sure you are about your labels, from 0 to 1."""

TEACHER_EXTRA = """

rationale: one short sentence explaining the labels. Write it first, then the labels."""


def format_input(ticker: str, company: str, title: str, summary: str) -> str:
    """The user message for one article. Never add dates or prices here."""
    summary = (summary or "").strip() or "(none)"
    return f"Ticker: {ticker}\nCompany: {company}\nHeadline: {title.strip()}\nSummary: {summary}"


LABEL_PROPERTIES = {
    "relevant": {"type": "boolean"},
    "event_type": {"type": "string", "enum": EVENT_TYPES},
    "sentiment": {"type": "number"},
    "confidence": {"type": "number"},
}

STUDENT_SCHEMA = {  # what the 3B and the benchmark models must return
    "type": "object",
    "properties": LABEL_PROPERTIES,
    "required": list(LABEL_PROPERTIES),
}

TEACHER_SCHEMA = {  # rationale first: the teacher reasons briefly, then labels
    "type": "object",
    "properties": {"rationale": {"type": "string"}, **LABEL_PROPERTIES},
    "required": ["rationale", *LABEL_PROPERTIES],
}


def parse_labels(text: str) -> dict:
    """Parse and check one model reply. Raises ValueError if it isn't usable.

    Numbers outside their range are clipped (a model saying 1.2 means 'very positive').
    """
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    missing = [k for k in LABEL_PROPERTIES if k not in obj]
    if missing:
        raise ValueError(f"missing fields {missing}")
    if obj["event_type"] not in EVENT_TYPES:
        raise ValueError(f"unknown event_type {obj['event_type']!r}")
    relevant = obj["relevant"]
    if isinstance(relevant, str):
        relevant = relevant.strip().lower() in ("true", "1", "yes")
    return {
        "relevant": int(bool(relevant)),
        "event_type": obj["event_type"],
        "sentiment": round(min(1.0, max(-1.0, float(obj["sentiment"]))), 3),
        "confidence": round(min(1.0, max(0.0, float(obj["confidence"]))), 3),
        "rationale": str(obj.get("rationale", "")).strip(),
    }
