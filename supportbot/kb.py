"""Temporal knowledge base.

One JSON object per line. Each fact may carry:

- `available_at` (UTC timestamp): when support could first have known it
- `valid_from` / `valid_to` (local dates): when it started / stopped being true

A ticket asked at time T only sees facts with `available_at <= T` that have not
expired, so replaying past tickets never leaks answers learned later.
"""

import datetime as dt
import json
import math
import re
from collections import Counter, OrderedDict
from functools import lru_cache

RECENT_DAYS = 14
TEMPORARY_KINDS = ("incident", "promo", "status")


@lru_cache(maxsize=4)
def load(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def local(as_of_utc: str, tz_offset_hours: float = 0) -> dt.datetime:
    return dt.datetime.fromisoformat(as_of_utc[:19]) + dt.timedelta(hours=tz_offset_hours)


def facts_at(facts: list, as_of_utc: str, tz_offset_hours: float = 0, freeze_at_utc: str | None = None) -> list:
    """Facts visible at `as_of_utc`; `freeze_at_utc` caps the KB at an earlier snapshot."""
    cut = min(as_of_utc[:19], freeze_at_utc[:19]) if freeze_at_utc else as_of_utc[:19]
    today = local(as_of_utc, tz_offset_hours).date().isoformat()
    out = []
    for f in facts:
        if (f.get("available_at") or "")[:19] > cut:
            continue
        if f.get("valid_to") and f["valid_to"] < today:
            continue
        out.append(f)
    return out


def _span(f: dict) -> str:
    span = f.get("valid_from") or ""
    if f.get("valid_to"):
        span += f" ~ {f['valid_to']}"
    return span


def _line(f: dict, with_scope: bool = True) -> str:
    scope = f" (applies when: {f['applies_when']})" if with_scope and f.get("applies_when") else ""
    return f"- [{f['id']}|{f.get('kind', '')}|{_span(f)}] {f['title']}: {f['fact']}{scope}"


def render(facts: list, as_of_utc: str, tz_offset_hours: float = 0, brand: str = "") -> str:
    """The whole visible KB as one prompt block, recent temporary statuses first."""
    now = local(as_of_utc, tz_offset_hours)
    today = now.date().isoformat()
    since = (now.date() - dt.timedelta(days=RECENT_DAYS)).isoformat()
    lines = [f"# {brand} support knowledge base (as of {now:%Y-%m-%d %H:%M}, {len(facts)} entries)".replace("#  ", "# "),
             "Entry format: [id|kind|valid_from ~ valid_to] title: fact (applies when: ...). "
             "When several entries cover the same thing at different times, the latest valid_from wins."]
    recent = [f for f in facts if (f.get("kind") in TEMPORARY_KINDS or (f.get("valid_to") or "") >= today)
              and (f.get("valid_from") or "") >= since]
    shown = set()
    if recent:
        recent.sort(key=lambda f: f.get("valid_from") or "", reverse=True)
        lines.append(f"\n## Recent temporary status (last {RECENT_DAYS} days; overrides the regular rules below)")
        for f in recent:
            lines.append(_line(f, with_scope=False))
            shown.add(f["id"])
    groups = OrderedDict()
    for f in facts:
        if f["id"] not in shown:
            groups.setdefault(f.get("topic") or "Other", []).append(f)
    for topic, items in groups.items():
        items.sort(key=lambda f: (f.get("valid_from") or "", f["id"]))
        lines.append(f"\n## {topic}")
        lines += [_line(f) for f in items]
    return "\n".join(lines)


def render_brief(facts: list) -> str:
    return "\n".join(_line(f, with_scope=False) for f in facts)


# ---------------------------------------------------------------- retrieval

def tokens(text: str) -> list:
    """ASCII words plus CJK character bigrams, so mixed-language tickets retrieve well."""
    text = (text or "").lower()
    toks = re.findall(r"[a-z0-9][a-z0-9._\-]*", text)
    for seg in re.sub(r"[^一-鿿぀-ヿ가-힯]+", " ", text).split():
        toks += [seg[i:i + 2] for i in range(len(seg) - 1)] or [seg]
    return toks


def search(facts: list, query: str, k: int = 10) -> list:
    """BM25 over title/fact/applies_when of the given (already time-filtered) facts."""
    docs = [tokens(f"{f['title']} {f['title']} {f.get('applies_when') or ''} {f['fact']}") for f in facts]
    n = len(docs) or 1
    avg = (sum(map(len, docs)) / n) or 1
    df = Counter(t for d in docs for t in set(d))
    q = set(tokens(query))
    scored = []
    for f, d in zip(facts, docs):
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        if s > 0:
            scored.append((s, f))
    scored.sort(key=lambda x: -x[0])
    return [f for _, f in scored[:k]]
