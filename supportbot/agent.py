"""The support workflow: triage -> investigate & answer (tool loop) -> verify (-> revise) -> safety.

Deterministic control flow around a handful of LLM stages:
  1. triage: split the ticket into asks and categorize them (drives which tools are mandatory)
  2. answer: tool-calling loop that must end in `submit_reply`; a reply is refused
     until the mandatory tools for the triaged categories have been called
  3. verify: an independent check of the draft against KB + tool evidence;
     critical findings drive up to MAX_REVISIONS revision rounds
  4. safety: leftover findings are acted on narrowly (escalate, strip refund promises),
     long replies get one compression pass, and only actions the reply cites survive

A case is a plain dict:
  {"ticket_id": 1, "user_id": 42, "subject": "...", "as_of": "2026-01-01T08:00:00" (UTC),
   "history":  [{"t": utc, "role": "user|agent|ai", "text": "..."}],        # earlier turns, optional
   "messages": [{"t": utc, "text": "...", "images": ["https://... or data:image/..."]}]}  # unanswered
"""

import datetime as dt
import json
import re
import time

from . import kb, llm, prompts, tools
from .config import cfg

MAX_TURNS = 10
MAX_REVISIONS = 2
MAX_IMAGES = 4
REFUND_WORDS = re.compile(r"refund|reimburse|compensat|money back|credit(ed)? back|退款|退还|退回|补偿|赔偿|补发", re.I)


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _args(call) -> dict:
    try:
        return json.loads(call.get("arguments") or "{}")
    except ValueError:
        return {}


class Run:
    def __init__(self, case: dict, effort: str | None = None, kb_freeze_utc: str | None = None):
        c = cfg()
        self.cfg = c
        self.case = case
        case.setdefault("as_of", utcnow())
        self.effort = effort or c.effort
        # side steps (triage, rewrites, safety) never think harder than the answer itself
        self.aux_effort = "medium" if self.effort in ("medium", "high", "xhigh", "max") else self.effort
        self.ctx = tools.ToolCtx(case.get("user_id"), case["as_of"], case.get("ticket_id"))
        self.facts = kb.facts_at(kb.load(c.kb_file), case["as_of"], c.tz_offset_hours, kb_freeze_utc)
        policy = prompts.POLICY.format(brand=c.brand, now=self._local(case["as_of"]))
        if c.policy_file:
            with open(c.policy_file, encoding="utf-8") as f:
                policy += "\n\n## Business rules\n" + f.read().strip()
        images = [u for m in case["messages"] for u in m.get("images") or []][:MAX_IMAGES]
        self.base = [
            llm.dev(policy),
            llm.dev(kb.render(self.facts, case["as_of"], c.tz_offset_hours, c.brand)),
            llm.user(self.ticket_text(), images),
        ]
        self.trace = {"ticket_id": case.get("ticket_id"), "kb_facts": len(self.facts), "stages": [], "tool_calls": [],
                      "usage": {"input": 0, "cached": 0, "output": 0}, "llm_calls": 0}

    def _local(self, utc: str) -> str:
        return kb.local(utc, self.cfg.tz_offset_hours).strftime("%Y-%m-%d %H:%M")

    def ticket_text(self) -> str:
        case = self.case
        lines = [f"Ticket #{case.get('ticket_id', '')}  Subject: {case.get('subject', '')}"]
        if case.get("history"):
            lines.append("[Earlier conversation (context; what was already handled needs no second answer)]")
            for h in case["history"]:
                lines.append(f"[{self._local(h['t'])}] {h['role']}: {h.get('text') or ''}")
            lines.append("[The customer's latest, unanswered messages (reply to these)]")
        n = 0
        for m in case["messages"]:
            tag = ""
            for _ in m.get("images") or []:
                n += 1
                tag += f" [image {n}]"
            stamp = f"[{self._local(m['t'])}] " if m.get("t") else ""
            lines.append(f"{stamp}customer: {m.get('text') or ''}{tag}")
        return "\n".join(lines)

    def _call(self, items, tool_list, tool_choice=None, effort=None):
        resp = llm.respond(items, tools=tool_list, tool_choice=tool_choice, effort=effort or self.effort)
        u = llm.usage(resp)
        for k in u:
            self.trace["usage"][k] += u[k]
        self.trace["llm_calls"] += 1
        return resp

    # -- stage 1
    def triage(self) -> dict:
        # Categorizing needs no KB; skipping it keeps this call small.
        items = [self.base[0], self.base[2], llm.dev(prompts.TRIAGE_INSTR)]
        resp = self._call(items, [prompts.TRIAGE_TOOL],
                          tool_choice={"type": "function", "name": "submit_triage"}, effort=self.aux_effort)
        calls = llm.function_calls(resp)
        tri = _args(calls[0]) if calls else {"issues": [], "summary": "", "screenshot_text": ""}
        self.trace["stages"].append({"stage": "triage", "out": tri})
        return tri

    def required_tools(self, tri: dict) -> set:
        req = set()
        if self.case.get("user_id") is None:
            return req  # anonymous ticket: there is no account to look up
        for it in tri.get("issues", []):
            if it.get("about_own_account"):
                req |= tools.required_for(it.get("category"))
        return req

    # -- stage 2
    def answer(self, tri: dict, convo: list | None = None, revise_issues: list | None = None) -> tuple:
        req = self.required_tools(tri)
        note = f"Before submitting you must have called: {', '.join(sorted(req))}." if req else ""
        if convo is None:
            query = self.ticket_text() + " " + " ".join(i.get("ask", "") for i in tri.get("issues", []))
            hits = kb.search(self.facts, query, k=12)
            # per-ask retrieval surfaces the entry for each sub-question of multi-part tickets
            seen = {f["id"] for f in hits}
            for it in tri.get("issues", [])[:4]:
                for f in kb.search(self.facts, f"{it.get('ask', '')} {it.get('category', '')}", k=5):
                    if f["id"] not in seen and len(hits) < 24:
                        hits.append(f)
                        seen.add(f["id"])
            self.trace["retrieved"] = [f["id"] for f in hits]
            convo = self.base + [
                llm.dev("## Knowledge-base entries most relevant to this ticket (retrieved automatically; the full "
                        "knowledge base is above)\n" + kb.render_brief(hits)),
                llm.dev(prompts.ANSWER_INSTR.format(triage=json.dumps(tri, ensure_ascii=False), required_note=note))]
        if revise_issues:
            convo = convo + [llm.dev(prompts.REVISE_INSTR.format(issues="\n".join(f"- {x}" for x in revise_issues)))]
        tool_list = tools.specs() + [prompts.SEARCH_TOOL, prompts.SUBMIT_TOOL]
        called = {c["name"] for c in self.trace["tool_calls"]}
        for turn in range(MAX_TURNS):
            force = turn == MAX_TURNS - 1
            resp = self._call(convo, tool_list,
                              tool_choice={"type": "function", "name": "submit_reply"} if force else "required")
            calls = llm.function_calls(resp)
            convo = convo + llm.replay_items(resp)
            if not calls:
                convo.append(llm.dev(prompts.NUDGE))
                continue
            submitted = None
            for c in calls:
                if c["name"] == "submit_reply":
                    missing = req - called
                    if missing and not force:
                        out = json.dumps({"error": f"Not verified yet: call {', '.join(sorted(missing))} first"})
                    else:
                        submitted = _args(c)
                        out = json.dumps({"ok": True})
                elif c["name"] == "search_kb":
                    q = _args(c).get("query") or ""
                    out = kb.render_brief(kb.search(self.facts, q, k=8)) or "(no matching entries)"
                    self.trace["tool_calls"].append({"name": "search_kb", "args": {"query": q}, "secs": 0, "output": out})
                else:
                    t0 = time.time()
                    out = tools.run(self.ctx, c["name"], c.get("arguments"))
                    called.add(c["name"])
                    # full text: verify reads its evidence from here
                    self.trace["tool_calls"].append({"name": c["name"], "args": _args(c),
                                                     "secs": round(time.time() - t0, 2), "output": out})
                convo.append({"type": "function_call_output", "call_id": c["call_id"], "output": out})
            if submitted is not None:
                return submitted, convo
        return {"decision": "handoff", "reply": prompts.FALLBACK_REPLY, "handoff_reason": "no answer within the turn limit",
                "internal_note": "", "kb_ids": [], "actions": [], "confidence": 0, "close_ticket": False}, convo

    # -- stage 3
    def verify(self, draft: dict) -> dict:
        evidence = "\n\n".join(f"### Tool {c['name']}({json.dumps(c['args'], ensure_ascii=False)})\n{c['output']}"
                               for c in self.trace["tool_calls"]) or "(no tools were called)"
        actions = [self.ctx.actions[a]["summary"] for a in draft.get("actions") or [] if a in self.ctx.actions]
        items = self.base + [llm.dev(prompts.VERIFY_INSTR), llm.dev(
            f"## Tool evidence\n{evidence}\n\n## Reply about to be sent\nDecision: {draft.get('decision')}\n"
            f"Reply: {draft.get('reply')}\nHandoff reason: {draft.get('handoff_reason')}\n"
            f"Actions to execute on send: {'; '.join(actions) or 'none'}\n"
            f"Close ticket after sending: {'yes' if draft.get('close_ticket') else 'no'}")]
        resp = self._call(items, [prompts.VERIFY_TOOL], tool_choice={"type": "function", "name": "submit_verdict"})
        calls = llm.function_calls(resp)
        v = _args(calls[0]) if calls else {"issues": [{"severity": "critical", "kind": "other",
                                                       "problem": "QA returned no verdict", "fix": ""}]}
        issues = v.get("issues") or []
        v["ok"] = not any(i.get("severity") == "critical" for i in issues)
        v["lines"] = [f"[{i.get('severity')}/{i.get('kind')}] {i.get('problem')} -> {i.get('fix')}" for i in issues]
        self.trace["stages"].append({"stage": "verify", "out": v})
        return v

    def safety_check(self, final: dict) -> list:
        resp = self._call([llm.dev(prompts.SAFETY_INSTR.format(reply=final.get("reply", "")))], [prompts.SAFETY_TOOL],
                          tool_choice={"type": "function", "name": "submit_safety"}, effort=self.aux_effort)
        calls = llm.function_calls(resp)
        v = _args(calls[0]) if calls else {}
        self.trace["stages"].append({"stage": "safety", "out": v})
        return (v.get("sentences") or []) if v.get("violates") else []

    def compress(self, final: dict) -> dict:
        """One tool-free rewrite that keeps the facts and drops process, data and boilerplate."""
        reply = final.get("reply", "")
        resp = self._call([llm.dev(prompts.COMPRESS_INSTR.format(reply=reply, ask=self.ticket_text()[-1500:]))],
                          None, effort=self.aux_effort)
        text = llm.output_text(resp).strip()
        if not text or len(text) >= len(reply):
            return final
        self.trace["stages"].append({"stage": "compressed", "out": {"before": reply, "after": text}})
        return dict(final, reply=text)

    def run(self) -> dict:
        t0 = time.time()
        tri = self.triage()
        final, convo = self.answer(tri)
        self.trace["stages"].append({"stage": "draft", "out": final})
        verdict = self.verify(final)
        rounds = 0
        while not verdict.get("ok") and rounds < MAX_REVISIONS:
            rounds += 1
            # only critical findings drive a revision; "could be more complete" notes just lengthen replies
            crit = [line for line in verdict["lines"] if line.startswith("[critical")]
            final, convo = self.answer(tri, convo=convo, revise_issues=crit or verdict["lines"])
            self.trace["stages"].append({"stage": "revised", "out": final})
            verdict = self.verify(final)
        # The verifier drives revisions but is noisy, so after the last round its leftover
        # findings are acted on narrowly instead of discarding the answer:
        #  - should_handoff: keep the reply, make sure the ticket also goes to a human
        #  - refund/compensation promise: confirm with a single-purpose check, then strip it
        #  - anything else still critical: the reply is not sent automatically (needs_review)
        left = [i for i in verdict.get("issues") or [] if i.get("severity") == "critical"]
        kinds = {i.get("kind") for i in left}
        if "should_handoff" in kinds and final.get("decision") in ("answer", "clarify"):
            final = dict(final, decision="answer_and_handoff", close_ticket=False,
                         handoff_reason=final.get("handoff_reason") or "; ".join(
                             i.get("problem", "")[:120] for i in left if i.get("kind") == "should_handoff"))
            self.trace["stages"].append({"stage": "escalated"})
        restricted_ask = any(i.get("category") in prompts.RESTRICTED_CATEGORIES for i in tri.get("issues", []))
        if ("refund_or_comp_promise" in kinds or restricted_ask) and REFUND_WORDS.search(final.get("reply", "")):
            bad = self.safety_check(final)
            if bad:
                final, convo = self.answer(tri, convo=convo, revise_issues=[prompts.SAFETY_FIX + " | ".join(bad)])
                self.trace["stages"].append({"stage": "safety_revised", "out": final})
        reply = (final.get("reply") or "").strip()
        self.trace["needs_review"] = bool(kinds - {"should_handoff", "refund_or_comp_promise"})
        if reply.upper() == "WAIT":
            self.trace.update(actions=[], final=dict(final, reply="WAIT"), secs=round(time.time() - t0, 1))
            return self.trace
        limit = self.cfg.max_reply_chars
        if limit and len(reply) > limit:
            final = self.compress(final)
        # A registered action only runs if the final reply cites it.
        cited = set(final.get("actions") or [])
        dropped = [a for k, a in self.ctx.actions.items() if k not in cited]
        if dropped:
            self.trace["stages"].append({"stage": "actions_dropped", "out": dropped})
        self.trace["actions"] = [a for k, a in self.ctx.actions.items() if k in cited]
        self.trace["final"] = final
        self.trace["secs"] = round(time.time() - t0, 1)
        return self.trace
