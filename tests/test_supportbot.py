"""Offline tests: the LLM is replaced by a scripted fake, so no network and no API key are needed."""

import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = tempfile.mkdtemp()
os.environ.update(SB_ENV_FILE=os.path.join(TMP, "none.env"), SB_DB=os.path.join(TMP, "test.sqlite"),
                  SB_KB_FILE=str(ROOT / "examples" / "kb.jsonl"), SB_TOOLS=str(ROOT / "examples" / "demo_tools.py"),
                  SB_AUTO_SEND="1", SB_POLICY_FILE="")

from supportbot import anthropic, cli, config, kb, llm, store, tools  # noqa: E402
from supportbot.agent import Run  # noqa: E402

config.reset()
tools.load_plugins(config.cfg().tools)


def call(name, args, call_id="c1"):
    return {"output": [{"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(args)}],
            "usage": {"input_tokens": 10, "output_tokens": 5}}


class FakeLlm:
    """Answers each stage by looking at which structured-output tool is on offer."""

    def __init__(self, reply, verdicts=None, pre_calls=()):
        self.reply, self.verdicts, self.pre_calls = reply, list(verdicts or []), list(pre_calls)
        self.submits = 0

    def __call__(self, items, tools=None, tool_choice=None, effort=None, **_kw):
        names = {t["name"] for t in tools or []}
        if "submit_triage" in names:
            return call("submit_triage", {"issues": [{"ask": "balance is negative", "category": "billing",
                                                      "about_own_account": True}],
                                          "summary": "s", "screenshot_text": ""})
        if "submit_verdict" in names:
            return call("submit_verdict", {"issues": self.verdicts.pop(0) if self.verdicts else []})
        if "submit_reply" in names:
            if "Not verified yet" in str(items[-1].get("output", "")):
                return call("get_account", {}, call_id="late")  # do what the refusal asked for
            if self.pre_calls:
                return call(self.pre_calls.pop(0), {}, call_id=f"t{len(self.pre_calls)}")
            self.submits += 1
            reply = self.reply(items) if callable(self.reply) else self.reply
            return call("submit_reply", reply, call_id=f"s{self.submits}")
        raise AssertionError(f"unexpected call with tools {names}")


def reply(**kw):
    base = {"decision": "answer", "reply": "Done.", "handoff_reason": "", "internal_note": "", "kb_ids": [],
            "actions": [], "confidence": 0.9, "close_ticket": False}
    return dict(base, **kw)


def case():
    return json.loads((ROOT / "examples" / "ticket.json").read_text())


class KbTest(unittest.TestCase):
    def setUp(self):
        self.facts = kb.load(str(ROOT / "examples" / "kb.jsonl"))

    def test_time_filter(self):
        ids = lambda t: {f["id"] for f in kb.facts_at(self.facts, t)}  # noqa: E731
        self.assertNotIn("STAT-001", ids("2026-01-05T08:00:00"))  # not known yet
        self.assertIn("STAT-001", ids("2026-01-05T12:00:00"))
        self.assertNotIn("STAT-001", ids("2026-01-06T12:00:00"))  # expired
        self.assertNotIn("STAT-001", {f["id"] for f in kb.facts_at(self.facts, "2026-01-05T12:00:00",
                                                                  freeze_at_utc="2026-01-01T00:00:00")})

    def test_search_and_render(self):
        self.assertEqual(kb.search(self.facts, "how to rotate my api key")[0]["id"], "ACC-001")
        self.assertIn("钥匙", "".join(kb.tokens("轮换钥匙")))
        text = kb.render(kb.facts_at(self.facts, "2026-01-05T12:00:00"), "2026-01-05T12:00:00", brand="Acme")
        self.assertLess(text.index("Recent temporary status"), text.index("## Billing"))


class AnthropicTranslateTest(unittest.TestCase):
    def test_build_request(self):
        items = [llm.dev("policy"), llm.dev("kb"), llm.user("hi"),
                 {"type": "function_call", "call_id": "x", "name": "get_account", "arguments": "{}"},
                 {"type": "function_call_output", "call_id": "x", "output": "{}"}, llm.dev("fix it")]
        body = anthropic.build_request(items, [llm.fn_tool("get_account", "d", tools.NO_PARAMS)], "required",
                                       "high", "claude-x")
        self.assertEqual([b["text"] for b in body["system"]], ["policy", "kb"])
        self.assertIn("cache_control", body["system"][-1])
        self.assertEqual([m["role"] for m in body["messages"]], ["user", "assistant", "user"])
        self.assertEqual([b["type"] for b in body["messages"][2]["content"]], ["tool_result", "text"])
        self.assertEqual(body["tool_choice"], {"type": "any"})

    def test_parse_stream(self):
        events = [{"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
                  {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "a", "name": "f"}},
                  {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"q": 1}'}},
                  {"type": "content_block_stop", "index": 0},
                  {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}}]
        resp = anthropic.as_response(*anthropic.parse_sse([b"data: " + json.dumps(e).encode() for e in events]))
        self.assertEqual(llm.function_calls(resp)[0]["arguments"], '{"q": 1}')
        self.assertEqual(llm.usage(resp), {"input": 3, "cached": 0, "output": 7})
        self.assertEqual(llm.replay_items(resp)[0]["_raw"][0]["input"], {"q": 1})


class AgentTest(unittest.TestCase):
    def setUp(self):
        if os.path.exists(os.environ["SB_DB"]):
            os.remove(os.environ["SB_DB"])
        store.init_demo(os.environ["SB_DB"])
        self.real = llm.respond

    def tearDown(self):
        llm.respond = self.real

    def test_required_tool_is_enforced(self):
        fake = llm.respond = FakeLlm(reply())
        trace = Run(case()).run()
        self.assertEqual(fake.submits, 2)  # first submit refused: get_account had not been called
        self.assertEqual(trace["final"]["reply"], "Done.")
        self.assertFalse(trace["needs_review"])

    def test_only_cited_actions_survive(self):
        llm.respond = FakeLlm(reply(), pre_calls=["get_account", "waive_small_debt"])
        trace = Run(case()).run()
        self.assertEqual(trace["actions"], [])
        self.assertEqual(trace["stages"][-1]["stage"], "actions_dropped")
        llm.respond = FakeLlm(reply(actions=["waive_debt-1"]), pre_calls=["get_account", "waive_small_debt"])
        self.assertEqual([a["type"] for a in Run(case()).run()["actions"]], ["waive_debt"])

    def test_critical_finding_revises_then_escalates(self):
        crit = [{"severity": "critical", "kind": "should_handoff", "problem": "needs a human", "fix": "hand off"}]
        fake = llm.respond = FakeLlm(reply(close_ticket=True), verdicts=[crit, crit, crit],
                                     pre_calls=["get_account"])
        trace = Run(case()).run()
        self.assertEqual(fake.submits, 3)  # draft + 2 revisions
        self.assertEqual(trace["final"]["decision"], "answer_and_handoff")
        self.assertFalse(trace["final"]["close_ticket"])

    def test_serve_executes_action_and_closes(self):
        llm.respond = FakeLlm(reply(reply="Your balance has been reset.", actions=["waive_debt-1"], close_ticket=True),
                              pre_calls=["get_account", "waive_small_debt"])
        conn = store.connect(os.environ["SB_DB"])
        ticket = [t for t in store.pending_tickets(conn) if t["id"] == 2][0]
        cli.process(conn, ticket)
        self.assertEqual(store.rows(conn, "SELECT balance_cents FROM users WHERE id = 2")[0]["balance_cents"], 0)
        self.assertEqual(store.rows(conn, "SELECT status FROM tickets WHERE id = 2")[0]["status"], "closed")
        self.assertEqual(store.rows(conn, "SELECT role FROM ticket_messages WHERE ticket_id = 2")[-1]["role"], "ai")
        self.assertNotIn(2, [t["id"] for t in store.pending_tickets(conn)])

    def test_failed_action_blocks_the_reply(self):
        llm.respond = FakeLlm(reply(actions=["waive_debt-1"]), pre_calls=["get_account", "waive_small_debt"])
        conn = store.connect(os.environ["SB_DB"])
        ticket = [t for t in store.pending_tickets(conn) if t["id"] == 2][0]
        trace = Run(store.build_case(conn, ticket)).run()
        draft_id = store.save_draft(conn, ticket, trace, "draft", "2026-01-06T09:00:00")
        conn.execute("UPDATE users SET balance_cents = -5000 WHERE id = 2")  # debt grew past the limit meanwhile
        conn.commit()
        self.assertEqual(cli.deliver(conn, draft_id), "failed")
        self.assertEqual(len(store.rows(conn, "SELECT 1 FROM ticket_messages WHERE ticket_id = 2")), 1)


if __name__ == "__main__":
    unittest.main()
