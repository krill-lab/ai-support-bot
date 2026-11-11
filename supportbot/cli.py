"""Command line entry point.

  supportbot init-demo                 create demo.sqlite with two sample tickets
  supportbot ask examples/ticket.json  run one case file and print the full trace
  supportbot serve [--once]            poll the ticket desk and answer / draft replies
  supportbot drafts                    list drafts waiting for approval
  supportbot approve <draft_id>        send a draft (runs its registered actions first)
"""

import argparse
import json
import logging
import time

from . import store, tools
from .agent import Run, utcnow
from .config import cfg

log = logging.getLogger("supportbot")


def deliver(conn, draft_id: int) -> str:
    """Send one draft: run its actions, post the reply, move the ticket. Returns the new draft status."""
    d = store.rows(conn, "SELECT * FROM drafts WHERE id = ?", (draft_id,))[0]
    if d["status"] not in ("draft", "sending"):
        return d["status"]
    try:
        # actions first: a reply that says "done" must never go out when the action failed
        tools.execute(json.loads(d["actions"] or "[]"))
        store.post_reply(conn, d["ticket_id"], d["reply"], d["decision"], bool(d["close_ticket"]), utcnow())
        status = "sent"
    except Exception as e:
        log.error("draft %s not sent: %s", draft_id, e)
        status = "failed"
    conn.execute("UPDATE drafts SET status = ? WHERE id = ?", (status, draft_id))
    conn.commit()
    return status


def process(conn, ticket: dict) -> None:
    case = store.build_case(conn, ticket)
    trace = Run(case).run()
    final = trace.get("final") or {}
    now = utcnow()
    if final.get("reply") == "WAIT":
        # already handed off and the customer is only nudging: nothing is sent, the team is reminded
        store.save_draft(conn, ticket, trace, "waiting", now)
        conn.execute("UPDATE tickets SET status = 'pending_human' WHERE id = ?", (ticket["id"],))
        conn.commit()
        log.info("ticket %s: waiting for a human", ticket["id"])
        return
    draft_id = store.save_draft(conn, ticket, trace, "draft", now)
    if cfg().auto_send and not trace.get("needs_review"):
        log.info("ticket %s: %s -> %s", ticket["id"], final.get("decision"), deliver(conn, draft_id))
    else:
        log.info("ticket %s: draft %s waits for approval (%s)", ticket["id"], draft_id, final.get("decision"))


def cmd_serve(a) -> None:
    conn = store.connect(cfg().db_path)
    while True:
        for t in store.pending_tickets(conn):
            try:
                process(conn, t)
            except Exception:
                log.exception("ticket %s failed", t["id"])
        if a.once:
            return
        time.sleep(a.interval)


def cmd_ask(a) -> None:
    with open(a.case, encoding="utf-8") as f:
        case = json.load(f)
    trace = Run(case, effort=a.effort, kb_freeze_utc=a.kb_freeze).run()
    print(json.dumps(trace if a.trace else trace.get("final"), ensure_ascii=False, indent=2))


def cmd_drafts(_a) -> None:
    conn = store.connect(cfg().db_path)
    for d in store.rows(conn, "SELECT id, ticket_id, decision, reply, actions FROM drafts WHERE status = 'draft'"):
        print(f"#{d['id']} ticket {d['ticket_id']} [{d['decision']}] actions={d['actions']}\n  {d['reply']}\n")


def cmd_approve(a) -> None:
    print(deliver(store.connect(cfg().db_path), a.draft_id))


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="supportbot", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-demo").set_defaults(fn=lambda _a: store.init_demo(cfg().db_path))
    s = sub.add_parser("ask")
    s.add_argument("case")
    s.add_argument("--effort")
    s.add_argument("--kb-freeze", help="cap the knowledge base at this UTC time (replay)")
    s.add_argument("--trace", action="store_true", help="print the whole trace, not just the final reply")
    s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("serve")
    s.add_argument("--once", action="store_true")
    s.add_argument("--interval", type=int, default=5)
    s.set_defaults(fn=cmd_serve)
    sub.add_parser("drafts").set_defaults(fn=cmd_drafts)
    s = sub.add_parser("approve")
    s.add_argument("draft_id", type=int)
    s.set_defaults(fn=cmd_approve)
    a = p.parse_args(argv)
    tools.load_plugins(cfg().tools)
    a.fn(a)


if __name__ == "__main__":
    main()
