"""A tiny SQLite ticket desk, so the bot runs end to end without any other system.

Replace this module (and the demo tools) with adapters for your own helpdesk and
databases; the agent itself only needs a case dict (see `supportbot.agent`).
"""

import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, email TEXT, name TEXT, plan TEXT, plan_expires_at TEXT, balance_cents INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY, user_id INTEGER, item TEXT, amount_cents INTEGER, status TEXT,
    created_at TEXT, paid_at TEXT);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY, user_id INTEGER, subject TEXT, status TEXT DEFAULT 'open', created_at TEXT);
CREATE TABLE IF NOT EXISTS ticket_messages (
    id INTEGER PRIMARY KEY, ticket_id INTEGER, role TEXT, text TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS drafts (
    id INTEGER PRIMARY KEY, ticket_id INTEGER, last_msg_id INTEGER, status TEXT, decision TEXT, reply TEXT,
    close_ticket INTEGER, actions TEXT, trace TEXT, created_at TEXT, UNIQUE(ticket_id, last_msg_id));
"""

DEMO = """
INSERT INTO users VALUES (1, 'ada@example.com', 'Ada', 'pro', '2030-01-01T00:00:00', 1250);
INSERT INTO users VALUES (2, 'bob@example.com', 'Bob', 'free', NULL, -40);
INSERT INTO orders VALUES (101, 1, 'Pro plan (monthly)', 2000, 'paid', '2026-01-02T09:00:00', '2026-01-02T09:01:00');
INSERT INTO orders VALUES (102, 2, 'Credit top-up', 1000, 'pending', '2026-01-05T10:00:00', NULL);
INSERT INTO tickets VALUES (1, 1, 'How do I rotate my API key?', 'open', '2026-01-06T08:00:00');
INSERT INTO ticket_messages VALUES (1, 1, 'user', 'I think my key leaked. How can I rotate it?', '2026-01-06T08:00:00');
INSERT INTO tickets VALUES (2, 2, 'Paid but nothing arrived', 'open', '2026-01-06T08:05:00');
INSERT INTO ticket_messages VALUES (2, 2, 'user',
    'I topped up yesterday but my balance is still negative and my requests are rejected.', '2026-01-06T08:05:00');
"""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def init_demo(path: str) -> None:
    conn = connect(path)
    if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        conn.executescript(DEMO)
    conn.commit()
    conn.close()


def rows(conn, sql: str, args=()) -> list:
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def pending_tickets(conn) -> list:
    """Open tickets whose latest message is the customer's and has no draft yet."""
    return rows(conn, """
        SELECT t.*, m.id AS last_msg_id, m.created_at AS last_at FROM tickets t
        JOIN ticket_messages m ON m.id = (SELECT MAX(id) FROM ticket_messages WHERE ticket_id = t.id)
        WHERE t.status = 'open' AND m.role = 'user'
          AND NOT EXISTS (SELECT 1 FROM drafts d WHERE d.ticket_id = t.id AND d.last_msg_id = m.id)
        ORDER BY m.id""")


def build_case(conn, ticket: dict) -> dict:
    msgs = rows(conn, "SELECT * FROM ticket_messages WHERE ticket_id = ? ORDER BY id", (ticket["id"],))
    cut = len(msgs)
    while cut > 0 and msgs[cut - 1]["role"] == "user":
        cut -= 1  # the trailing run of customer messages is what needs an answer
    return {
        "ticket_id": ticket["id"], "user_id": ticket["user_id"], "subject": ticket["subject"],
        "as_of": msgs[-1]["created_at"],
        "history": [{"t": m["created_at"], "role": m["role"], "text": m["text"]} for m in msgs[:cut]],
        "messages": [{"t": m["created_at"], "text": m["text"]} for m in msgs[cut:]],
    }


def save_draft(conn, ticket: dict, trace: dict, status: str, now: str) -> int:
    final = trace.get("final") or {}
    cur = conn.execute(
        "INSERT INTO drafts (ticket_id, last_msg_id, status, decision, reply, close_ticket, actions, trace, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (ticket["id"], ticket["last_msg_id"], status, final.get("decision"), final.get("reply"),
         int(bool(final.get("close_ticket"))), json.dumps(trace.get("actions") or []),
         json.dumps(trace, ensure_ascii=False), now))
    conn.commit()
    return cur.lastrowid


def post_reply(conn, ticket_id: int, text: str, decision: str, close: bool, now: str) -> None:
    conn.execute("INSERT INTO ticket_messages (ticket_id, role, text, created_at) VALUES (?, 'ai', ?, ?)",
                 (ticket_id, text, now))
    if "handoff" in (decision or ""):
        status = "pending_human"
    elif close and decision == "answer":
        status = "closed"
    else:
        status = "open"
    conn.execute("UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id))
    conn.commit()
