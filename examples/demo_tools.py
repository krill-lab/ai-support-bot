"""Demo tools over the bundled SQLite desk. Copy this file and point the queries at your own systems.

Rules of thumb:
- tools only read; anything that changes data goes through `ctx.propose` + an `@executor`
- cut the data at `ctx.as_of`, so replaying an old ticket sees what support saw back then
- return small, self-describing dicts: the model and the QA stage both read them verbatim
"""

from supportbot import store
from supportbot.config import cfg
from supportbot.tools import executor, tool

WAIVE_LIMIT_CENTS = 100


def _db():
    return store.connect(cfg().db_path)


@tool("get_account", "The customer's account: plan, plan expiry and current balance.",
      required_for=("account", "billing", "errors_and_availability"))
def get_account(ctx, _args):
    rows = store.rows(_db(), "SELECT id, name, plan, plan_expires_at, balance_cents FROM users WHERE id = ?",
                      (ctx.user_id,))
    if not rows:
        return {"error": "no such account"}
    u = rows[0]
    return {"plan": u["plan"], "plan_expires_at": u["plan_expires_at"], "balance_usd": u["balance_cents"] / 100,
            "balance_is_negative": u["balance_cents"] < 0}


@tool("get_orders", "The customer's orders up to the time of the ticket, newest first, with payment status.",
      required_for=("orders_and_payments", "refund"))
def get_orders(ctx, _args):
    orders = store.rows(_db(), "SELECT id, item, amount_cents, status, created_at, paid_at FROM orders "
                               "WHERE user_id = ? AND created_at <= ? ORDER BY id DESC LIMIT 20",
                        (ctx.user_id, ctx.as_of))
    for o in orders:
        o["amount_usd"] = o.pop("amount_cents") / 100
    return {"orders": orders, "note": "status=pending means the payment has not been received"}


@tool("waive_small_debt", "Register a waiver that resets a small negative balance to zero. Returns eligible=true "
      "with an action_id when the debt is within the limit; the waiver only happens if the reply lists that action_id.")
def waive_small_debt(ctx, _args):
    rows = store.rows(_db(), "SELECT balance_cents FROM users WHERE id = ?", (ctx.user_id,))
    cents = rows[0]["balance_cents"] if rows else 0
    if cents >= 0:
        return {"eligible": False, "reason": "the balance is not negative"}
    if -cents > WAIVE_LIMIT_CENTS:
        return {"eligible": False, "reason": f"debt is above the ${WAIVE_LIMIT_CENTS / 100:.2f} auto-waive limit"}
    action_id = ctx.propose("waive_debt", {"user_id": ctx.user_id, "max_cents": WAIVE_LIMIT_CENTS},
                            f"reset the balance of user {ctx.user_id} from ${cents / 100:.2f} to $0")
    return {"eligible": True, "action_id": action_id, "debt_usd": -cents / 100}


@executor("waive_debt")
def run_waive_debt(payload):
    conn = _db()
    # re-check against live data: the balance may have changed since the draft was written
    cur = conn.execute("UPDATE users SET balance_cents = 0 WHERE id = ? AND balance_cents < 0 AND balance_cents >= ?",
                       (payload["user_id"], -payload["max_cents"]))
    conn.commit()
    if cur.rowcount != 1:
        raise RuntimeError("balance changed, waiver refused")
    return {"ok": True}
