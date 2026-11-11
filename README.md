# ai-support-bot

A small, replayable customer-support agent: a fixed multi-stage workflow, read-only tools, and a
knowledge base that knows *when* each fact was true.

[中文说明](README.zh-CN.md)

It is about 1,400 lines of Python with one dependency (`requests`). There is no framework: the control
flow is ordinary code, and the model is only asked for structured output at each stage.

## Why it is built this way

- **The model does not decide the process.** Triage, investigation, QA and the safety pass are separate
  calls with their own schema. Code decides what happens between them.
- **Verify before answering.** Triage labels each ask with a category. Tools declare which categories they
  are mandatory for, and `submit_reply` is refused until those tools have been called.
- **Tools never write.** A tool that wants to change something registers a pending action. It runs only
  when the reply is sent, only if the reply cites it, and its executor re-checks live data first.
- **A second opinion on every reply.** An independent QA call checks the draft against the knowledge base
  and the tool evidence. Critical findings trigger up to two revisions.
- **Refunds and compensation are fenced off.** The agent may restate published policy; promises and
  eligibility judgements are detected and stripped, and the ticket goes to a human.
- **Replayable.** Every fact has `available_at` / `valid_to`, and every tool receives the ticket's
  timestamp, so re-running an old ticket sees what support could have known then. That makes offline
  evaluation against past human replies honest.

## Workflow

```
ticket ─► triage ─► retrieve (BM25) ─► tool loop ─► submit_reply ─► QA verify ─► safety ─► send / draft
                                           ▲                            │
                                           └──── revise (max 2) ◄───────┘
```

1. **Triage** splits the ticket into asks and categories (no knowledge base in this call, so it is cheap).
2. **Retrieve** runs BM25 per ticket and per ask; the hits are placed right before the answer step (the full
   knowledge base is in context too, behind a cache breakpoint).
3. **Answer** is a tool loop that must end in `submit_reply` with one of four decisions:
   `answer`, `answer_and_handoff`, `handoff`, `clarify`.
4. **Verify** grades findings as critical or minor. Only critical ones drive a revision.
5. **Safety** acts narrowly on what is left: escalate to a human, strip refund promises, compress an
   over-long reply, drop actions the reply does not cite. Anything else still critical is held as a draft.

## Quick start

```bash
pip install -e .
cp .env.example .env          # set SB_ANTHROPIC_API_KEY (or the SB_OPENAI_* pair)
supportbot init-demo          # demo.sqlite: two customers, two open tickets
supportbot ask examples/ticket.json --trace   # one case, full trace
supportbot serve --once       # answer every open ticket in the demo desk
supportbot drafts             # replies waiting for approval (SB_AUTO_SEND=0)
supportbot approve 1
```

Tests run offline with a scripted fake model:

```bash
python -m unittest discover -s tests
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SB_BRAND` | `Acme` | Product name used in the prompt |
| `SB_MODEL` | `claude-haiku-5-5` | `claude-*` uses the Anthropic Messages API, anything else an OpenAI-compatible Responses API |
| `SB_EFFORT` | `medium` | Reasoning effort of the answer and QA stages (`low` … `max`) |
| `SB_ANTHROPIC_API_KEY`, `SB_ANTHROPIC_BASE_URL` | | Anthropic backend |
| `SB_OPENAI_API_KEY`, `SB_OPENAI_BASE_URL` | | Responses backend |
| `SB_THINKING` | `adaptive` | `off` sends no thinking parameters (for models that reject them) |
| `SB_KB_FILE` | `examples/kb.jsonl` | Knowledge base |
| `SB_POLICY_FILE` | | Extra business rules appended to the built-in policy |
| `SB_TOOLS` | `examples/demo_tools.py` | Comma-separated Python files that register tools |
| `SB_TZ_OFFSET_HOURS` | `0` | Customers' local time relative to UTC |
| `SB_MAX_REPLY_CHARS` | `600` | Longer replies get one compression pass; `0` disables it |
| `SB_AUTO_SEND` | `0` | `1` sends replies without approval |
| `SB_DB` | `demo.sqlite` | The demo ticket desk |

## Knowledge base

One JSON object per line:

```json
{"id": "BILL-002", "topic": "Billing", "kind": "rule", "title": "Pending payments",
 "fact": "An order stays pending until the payment provider confirms it ...",
 "applies_when": "optional scope", "valid_from": "2025-06-01", "valid_to": null,
 "available_at": "2025-06-01T00:00:00"}
```

- `available_at` (UTC) is when support could first have known the fact; later tickets see it, earlier
  replays do not.
- `valid_to` (local date) expires a fact. Entries of kind `incident`, `promo` or `status` from the last
  14 days are rendered first as "recent temporary status" and override the regular rules.

## Writing tools

```python
from supportbot.tools import tool, executor

@tool("get_orders", "The customer's orders up to the time of the ticket.",
      required_for=("orders_and_payments", "refund"))
def get_orders(ctx, args):
    return {"orders": my_db.orders(ctx.user_id, before=ctx.as_of)}

@tool("extend_trial", "Register a 7-day trial extension. Returns an action_id.")
def extend_trial(ctx, args):
    return {"eligible": True,
            "action_id": ctx.propose("extend_trial", {"user_id": ctx.user_id}, "extend the trial by 7 days")}

@executor("extend_trial")
def run_extend_trial(payload):
    my_api.extend_trial(payload["user_id"])   # re-check eligibility here: time has passed since the draft
```

Categories are listed in `supportbot/prompts.py` (`CATEGORIES`). Connecting your own helpdesk means
replacing `supportbot/store.py`: produce a case dict, and deliver the final reply.

## Before you turn on auto-send

Run with `SB_AUTO_SEND=0` first and read the drafts and traces. Replay a few weeks of real tickets with
`--kb-freeze` and compare the output with what your human agents wrote. Keep write actions small, capped
and re-checked in their executor.

## License

MIT
