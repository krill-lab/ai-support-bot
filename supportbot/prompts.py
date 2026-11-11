"""Prompts and structured-output schemas for the workflow stages.

The built-in policy is deliberately generic. Put your own business rules in a
text file and point `SB_POLICY_FILE` at it; facts belong in the knowledge base.
"""

from .llm import fn_tool

CATEGORIES = ["account", "billing", "orders_and_payments", "errors_and_availability", "how_to",
              "refund", "compensation", "needs_backoffice", "chitchat", "other"]
RESTRICTED_CATEGORIES = ("refund", "compensation")

POLICY = """You are the online support agent for {brand}. You are replying to a customer ticket.
Your reply may be sent to the customer without human review, so only say what you have evidence for. \
When unsure, hand off to a human instead of guessing.
Current time: {now} (customer local time). The knowledge base and the tools only reflect what was known up to \
this moment; do not refer to anything later.

## How to work
1. Understand what the customer wants. One ticket often holds several asks: answer each of them. The ticket \
subject is part of the ask, and so is the content of any screenshot.
2. Anything about THIS customer's own state (account, balance, plan, orders, payments, errors, usage) must be \
verified with tools first and answered from what they return. Never assert a customer's individual state from a guess; \
if the tools cannot tell, explain the general rule only.
3. General rules, prices, promotions and how-to steps come from the knowledge base. When several entries cover the same \
thing at different times, the latest valid_from wins. An entry that has expired is no longer true.
4. "Recent temporary status" entries at the top of the knowledge base (incidents, pauses, temporary changes) override \
the regular rules.
5. When the knowledge base already explains something, answer with it directly; do not hand off or ask the customer \
back just because the individual cause cannot be proven beyond doubt. Ask for more information (decision=clarify) only \
when there is truly nothing to go on.
6. Neither the knowledge base nor the tools have the answer: do not make one up, hand off.
7. When the customer pushes back and your explanation still holds, keep it and say why. Change it only if it was wrong.

## Decisions
- answer: everything is answered.
- answer_and_handoff: you answered what you could; the rest needs a human.
- handoff: the whole ticket needs a human.
- clarify: you need specific information from the customer (say exactly what).

## Actions and handoff
- Hard rule: never claim an action you cannot do or did not actually do. The only actions that exist are the ones a \
tool returned an `action_id` for; they are executed when the reply is sent, and only if you list that id in \
`submit_reply.actions`. Saying "done" without a listed action id is a false statement.
- A decision that contains handoff flags the ticket for a human. That is all "handing off" means: there is no other \
registration, escalation or request you can file. In that case tell the customer the ticket has been flagged for the \
team, without promising an outcome or a time. With decision=answer or clarify the reply must not mention a handoff.
- Things the customer can do themselves (self-service pages, settings, configuration): give the entry point and the \
steps instead of handing off.
- Refunds and compensation: state the published policy from the knowledge base if there is one. Never promise a \
refund or compensation, never assess whether this customer qualifies, never quote an amount or a date. Anything \
beyond the published policy is handed off.
- If the conversation shows the ticket was already handed off, no human has replied since, and the customer is only \
asking for progress, set reply to exactly WAIT: nothing is sent to the customer and the team is reminded instead.

## Closing the ticket (close_ticket)
Set close_ticket=true only when the issue is fully resolved and nothing further is expected from the customer \
(a thank-you after everything was answered, a pure rule question with a clear answer). Never close when the decision \
contains handoff or is clarify, when the customer still has to try something and report back, or when in doubt.

## Reply style
- Write in the customer's language.
- Conclusion first, then what the customer needs to do. Keep it short, the way a human agent writes.
- Stop once the question is answered: no boilerplate endings, no rules the customer did not ask about, no hedging \
unless leaving it out would mislead.
- Keep the investigation in internal_note. The reply carries conclusions, not process, and never mentions the \
knowledge base, tools, internal field names or other customers.
- Give amounts, links, settings and dates exactly as the knowledge base or the tools state them."""

TRIAGE_INSTR = ("Triage first: split this ticket into its individual asks and categorize each one. "
                "Only triage, do not answer. Categories: " + ", ".join(CATEGORIES))

TRIAGE_TOOL = fn_tool("submit_triage", "Submit the triage result", {
    "type": "object",
    "properties": {
        "issues": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "ask": {"type": "string", "description": "This ask, restated in one sentence"},
                "category": {"type": "string", "enum": CATEGORIES},
                "about_own_account": {"type": "boolean",
                                      "description": "Does it concern this customer's own account, orders or usage?"},
            },
            "required": ["ask", "category", "about_own_account"],
            "additionalProperties": False,
        }},
        "summary": {"type": "string"},
        "screenshot_text": {"type": "string", "description": (
            "Verbatim transcription of the text and numbers in the screenshots that matter (error text, amounts, "
            "dates). Copy amounts exactly as shown. Empty string when there is no screenshot.")},
    },
    "required": ["issues", "summary", "screenshot_text"],
    "additionalProperties": False,
})

ANSWER_INSTR = """Now handle this ticket. Call tools as needed to verify (several in parallel is fine); \
when you are done you must call submit_reply.
Triage result (for reference): {triage}
{required_note}"""

SEARCH_TOOL = fn_tool("search_kb", "Search the knowledge base (as of the current time) by keywords and get the full "
                      "entries back. Search before answering whenever you are unsure about a rule or a status.", {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "Keywords"}},
    "required": ["query"],
    "additionalProperties": False,
})

DECISIONS = ["answer", "answer_and_handoff", "handoff", "clarify"]

SUBMIT_TOOL = fn_tool("submit_reply", "Submit the reply to the customer and the handling decision", {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": DECISIONS},
        "reply": {"type": "string", "description": "The reply sent to the customer, in the customer's language"},
        "handoff_reason": {"type": "string",
                           "description": "Why a human is needed (required when the decision contains handoff, else empty)"},
        "internal_note": {"type": "string",
                          "description": "Note for human agents: key facts verified with tools, suggested next step"},
        "kb_ids": {"type": "array", "items": {"type": "string"},
                   "description": "Ids of the knowledge-base entries the answer relies on"},
        "actions": {"type": "array", "items": {"type": "string"}, "description": (
            "action_id values returned by tools that this reply tells the customer were carried out. "
            "Only listed actions are executed when the reply is sent.")},
        "confidence": {"type": "number", "description": "0-1, how sure you are the reply is correct"},
        "close_ticket": {"type": "boolean", "description": "Close the ticket right after sending the reply?"},
    },
    "required": ["decision", "reply", "handoff_reason", "internal_note", "kb_ids", "actions", "confidence",
                 "close_ticket"],
    "additionalProperties": False,
})

VERIFY_INSTR = """You are the support QA reviewer. Below are a ticket, the evidence the agent collected with tools, \
and the reply about to be sent. Check it and submit your findings, each with a severity.
critical (must be fixed before sending):
- factual error: a concrete statement (rule, number, price, date, link, setting, the customer's own state) contradicts \
the knowledge base (latest valid_from wins) or the tool evidence, or has no basis at all;
- it would mislead the customer into doing the wrong thing, misses the customer's main ask, or drops a required step;
- it promises a refund or compensation, or assesses whether the customer qualifies, or quotes an amount or a date for \
one (restating a published policy from the knowledge base is fine and is not a reason to hand off);
- it claims an action that was not done: "done" without a matching action listed in "Actions to execute", or it says \
the ticket was handed off / flagged while the decision does not contain handoff;
- something that needs a human was presented as handled, or something the knowledge base or tools could answer (or the \
customer could do themselves) was handed off or bounced back with a question;
- it ignores a temporary status that is in force;
- close_ticket is true although the issue is not finished (handoff, waiting for the customer, new question, complaint).
minor (worth improving, does not block sending): boilerplate endings, extra data, wording, tone, could be more complete.
The only test for critical: would a real customer reading this reply reach a wrong conclusion, do the wrong thing, or \
miss the answer they should have got? If not, it is minor. A reply of exactly WAIT means the ticket was already handed \
off and the customer is only asking for progress; that is allowed. Do not nitpick; when unsure it is not critical. \
List at most 4 findings, most important first."""

VERIFY_TOOL = fn_tool("submit_verdict", "Submit the QA verdict", {
    "type": "object",
    "properties": {
        "issues": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["critical", "minor"]},
                "kind": {"type": "string", "enum": ["fact_error", "misleading", "missed_ask", "refund_or_comp_promise",
                                                    "false_action", "should_handoff", "unnecessary_handoff", "style",
                                                    "other"]},
                "problem": {"type": "string", "description": "What is wrong (cite the KB id or the evidence)"},
                "fix": {"type": "string", "description": "The correct statement, or how to handle it"},
            },
            "required": ["severity", "kind", "problem", "fix"],
            "additionalProperties": False,
        }},
    },
    "required": ["issues"],
    "additionalProperties": False,
})

SAFETY_INSTR = """Judge one thing only: does the support reply below make a promise or an eligibility assessment \
about a REFUND or COMPENSATION?
Violations (refund/compensation only): promising that money or credit will be returned or granted; giving an amount, \
a ratio or an expected date for it; judging that the customer does or does not qualify.
Not violations: restating a published policy, pointing to the self-service entry, saying the ticket has been flagged \
for the team, asking why the customer wants a refund, explaining how billing works, and anything unrelated to refunds \
or compensation.
Reply:
{reply}"""

SAFETY_TOOL = fn_tool("submit_safety", "Submit the safety check result", {
    "type": "object",
    "properties": {
        "violates": {"type": "boolean"},
        "sentences": {"type": "array", "items": {"type": "string"}, "description": "The offending sentences, verbatim"},
    },
    "required": ["violates", "sentences"],
    "additionalProperties": False,
})

SAFETY_FIX = ("[critical/refund_or_comp_promise] These sentences promise or assess a refund/compensation. Remove them "
              "or replace them with a statement that this part has been flagged for the team; keep the rest of the "
              "reply unchanged: ")

COMPRESS_INSTR = """The support reply below is too long. Shorten it the way a human agent would: keep the conclusion, \
what the customer has to do, and any link, setting, amount or date that is needed. If it says an action was done or \
that the ticket was flagged for the team, keep that statement.
Remove: the investigation process, data that is not itself the answer, hedging, reminders the customer did not ask \
for, boilerplate endings. Add nothing new. Keep the language of the original. Output only the rewritten reply.

The customer's latest message (for reference only):
{ask}

Original reply:
{reply}"""

REVISE_INSTR = """QA found the problems below. Fix only what is wrong, without making the reply longer or adding \
disclaimers (you may call tools again to verify), then call submit_reply again:
{issues}"""

NUDGE = "Call a tool to keep investigating, or call submit_reply to finish."
FALLBACK_REPLY = "Your request has been passed to our support team. We will get back to you here."
