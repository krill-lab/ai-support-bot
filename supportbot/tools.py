"""Tool registry.

A tool is a plain function `fn(ctx, args) -> dict` registered with `@tool`.
Tools are read-only by design: a tool that wants to *change* something calls
`ctx.propose(...)`, which only registers a pending action. The action runs
later, at send time, through its `@executor` — and only if the final reply
lists it — so a draft that is rewritten, rejected or never sent changes nothing.

Every tool receives `ctx.as_of` (the ticket's timestamp, UTC). Tools should cut
their data at that moment, so replaying an old ticket sees what support could
have seen back then.
"""

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path

from .llm import fn_tool

NO_PARAMS = {"type": "object", "properties": {}, "additionalProperties": False}


@dataclass
class ToolCtx:
    user_id: int | str | None
    as_of: str
    ticket_id: int | str | None = None
    actions: dict = field(default_factory=dict)

    def propose(self, kind: str, payload: dict, summary: str) -> str:
        """Register a pending write. Returns the action id the model must cite in `submit_reply.actions`."""
        action_id = f"{kind}-{len(self.actions) + 1}"
        self.actions[action_id] = {"id": action_id, "type": kind, "payload": payload, "summary": summary}
        return action_id


@dataclass
class Tool:
    name: str
    description: str
    params: dict
    fn: object
    required_for: tuple = ()


TOOLS: dict = {}
EXECUTORS: dict = {}


def tool(name: str, description: str, params: dict | None = None, required_for: tuple = ()):
    """Register a tool. `required_for` lists triage categories for which this tool must be
    called before a reply is accepted (when the ask is about the customer's own account)."""
    def deco(fn):
        TOOLS[name] = Tool(name, description, params or NO_PARAMS, fn, tuple(required_for))
        return fn
    return deco


def executor(kind: str):
    """Register the function that actually performs a proposed action: `fn(payload) -> dict`."""
    def deco(fn):
        EXECUTORS[kind] = fn
        return fn
    return deco


def load_plugins(paths) -> None:
    """Import the python files that register tools (idempotent per path)."""
    for p in paths:
        path = Path(p).resolve()
        name = f"supportbot_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load tools from {p}")
        spec.loader.exec_module(importlib.util.module_from_spec(spec))


def specs() -> list:
    return [fn_tool(t.name, t.description, t.params) for t in TOOLS.values()]


def required_for(category: str) -> set:
    return {t.name for t in TOOLS.values() if category in t.required_for}


def run(ctx: ToolCtx, name: str, args_json: str | None) -> str:
    t = TOOLS.get(name)
    if t is None:
        return json.dumps({"error": f"unknown tool {name}"})
    try:
        args = json.loads(args_json or "{}")
    except ValueError:
        args = {}
    try:
        out = t.fn(ctx, args)
    except Exception as e:  # a broken tool must not take the whole run down; the model sees the error
        out = {"error": f"{type(e).__name__}: {e}"[:400]}
    return json.dumps(out, ensure_ascii=False, default=str)


def execute(actions: list) -> list:
    """Run confirmed actions at send time. Raises if any of them fails or has no executor."""
    results = []
    for a in actions:
        fn = EXECUTORS.get(a["type"])
        if fn is None:
            raise RuntimeError(f"no executor registered for action type {a['type']}")
        results.append({"id": a["id"], "result": fn(a["payload"])})
    return results
