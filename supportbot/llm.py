"""LLM client for the harness: one call shape, two backends.

`claude-*` models go to the Anthropic Messages API (`supportbot.anthropic`); every
other model goes to an OpenAI-compatible Responses API. System rules are sent as
`developer` input items on both sides (the Anthropic side translates them).
"""

import json
import random
import time

import requests

from . import anthropic
from .config import cfg

NO_RETRY = (400, 401, 403, 404, 413, 422)


class LlmError(RuntimeError):
    pass


def _parse_sse(lines) -> dict:
    completed = failed = None
    for raw in lines:
        if not raw or not raw.startswith(b"data: "):
            continue
        try:
            ev = json.loads(raw[6:])
        except ValueError:
            continue
        t = ev.get("type")
        if t == "response.completed":
            completed = ev["response"]
        elif t in ("response.failed", "response.incomplete", "error"):
            failed = ev
    if completed is None:
        raise LlmError(f"stream ended without completion: {json.dumps(failed, ensure_ascii=False)[:600]}")
    return completed


def respond(input_items: list, tools: list | None = None, tool_choice=None, effort: str = "medium",
            model: str | None = None, retries: int = 5, timeout: int = 600) -> dict:
    """One model turn. Returns a completed Responses-style `response` object."""
    c = cfg()
    model = model or c.model
    if anthropic.is_anthropic_model(model):
        try:
            return anthropic.respond(c, input_items, tools, tool_choice, effort, model, retries, timeout)
        except anthropic.AnthropicError as e:
            raise LlmError(str(e)) from e
    if not c.openai_api_key:
        raise LlmError("no API key (set SB_OPENAI_API_KEY or OPENAI_API_KEY)")
    body = {"model": model, "input": input_items, "stream": True, "store": False,
            "reasoning": {"effort": effort if effort in ("low", "medium", "high") else "high"}}
    if tools:
        body["tools"] = tools
        body["parallel_tool_calls"] = True
    if tool_choice:
        body["tool_choice"] = tool_choice
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(c.openai_base_url + "/responses",
                              headers={"Authorization": f"Bearer {c.openai_api_key}", "Accept": "text/event-stream"},
                              json=body, stream=True, timeout=timeout)
        except requests.RequestException as e:
            last = str(e)
        else:
            if r.status_code == 200:
                try:
                    return _parse_sse(r.iter_lines())
                except (LlmError, requests.RequestException) as e:
                    last = str(e)
            else:
                last = f"HTTP {r.status_code}: {r.text[:400]}"
                if r.status_code in NO_RETRY:
                    raise LlmError(last)
        time.sleep(min(60, 3 * 2 ** attempt) + random.random() * 2)
    raise LlmError(f"exhausted retries: {last}")


def output_text(resp: dict) -> str:
    return "".join(c.get("text", "") for item in resp.get("output", []) if item.get("type") == "message"
                   for c in item.get("content", []) if c.get("type") == "output_text")


def function_calls(resp: dict) -> list:
    return [i for i in resp.get("output", []) if i.get("type") == "function_call"]


def usage(resp: dict) -> dict:
    u = resp.get("usage") or {}
    return {"input": u.get("input_tokens", 0),
            "cached": (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
            "output": u.get("output_tokens", 0)}


def dev(text: str) -> dict:
    return {"role": "developer", "content": [{"type": "input_text", "text": text}]}


def user(text: str, images: list | None = None) -> dict:
    content = [{"type": "input_text", "text": text}]
    for url in images or []:
        content.append({"type": "input_image", "image_url": url, "detail": "high"})
    return {"role": "user", "content": content}


def fn_tool(name: str, description: str, params: dict) -> dict:
    return {"type": "function", "name": name, "description": description, "parameters": params}


def replay_items(resp: dict) -> list:
    """Output items to feed back as input on the next turn (drop server-only fields)."""
    if "_anthropic_blocks" in resp:
        return anthropic.replay_items(resp)
    items = []
    for it in resp.get("output", []):
        if it.get("type") == "function_call":
            items.append({"type": "function_call", "call_id": it["call_id"],
                          "name": it["name"], "arguments": it.get("arguments", "{}")})
        elif it.get("type") == "message":
            text = "".join(c.get("text", "") for c in it.get("content", []) if c.get("type") == "output_text")
            if text:
                items.append({"role": "assistant", "content": [{"type": "output_text", "text": text}]})
    return items
