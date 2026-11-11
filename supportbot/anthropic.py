"""Anthropic Messages API backend, behind the call shape `supportbot.llm` exposes.

The harness speaks the OpenAI Responses vocabulary everywhere (`developer`/`user`
items, `function_call` / `function_call_output`, a completed `response` object),
so everything here is translation: items in, a Messages request out, and a
Responses-shaped dict back.

Replay keeps the assistant turn's raw blocks (`_raw`), because a tool result may
only follow an assistant turn that still carries its signed thinking blocks.
"""

import json
import logging
import random
import re
import time

import requests

API_VERSION = "2023-06-01"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
# blocks are echoed back verbatim on the next turn, minus the fields the API only ever sends out
KEEP = {"text": ("type", "text"), "thinking": ("type", "thinking", "signature"),
        "redacted_thinking": ("type", "data"), "tool_use": ("type", "id", "name", "input")}
NO_RETRY = (400, 401, 403, 404, 413, 422)

log = logging.getLogger("supportbot.anthropic")


class AnthropicError(RuntimeError):
    pass


def is_anthropic_model(model: str) -> bool:
    return (model or "").startswith("claude")


# ---------------------------------------------------------------- items -> request

def _image_block(url: str) -> dict:
    m = re.match(r"data:(image/[-\w.+]+);base64,(.*)$", url or "", re.S)
    if m:
        return {"type": "image", "source": {"type": "base64", "media_type": m.group(1), "data": m.group(2).strip()}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _content_blocks(item: dict) -> list:
    out = []
    for c in item.get("content") or []:
        t = c.get("type")
        if t in ("input_text", "output_text", "text"):
            if c.get("text"):
                out.append({"type": "text", "text": c["text"]})
        elif t == "input_image":
            out.append(_image_block(c.get("image_url") or ""))
    return out


def _assistant_blocks(blocks: list) -> list:
    out = []
    for b in blocks or []:
        keep = KEEP.get(b.get("type"))
        if keep:
            out.append({k: b[k] for k in keep if k in b})
    return out


def _tool_choice(tool_choice) -> dict | None:
    if not tool_choice or tool_choice == "auto":
        return None
    if tool_choice == "required":
        return {"type": "any"}
    if tool_choice == "none":
        return {"type": "none"}
    if isinstance(tool_choice, dict) and tool_choice.get("name"):
        return {"type": "tool", "name": tool_choice["name"]}
    return None


def build_request(input_items: list, tools: list | None, tool_choice, effort: str, model: str,
                  max_tokens: int = 16000, thinking: str = "adaptive") -> dict:
    """Translate Responses input items into a Messages request body.

    The leading run of `developer` items becomes the system prompt (the policy and
    the KB, which repeat across every call and so carry the cache breakpoint); any
    later `developer` item keeps its position as a text block in the conversation,
    because the harness uses position for recency ("fix these issues", "keep going").
    """
    system, msgs = [], []
    cur = None
    turns = any(it.get("role") in ("user", "assistant") or
                it.get("type") in ("function_call", "function_call_output") for it in input_items)

    def push(role: str, blocks: list):
        nonlocal cur
        if not blocks:
            return
        if cur is None or cur["role"] != role:
            cur = {"role": role, "content": []}
            msgs.append(cur)
        cur["content"] += blocks

    for it in input_items:
        role, typ = it.get("role"), it.get("type")
        if role == "developer":
            text = "".join(c.get("text", "") for c in it.get("content") or [])
            if not msgs and turns:
                system.append({"type": "text", "text": text})
            else:
                push("user", [{"type": "text", "text": text}] if text else [])
        elif role == "assistant":
            push("assistant", it["_raw"] if "_raw" in it else _content_blocks(it))
        elif role == "user":
            push("user", _content_blocks(it))
        elif typ == "function_call":
            try:
                args = json.loads(it.get("arguments") or "{}")
            except ValueError:
                args = {}
            push("assistant", [{"type": "tool_use", "id": it["call_id"], "name": it["name"], "input": args}])
        elif typ == "function_call_output":
            push("user", [{"type": "tool_result", "tool_use_id": it["call_id"], "content": str(it.get("output") or "")}])

    if not msgs:  # a side step that is pure instructions (e.g. the safety check): send it as the turn
        msgs = [{"role": "user", "content": system or [{"type": "text", "text": "Continue."}]}]
        system = []
    if system:
        system[-1]["cache_control"] = {"type": "ephemeral"}

    body = {"model": model, "max_tokens": max_tokens, "messages": msgs, "stream": True}
    if thinking == "adaptive":
        body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {"effort": effort if effort in EFFORTS else "high"}
    if system:
        body["system"] = system
    if tools:
        body["tools"] = [{"name": t["name"], "description": t.get("description") or "",
                          "input_schema": t.get("parameters") or {"type": "object", "properties": {}}} for t in tools]
        tc = _tool_choice(tool_choice)
        if tc:
            body["tool_choice"] = tc
    return body


# ---------------------------------------------------------------- response -> Responses shape

def parse_sse(lines) -> tuple:
    blocks, usage, stop = [], {}, None
    partial = {}
    for raw in lines:
        if not raw or not raw.startswith(b"data: "):
            continue
        try:
            ev = json.loads(raw[6:])
        except ValueError:
            continue
        t = ev.get("type")
        if t == "message_start":
            usage = dict((ev.get("message") or {}).get("usage") or {})
        elif t == "content_block_start":
            i = ev["index"]
            while len(blocks) <= i:
                blocks.append({})
            blocks[i] = dict(ev.get("content_block") or {})
            partial[i] = ""
        elif t == "content_block_delta":
            i, d = ev["index"], ev.get("delta") or {}
            if i >= len(blocks):
                continue
            kind = d.get("type")
            if kind == "text_delta":
                blocks[i]["text"] = blocks[i].get("text", "") + d.get("text", "")
            elif kind == "thinking_delta":
                blocks[i]["thinking"] = blocks[i].get("thinking", "") + d.get("thinking", "")
            elif kind == "signature_delta":
                blocks[i]["signature"] = blocks[i].get("signature", "") + d.get("signature", "")
            elif kind == "input_json_delta":
                partial[i] = partial.get(i, "") + d.get("partial_json", "")
        elif t == "content_block_stop":
            i = ev["index"]
            if i < len(blocks) and blocks[i].get("type") == "tool_use" and partial.get(i):
                try:
                    blocks[i]["input"] = json.loads(partial[i])
                except ValueError:
                    blocks[i]["input"] = {}
        elif t == "message_delta":
            stop = (ev.get("delta") or {}).get("stop_reason") or stop
            usage.update(ev.get("usage") or {})
        elif t == "error":
            raise AnthropicError(f"stream error: {json.dumps(ev.get('error'), ensure_ascii=False)[:400]}")
    if not blocks:
        raise AnthropicError(f"stream ended with no content (stop_reason={stop})")
    return blocks, usage, stop


def as_response(blocks: list, usage: dict, stop: str | None) -> dict:
    """A completed-`response`-shaped dict, plus the raw blocks for replay."""
    output = []
    for b in blocks:
        if b.get("type") == "text" and b.get("text"):
            output.append({"type": "message", "content": [{"type": "output_text", "text": b["text"]}]})
        elif b.get("type") == "tool_use":
            output.append({"type": "function_call", "call_id": b.get("id", ""), "name": b.get("name", ""),
                           "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)})
    if stop in ("max_tokens", "refusal"):
        log.warning("turn ended on stop_reason=%s (%s output tokens)", stop, usage.get("output_tokens"))
    cached = int(usage.get("cache_read_input_tokens") or 0)
    return {
        "output": output,
        "status": "completed",
        "stop_reason": stop,
        # same convention as the Responses side: input includes what was read from cache
        "usage": {"input_tokens": int(usage.get("input_tokens") or 0) + cached
                                  + int(usage.get("cache_creation_input_tokens") or 0),
                  "input_tokens_details": {"cached_tokens": cached},
                  "output_tokens": int(usage.get("output_tokens") or 0)},
        "_anthropic_blocks": _assistant_blocks(blocks),
    }


def replay_items(resp: dict) -> list:
    """The assistant turn to feed back, keeping the signed thinking blocks intact."""
    blocks = resp.get("_anthropic_blocks") or []
    return [{"role": "assistant", "_raw": blocks}] if blocks else []


# ---------------------------------------------------------------- request

def respond(cfg, input_items: list, tools, tool_choice, effort: str, model: str, retries: int, timeout: int) -> dict:
    if not cfg.anthropic_api_key:
        raise AnthropicError("no Anthropic API key (set SB_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY)")
    body = build_request(input_items, tools, tool_choice, effort, model, cfg.max_tokens, cfg.thinking)
    headers = {"x-api-key": cfg.anthropic_api_key, "anthropic-version": API_VERSION,
               "content-type": "application/json", "accept": "text/event-stream"}
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(cfg.anthropic_base_url + "/messages", headers=headers, json=body,
                              stream=True, timeout=timeout)
        except requests.RequestException as e:
            last = str(e)
        else:
            if r.status_code == 200:
                try:
                    return as_response(*parse_sse(r.iter_lines()))
                except (AnthropicError, requests.RequestException) as e:
                    last = str(e)
            else:
                last = f"HTTP {r.status_code}: {r.text[:400]}"
                if r.status_code in NO_RETRY:
                    raise AnthropicError(last)
        time.sleep(min(60, 3 * 2 ** attempt) + random.random() * 2)
    raise AnthropicError(f"exhausted retries: {last}")
