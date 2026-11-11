"""Runtime configuration, all from environment variables (optionally a `.env` file).

Two LLM backends live side by side, picked per model name (see `supportbot.llm`):

- `claude-*` goes to the Anthropic Messages API.
- anything else goes to an OpenAI-compatible Responses API.

Secrets are only ever read from the environment, never from code.
"""

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Minimal `.env` loader: KEY=VALUE lines, existing environment wins."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Config:
    brand: str
    model: str
    effort: str
    openai_base_url: str
    openai_api_key: str
    anthropic_base_url: str
    anthropic_api_key: str
    thinking: str
    max_tokens: int
    kb_file: str
    policy_file: str
    tz_offset_hours: float
    max_reply_chars: int
    tools: tuple
    db_path: str
    auto_send: bool


_cfg: Config | None = None


def load() -> Config:
    load_dotenv(os.environ.get("SB_ENV_FILE", ".env"))
    env = os.environ.get
    return Config(
        brand=env("SB_BRAND", "Acme"),
        model=env("SB_MODEL", "claude-haiku-5-5"),
        # reasoning effort of the answering stage: low, medium, high, xhigh, max
        effort=env("SB_EFFORT", "medium"),
        openai_base_url=env("SB_OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        openai_api_key=env("SB_OPENAI_API_KEY") or env("OPENAI_API_KEY") or "",
        anthropic_base_url=env("SB_ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/"),
        anthropic_api_key=env("SB_ANTHROPIC_API_KEY") or env("ANTHROPIC_API_KEY") or "",
        # adaptive = let the model decide how much to think, steered by `effort`; off = no thinking params
        thinking=env("SB_THINKING", "adaptive"),
        max_tokens=int(env("SB_MAX_TOKENS", "16000")),
        kb_file=env("SB_KB_FILE", "examples/kb.jsonl"),
        # extra business rules appended to the built-in policy prompt
        policy_file=env("SB_POLICY_FILE", ""),
        # customers' local time = UTC + this offset (used for dates shown to the model and KB expiry)
        tz_offset_hours=float(env("SB_TZ_OFFSET_HOURS", "0")),
        # replies longer than this get one compression pass; 0 disables it
        max_reply_chars=int(env("SB_MAX_REPLY_CHARS", "600")),
        # python files that register tools (comma separated)
        tools=tuple(p.strip() for p in env("SB_TOOLS", "examples/demo_tools.py").split(",") if p.strip()),
        db_path=env("SB_DB", "demo.sqlite"),
        # 1 = replies go straight to the customer; 0 = every reply waits as a draft for `approve`
        auto_send=env("SB_AUTO_SEND", "0") == "1",
    )


def cfg() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load()
    return _cfg


def reset() -> None:
    """Drop the cached config (tests, or after changing the environment)."""
    global _cfg
    _cfg = None
