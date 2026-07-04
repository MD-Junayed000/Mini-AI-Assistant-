"""PII / secret redaction — Item 4.

Patterns covered:
  - sk-/ghp_/xox[bpars]- (cloud / SaaS API tokens)
  - AWS access key IDs (AKIA/ASIA prefix, 16 chars of A-Z2-7)
  - JWTs (header.payload.signature with base64url chunks)
  - Email addresses (RFC 5322 loose)
  - IPv4 addresses
  - Credit-card-shaped numerals (Luhn-checked) — only the Luhn-validated ones,
    to avoid false-flagging normal 16-digit IDs.
"""
from __future__ import annotations

import re
from typing import Any

import structlog

# Email — strict enough to catch almost all real addresses without grabbing
# plain words.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# Token-shaped patterns
_TOKEN_RE = re.compile(r"\b(?:sk-|ghp_|gho_|ghu_|ghs_|ghr_|xoxb-|xoxp-|xoxa-|xoxs-)[A-Za-z0-9_-]{10,}")
_AWS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
# JWT: three base64url groups separated by dots. Boundaries by dot or whitespace.
# Real JWTs use multi-byte segments, but we keep the lower bound loose (>=2 chars
# in the signature) so synthetic fixtures like "eyJabc.def.ghi" still match.
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]{2,}\b")

# Credit-card-shaped — 13-19 digits, with optional spaces/dashes.
_CC_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn_ok(digits: str) -> bool:
    s = 0
    alt = False
    for ch in reversed(digits):
        if not ch.isdigit():
            continue
        n = int(ch)
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        s += n
        alt = not alt
    return s % 10 == 0


def redact_text(s: str) -> str:
    """Return the string with secrets, emails, IPs and Luhn-valid CCs masked."""
    if not isinstance(s, str) or not s:
        return s

    out = s
    out = _EMAIL_RE.sub("[REDACTED_EMAIL]", out)
    out = _AWS_KEY_RE.sub("[REDACTED_AWS_KEY]", out)
    out = _JWT_RE.sub("[REDACTED_JWT]", out)
    out = _TOKEN_RE.sub("[REDACTED_TOKEN]", out)
    out = _IPV4_RE.sub("[REDACTED_IP]", out)

    def cc_sub(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return "[REDACTED_CC]"
        return m.group(0)

    out = _CC_RE.sub(cc_sub, out)
    return out


def redact_value(value: Any) -> Any:
    """Recursively redact strings inside dicts/lists, pass others through."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_value(v) for v in value)
    return value


# ---------------------------------------------------------------------------
# structlog processor — runs on every log line.
# ---------------------------------------------------------------------------

def redact_processor(_logger: Any, _method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """structlog processor that scrubs known secrets from the log payload."""
    return redact_value(event_dict)
