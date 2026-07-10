---
name: local-ai-privacy
description: >-
  Keeps sensitive data private when using Claude Code with a cloud LLM.
  Use when the user wants to protect PII, credentials, API keys, patient data,
  customer records, or proprietary code from reaching the cloud; mentions
  HIPAA, GDPR, or data confidentiality; asks to "sanitize before sending to
  cloud", "mask sensitive data", "keep my data local", or "protect my privacy
  with Claude Code". Starts a local redacting proxy that intercepts every
  Claude Code request, redacts sensitive content via a local Lemonade model,
  and forwards only the clean version to the cloud — transparently, with no
  change to how the user types prompts.
  Do not use for image generation, TTS, or STT — use local-ai-use instead.
allowed-tools: Bash(curl:*), Bash(lemonade:*), Bash(python:*)
---

# Local AI Privacy (redacting proxy)

Intercepts every Claude Code request at the network level, redacts sensitive
content locally via a small Lemonade model, and forwards only the sanitized
version to the cloud. Completely transparent — the user types normally, the
cloud only ever sees masked values.

```
You type:  "My SSN is 123-45-6789"
               │
               ▼
     proxy (localhost:8080)          ← intercepts before cloud
               │
               ├─ Lemonade local ────  redacts "123-45-6789" → [SSN_0]
               │   Qwen3-1.7B-GGUF       (stays on your machine)
               │
               ▼
     Cloud model sees only:          ← "My SSN is [SSN_0]"
     AMD gateway / Anthropic
               │
               ▼
     Response streams back through proxy
     [SSN_0] → 123-45-6789 re-substituted before Claude Code renders it
```

## Step 1: check prerequisites

**Lemonade Server running:**
```bash
curl -s http://localhost:13305/api/v1/health
```
If not running, start it:
```bash
lemonade serve
```

**Redaction model present:**
```bash
curl -s "http://localhost:13305/api/v1/models?show_all=true" | python -m json.tool | grep Qwen3-1.7B-GGUF
```
If missing (one-time download, ~1.1 GB):
```bash
lemonade pull Qwen3-1.7B-GGUF
```

## Step 2: run the setup script (once)

Find the skill's scripts directory and run `start.py`:

```bash
python "$(find ~/.claude .claude -name start.py -path '*/local-ai-privacy/*' 2>/dev/null | head -1)"
```

This does three things automatically:
1. Saves your current cloud endpoint so the proxy knows where to forward
2. Starts the proxy as a background process (survives terminal close)
3. Patches `~/.claude/settings.json` so Claude Code routes through the proxy

Expected output:
```
[local-ai-privacy] Lemonade Server reachable at http://localhost:13305
[local-ai-privacy] Redaction model Qwen3-1.7B-GGUF available
[local-ai-privacy] Saved proxy config to ~/.claude/skills/local-ai-privacy/proxy.conf
[local-ai-privacy] Proxy started (PID 12345), log: ...proxy.log
[local-ai-privacy] Proxy ready at http://localhost:8080
[local-ai-privacy] Patched settings.json: ANTHROPIC_BASE_URL → http://localhost:8080

[local-ai-privacy] Setup complete. Now:
[local-ai-privacy]   1. Restart Claude Code
[local-ai-privacy]   2. Every prompt you type will be redacted locally before reaching the cloud
[local-ai-privacy]   3. To stop: python stop.py
```

## Step 3: restart Claude Code

Close this session and open a new one. From this point on, **every prompt
is automatically redacted** — no skill invocation needed, no changes to how
you type. The proxy runs in the background and intercepts all requests.

To verify it is working, check the proxy log:
```bash
cat ~/.claude/skills/local-ai-privacy/proxy.log
```
You will see lines like:
```
[proxy] Redacted entities: ['SSN', 'EMAIL']
[proxy] Forwarding redacted request to cloud
```

## To stop (restore direct cloud connection)

```bash
python "$(find ~/.claude .claude -name stop.py -path '*/local-ai-privacy/*' 2>/dev/null | head -1)"
```

Then restart Claude Code. Your original `ANTHROPIC_BASE_URL` is restored.

---

## Reference

For proxy options, troubleshooting, entity types, and model picker see
[reference.md](reference.md).
