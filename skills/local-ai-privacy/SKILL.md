---
name: local-ai-privacy
description: >-
  Keeps sensitive data private when using Claude Code with a cloud LLM.
  Use when the user wants to protect PII, credentials, API keys, patient data,
  customer records, or proprietary code from reaching the cloud; mentions
  HIPAA, GDPR, or data confidentiality; asks to "sanitize before sending to
  cloud", "mask sensitive data", "keep my data local", or "protect my privacy
  with Claude Code". Sets up a local redacting proxy that intercepts every
  Claude Code request, redacts sensitive content via a local Lemonade model,
  and forwards only the masked version to the cloud — transparently, with no
  change to how the user types prompts.
  Do not use for image generation, TTS, or STT — use local-ai-use instead.
allowed-tools: Bash(curl:*), Bash(lemonade:*), Bash(python:*), Bash(python3:*)
---

# Local AI Privacy (redacting proxy)

Redacts sensitive content locally before it ever reaches the cloud. A local
proxy sits between Claude Code and Anthropic: it swaps real values for
placeholders (`123-45-6789` → `[SSN_1]`), forwards only the masked request, and
swaps the real values back into the reply — all on your machine.

```
1. You type raw PII in the Claude Code terminal            ── your machine
      "My SSN is 123-45-6789, email a@b.com…"
                    │
2. Claude Code sends the request to ANTHROPIC_BASE_URL,
   which now points at the local proxy (localhost:8317)    ── your machine
                    ▼
3. The proxy asks the local model to find PII and
   swaps in placeholders                                   ── your machine
      "My SSN is [SSN_1], email [EMAIL_1]…"
                    │
4. Proxy sends ONLY the masked request to the cloud ───────► api.anthropic.com
                    ▼
5. The cloud model replies using the placeholders          ── Anthropic servers
                    │
6. Proxy swaps [SSN_1] → 123-45-6789 back in, locally       ── your machine
                    ▼
7. Claude Code shows you the reply with real values
```

Protection happens at **step 3, automatically, on every request** — there is
nothing to invoke per prompt. This skill's job is the one-time setup and
teardown of that proxy.

---

## Setup (run once)

### 1. Check prerequisites

**Lemonade Server running:**
```bash
curl -s http://localhost:13305/api/v1/health
```
If not running: `lemonade serve`

**Redaction model present:**
```bash
curl -s "http://localhost:13305/api/v1/models?show_all=true" | python3 -m json.tool | grep Qwen3.6-35B-A3B-NoThinking
```
If missing (one-time, ~18 GB — a 35B MoE): `lemonade pull Qwen3.6-35B-A3B-NoThinking`

### 2. Start the proxy

```bash
python3 "$(find ~/.claude .claude -name start.py -path '*/local-ai-privacy/*' 2>/dev/null | head -1)"
```

`start.py` saves your current cloud endpoint, starts the proxy in the
background, and — **only if the proxy comes up healthy** — patches
`ANTHROPIC_BASE_URL` in `settings.json` to route through it. If the proxy fails
to start it aborts and leaves `settings.json` untouched, so setup can never
break your connection. It auto-picks a free port if 8317 is taken.

### 3. Restart Claude Code

Close and reopen so Claude Code reads the new base URL. From now on **every**
request is redacted automatically.

---

## Everyday use

Nothing. Just type normally — the proxy redacts every request transparently, as
shown in the diagram above. You do not need to invoke this skill again.

---

## Verify it's working (optional)

Because redaction happens *before* the cloud model, the model itself can't show
you the mask — it never sees your raw data. To check the masking with your own
eyes, run this **in a terminal** (it hits only the local model, never the
cloud):

```bash
curl -s localhost:8317/redact -H 'Content-Type: application/json' \
  -d '{"text":"My SSN is 123-45-6789, email a@b.com"}' | python3 -m json.tool
```

You'll see each value mapped to a placeholder (`123-45-6789 → [SSN_1]`). You can
also watch live traffic:
```bash
tail -f ~/.claude/skills/local-ai-privacy/proxy.log
# → Forwarding redacted request (N known entities)
```

---

## Stop (restore direct cloud connection)

```bash
python3 "$(find ~/.claude .claude -name stop.py -path '*/local-ai-privacy/*' 2>/dev/null | head -1)"
```
Then restart Claude Code. Your original `ANTHROPIC_BASE_URL` is restored.

---

## Reference

For endpoints (`/health`, `/redact`, `/v1/messages`), the placeholder map,
proxy options, troubleshooting, and the model picker, see
[reference.md](reference.md).
