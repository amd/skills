---
name: local-ai-privacy
description: >-
  Redacts sensitive data from prompts before they reach cloud AI providers
  using a local Lemonade model. Use when the user wants to keep private data
  private while still using cloud LLMs; mentions PII, credentials, API keys,
  patient data, customer records, proprietary code, HIPAA, GDPR, or
  confidentiality; asks to "sanitize before sending to cloud", "mask sensitive
  data", "keep my data local", or "protect my privacy with Claude Code"; or
  works with codebases containing secrets and wants cloud assistance without
  data leakage. Do not use for image generation, TTS, or STT — use local-ai-use
  instead.
allowed-tools: Bash(curl:*), Bash(lemonade:*)
---

# Local AI Privacy (redact before cloud)

Intercept the user's content, redact sensitive data locally using a small
Lemonade model, then forward only the sanitized version to the cloud. Nothing
sensitive crosses the network boundary.

## Prerequisites check (do this first, every time)

Before redacting anything, verify the two prerequisites. If either fails, stop
and guide the user — never skip redaction and send unmasked content.

**1. Lemonade Server running:**
```bash
curl -s http://localhost:13305/api/v1/health
```
Expected: HTTP 200. If not reachable, tell the user:
> "Start Lemonade Server first: run `lemonade serve` in a terminal."

**2. Lemonade MCP registered in Claude Code:**
The `lemonade:lemonade_chat` tool must be available. If it is not registered,
tell the user to run this once:
```bash
claude mcp add lemonade --transport http http://localhost:13305/mcp
```
Then restart Claude Code. After that, the tool persists for all future sessions.

**3. Redaction model present:**
```bash
curl -s "http://localhost:13305/api/v1/models?show_all=true" | python -m json.tool | grep Qwen3-1.7B-GGUF
```
If the model is missing, pull it:
```bash
lemonade pull Qwen3-1.7B-GGUF
```
See [reference.md](reference.md#model-picker) for larger alternatives.

---

## Step 1: read the redaction system prompt

Read the system prompt from the skill's data file. Use the path where this
skill is installed (e.g. `~/.claude/skills/local-ai-privacy/data/redaction-prompt.txt`
or `.claude/skills/local-ai-privacy/data/redaction-prompt.txt`). The exact
path depends on where the user installed the skill — find it with:

```bash
find ~/.claude .claude -name "redaction-prompt.txt" 2>/dev/null | head -1
```

Read that file. Its contents are the system prompt you will pass to the
local model in Step 2.

## Step 2: call lemonade:lemonade_chat to redact

Call the `lemonade:lemonade_chat` MCP tool:

```json
{
  "model": "Qwen3-1.7B-GGUF",
  "messages": [
    {"role": "system", "content": "<contents of redaction-prompt.txt>"},
    {"role": "user",   "content": "<the content to protect>"}
  ],
  "temperature": 0,
  "max_tokens": 4096
}
```

The tool response is text. Parse it as JSON. Extract:
- `masked` — the sanitized version of the content (use this for the cloud)
- `mapping` — the `{"TYPE_N": "original value"}` map (keep in memory)

If the tool call fails or is unavailable:
- Run `lemonade status` and surface the output to the user
- **Do not fall back to sending unmasked content to the cloud**
- Stop and wait for the user to resolve the server issue

## Step 3: use the masked content for cloud reasoning

Send `masked` to the cloud model, not the original. The cloud reasons over
the redacted version and returns a response that may reference placeholders
(e.g. `[API_KEY_0]`).

Before showing any cloud response to the user, substitute placeholders back
using the `mapping`. The user always sees real values; the cloud never does.

## Routing rules

| Situation | Action |
|---|---|
| Content clearly contains or might contain sensitive data | Always redact (Steps 1–3) before any cloud call |
| Content is demonstrably safe (public docs, generic code with no secrets) | May skip redaction at your discretion. When in doubt, redact. |
| Cloud response contains `[TYPE_N]` placeholders | Re-substitute from `mapping` before showing the user |
| `lemonade:lemonade_chat` unavailable | Stop. Run `lemonade status`. Report to user. Never send unmasked. |

## What must always be redacted

Never send these to the cloud unmasked:
- Passwords, API keys, bearer tokens, OAuth secrets
- SSNs, passport numbers, national IDs
- Credit card or bank account numbers
- Private keys (SSH, TLS, GPG)
- Internal hostnames, VPN endpoints, private IP ranges
- Full names combined with contact details
- Medical record identifiers
- Database connection strings with credentials

---

## Reference

For the entity type catalog, model picker, placeholder format details, and
troubleshooting, see [reference.md](reference.md).
