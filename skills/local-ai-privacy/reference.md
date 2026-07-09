# Local AI Privacy: Reference

Detailed reference for the `local-ai-privacy` skill. Read this only when
the default path in `SKILL.md` doesn't cover a decision.

## Contents

- [Prerequisites](#prerequisites)
- [Redaction system prompt](#redaction-system-prompt)
- [Model picker](#model-picker)
- [Entity type catalog](#entity-type-catalog)
- [API key handling](#api-key-handling)
- [Placeholder format and re-substitution](#placeholder-format-and-re-substitution)
- [Verifying the MCP tool](#verifying-the-mcp-tool)
- [What the redaction model does NOT catch](#what-the-redaction-model-does-not-catch)

---

## Prerequisites

These three things must be true before the skill can run. Each is a one-time
user action; the skill checks for them and tells the user what to do if any is
missing.

**1. Lemonade Server running**

```bash
lemonade serve
```

Or launch from the Lemonade desktop app. The server binds to `http://localhost:13305`.

**2. Lemonade MCP registered in Claude Code (once per installation)**

```bash
claude mcp add lemonade --transport http http://localhost:13305/mcp
```

This registers the Lemonade MCP server globally in Claude Code. Do it once;
it persists across sessions and projects. Restart Claude Code after running it.

To verify it is registered:
```bash
claude mcp list
```

**3. Redaction model downloaded**

```bash
lemonade pull Qwen3-1.7B-GGUF
```

One-time download (~1.1 GB). After this, redaction runs fully offline.

---

## Redaction system prompt

The exact prompt is stored in `data/redaction-prompt.txt` inside the skill
folder. The agent reads this file and passes it verbatim as the `system`
message to `lemonade_chat`. Do not modify the JSON format line — the agent
parses the response as JSON and relies on the `masked` and `mapping` keys.

```
You are a data redactor. Your only job is to find and replace sensitive
content in the user's message.

Replace every instance of the following with a placeholder of the form
[TYPE_N] where TYPE is the entity type (all caps) and N is a zero-based
index unique per type:

- Personal identifiers: full names, usernames, employee IDs (NAME)
- Email addresses (EMAIL)
- Phone numbers (PHONE)
- Physical addresses, zip codes (ADDRESS)
- Social security numbers, national IDs (SSN)
- Dates of birth (DOB)
- Credit/debit card numbers (CARD)
- Bank account or routing numbers (BANK)
- Passwords, passphrases (PASSWORD)
- API keys, tokens, secret keys, bearer tokens (API_KEY)
- Private SSH/TLS keys or certificate fingerprints (PRIVATE_KEY)
- Internal hostnames, internal IP ranges, VPN IPs (INTERNAL_HOST)
- Database connection strings with credentials (DB_CONN)
- Medical record numbers, patient IDs (MEDICAL_ID)
- Any value that looks like a secret (SECRET)

Rules:
1. Do not redact publicly known company names, public domain names, or
   generic technical terms.
2. If the same value appears multiple times, always use the same placeholder.
3. If nothing is sensitive, return the original text unchanged.
4. Output ONLY a JSON object — no prose, no markdown fences.

JSON format:
{"masked": "...", "entities": ["TYPE", ...], "mapping": {"TYPE_N": "value", ...}}
```

---

## Model picker

| Model | Approx size | When to use |
|---|---|---|
| `Qwen3-1.7B-GGUF` | ~1.1 GB | **Default.** Fast CPU redaction; <2 s per prompt on Ryzen AI. |
| `Qwen3-4B-GGUF` | ~2.5 GB | When the user reports missed redactions on domain-specific secrets or multilingual content. |

To use a different model, pass it as the `model` argument in the
`lemonade:lemonade_chat` call in Step 2 of SKILL.md. Pull it first:

```bash
lemonade pull Qwen3-4B-GGUF
```

---

## Entity type catalog

| Placeholder prefix | What it covers |
|---|---|
| `NAME` | Full names, usernames, employee IDs |
| `EMAIL` | Email addresses |
| `PHONE` | Phone numbers in any local format |
| `ADDRESS` | Street addresses, postal codes |
| `SSN` | Social security numbers, national IDs |
| `DOB` | Dates of birth |
| `CARD` | Credit/debit card numbers |
| `BANK` | Bank account or routing numbers |
| `PASSWORD` | Passwords, passphrases |
| `API_KEY` | API keys, bearer tokens, OAuth secrets, auth tokens |
| `PRIVATE_KEY` | SSH private keys, TLS private keys, certificate fingerprints |
| `INTERNAL_HOST` | Internal hostnames, private IP ranges, VPN endpoints |
| `DB_CONN` | Database connection strings containing credentials |
| `MEDICAL_ID` | Medical record numbers, patient IDs |
| `SECRET` | Anything else that looks like a secret value |

---

## API key handling

If `LEMONADE_API_KEY` is set in the environment, the `lemonade:lemonade_chat`
MCP tool sends the key automatically — no extra action needed. The MCP
registration (`claude mcp add`) can also include it:

```bash
claude mcp add lemonade --transport http http://localhost:13305/mcp \
  --header "Authorization: Bearer $LEMONADE_API_KEY"
```

---

## Placeholder format and re-substitution

Placeholders follow `[TYPE_N]`:

- `TYPE` — entity category in caps (e.g. `EMAIL`, `API_KEY`)
- `N` — zero-based counter per type per request
- Same original value → same placeholder every time within one request

The `mapping` from the `lemonade_chat` response maps each placeholder back
to its original value. The agent keeps this in memory for the current session
and re-substitutes when showing the cloud's response to the user.

Example:
```
Original:  "Why is user john@corp.com hitting the limit on key sk-abc123?"
Masked:    "Why is user [EMAIL_0] hitting the limit on key [API_KEY_0]?"
Mapping:   {"EMAIL_0": "john@corp.com", "API_KEY_0": "sk-abc123"}

Cloud:     "[EMAIL_0] is sending 120 req/s, over the limit for [API_KEY_0]."
Shown:     "john@corp.com is sending 120 req/s, over the limit for sk-abc123."
```

The mapping is in-memory, scoped to the current session, and never written
to disk.

---

## Verifying the MCP tool

To confirm `lemonade_chat` is reachable before invoking the skill:

```bash
# Ping the MCP server
curl -s http://localhost:13305/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"ping"}'

# List available tools (lemonade_chat should appear)
curl -s http://localhost:13305/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | python -m json.tool
```

---

## What the redaction model does NOT catch

- **Obfuscated secrets:** base64, ROT13, or custom encoding will not be flagged.
- **Implicit references:** "use the key I mentioned earlier" with no value present.
- **Binary or non-text payloads** embedded in JSON.
- **Very long prompts** exceeding the model's context window (~32K tokens for
  `Qwen3-1.7B-GGUF`). Split the prompt or use `Qwen3-4B-GGUF` for better coverage.

For strict compliance requirements (HIPAA, PCI-DSS), combine this skill with a
purpose-built PII scanner on top of the redaction output.
