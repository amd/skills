# Local AI Privacy: Reference

## Contents

- [How it works](#how-it-works)
- [Proxy log and verification](#proxy-log-and-verification)
- [Troubleshooting](#troubleshooting)
- [Proxy options](#proxy-options)
- [Model picker](#model-picker)
- [Entity type catalog](#entity-type-catalog)
- [Redaction system prompt](#redaction-system-prompt)
- [API key handling](#api-key-handling)
- [Placeholder format](#placeholder-format)
- [Limitations](#limitations)

---

## How it works

`start.py` runs once. It:
1. Reads `ANTHROPIC_BASE_URL` from `~/.claude/settings.json` → that is the real cloud endpoint
2. Writes `~/.claude/skills/local-ai-privacy/proxy.conf` with the cloud URL and Lemonade config
3. Starts `proxy.py` as a background process using `pythonw.exe` (Windows) so it survives terminal close
4. Writes the PID to `proxy.pid` so `stop.py` can kill it later
5. Patches `settings.json`: `ANTHROPIC_BASE_URL` → `http://localhost:8080`

After a Claude Code restart, all requests flow:

```
Claude Code → proxy (localhost:8080) → Lemonade redaction → cloud endpoint
                                      (local, on-device)    (clean text only)
```

`stop.py` kills the proxy process and restores the original `ANTHROPIC_BASE_URL`.

---

## Proxy log and verification

The proxy writes to `~/.claude/skills/local-ai-privacy/proxy.log`.

```bash
# Watch live
Get-Content "$env:USERPROFILE\.claude\skills\local-ai-privacy\proxy.log" -Wait   # PowerShell
tail -f ~/.claude/skills/local-ai-privacy/proxy.log                               # bash
```

Per-request log lines:
```
[proxy] 10:23:01 Redacted entities: ['API_KEY', 'EMAIL']
[proxy] 10:23:01 Forwarding redacted request to cloud
```
```
[proxy] 10:23:15 No sensitive entities found
```

To confirm the proxy is running:
```bash
curl -s http://localhost:8080/__proxy_health
# → {"status":"ok"}
```

To check `settings.json` was patched:
```bash
python -c "import json; d=json.load(open('$env:USERPROFILE/.claude/settings.json')); print(d['env']['ANTHROPIC_BASE_URL'])"
# → http://localhost:8080
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot reach Lemonade Server` at start | Server not running | `lemonade serve` |
| `ModuleNotFoundError: requests` | Missing dependency | `pip install requests` |
| Claude Code gets connection refused | Proxy died or not started | Re-run `start.py`, check `proxy.log` |
| Claude Code gets 502 | Cloud endpoint unreachable | Check network; verify original URL in `proxy.conf` |
| Response shows `[SSN_0]` literally | Streaming re-substitution missed | Check `proxy.log`; model may have returned non-JSON |
| Redaction returns original text | Lemonade gave non-JSON | Use `--redaction-model Qwen3-4B-GGUF` in `proxy.conf` |
| `Address already in use` | Another process on 8080 | Edit `proxy_port` in `proxy.conf`, update `settings.json` manually |
| Proxy slow on first request after restart | Model cold-load by Lemonade | Normal — subsequent requests are fast once model is loaded |

If the proxy dies unexpectedly:
```bash
# Check the log for the error
cat ~/.claude/skills/local-ai-privacy/proxy.log | tail -20

# Restart manually
python ~/.claude/skills/local-ai-privacy/scripts/start.py
```

---

## Proxy options

`proxy.conf` controls all proxy behaviour. Edit it to change defaults without
re-running `start.py`:

```json
{
  "cloud_url":       "https://llm-api.amd.com/Anthropic",
  "lemonade_url":    "http://localhost:13305",
  "redaction_model": "Qwen3-1.7B-GGUF",
  "proxy_port":      8080
}
```

After editing, kill and restart the proxy:
```bash
python ~/.claude/skills/local-ai-privacy/scripts/stop.py
python ~/.claude/skills/local-ai-privacy/scripts/start.py
```

---

## Model picker

| Model | Approx size | Notes |
|---|---|---|
| `Qwen3-1.7B-GGUF` | ~1.1 GB | **Default.** <2 s redaction on Ryzen AI. Good for standard PII and credentials. |
| `Qwen3-4B-GGUF` | ~2.5 GB | Better coverage of domain-specific secrets, unusual formats, multilingual content. |

Pull before switching: `lemonade pull Qwen3-4B-GGUF`

---

## Entity type catalog

| Placeholder | What is redacted |
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
| `API_KEY` | API keys, bearer tokens, OAuth secrets |
| `PRIVATE_KEY` | SSH/TLS private keys, certificate fingerprints |
| `INTERNAL_HOST` | Internal hostnames, private IP ranges, VPN endpoints |
| `DB_CONN` | Database connection strings with credentials |
| `MEDICAL_ID` | Medical record numbers, patient IDs |
| `SECRET` | Anything else that looks like a secret |

---

## Redaction system prompt

Stored in `data/redaction-prompt.txt` inside the skill folder. The proxy
reads this file on every request. Edit it to add or remove entity types —
no code change needed, no restart required.

---

## API key handling

The proxy forwards all request headers from Claude Code to the cloud
unchanged — `x-api-key`, `Ocp-Apim-Subscription-Key`, `anthropic-version`,
custom headers. No auth changes are needed.

If `LEMONADE_API_KEY` is set in your environment, the proxy adds it when
calling Lemonade.

---

## Placeholder format

Format: `[TYPE_N]` where TYPE is the entity category in caps and N is a
zero-based counter unique per type per request. The same original value
always maps to the same placeholder within one request.

On SSE streaming responses, the proxy re-substitutes inside each
`text_delta` event. On non-streaming responses it re-substitutes in
all `content` blocks before returning to Claude Code.

---

## Limitations

- Secrets encoded in base64, ROT13, or custom obfuscation are not detected
- Implicit references ("the key I mentioned earlier") with no literal value present
- Non-text content blocks (images, binary data)
- Very long messages exceeding the model's context window (~32K tokens for 1.7B)
- The proxy only handles the Anthropic Messages API — tool results and
  system prompts are not currently redacted (user messages only)

For strict compliance (HIPAA, PCI-DSS), layer a deterministic PII scanner
on top.
