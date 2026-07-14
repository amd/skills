# Local AI Privacy: Reference

## Contents

- [How it works](#how-it-works)
- [Endpoints](#endpoints)
- [Proxy log and verification](#proxy-log-and-verification)
- [Troubleshooting](#troubleshooting)
- [Proxy options](#proxy-options)
- [Model picker](#model-picker)
- [What gets detected](#what-gets-detected)
- [Redaction system prompt](#redaction-system-prompt)
- [API key handling](#api-key-handling)
- [Placeholder format and the session map](#placeholder-format-and-the-session-map)
- [Limitations](#limitations)

---

## How it works

`start.py` runs once. It:
1. Reads `ANTHROPIC_BASE_URL` from `~/.claude/settings.json` → that is the real cloud endpoint
2. Writes `~/.claude/skills/local-ai-privacy/proxy.conf` with the cloud URL and Lemonade config
3. Starts `proxy.py` as a background process (survives terminal close) and writes its PID to `proxy.pid`
4. **Only if the proxy answers its health check**, patches `settings.json`: `ANTHROPIC_BASE_URL` → `http://127.0.0.1:8317`. If the proxy fails to come up, `start.py` aborts and leaves `settings.json` untouched, so setup can never strand Claude Code on a dead endpoint. If port 8317 is already taken it auto-selects a free port and writes that everywhere.

After a Claude Code restart, all requests flow:

```
Claude Code → proxy (127.0.0.1:8317) → local model redaction → real cloud endpoint
                                        (on-device)             (masked text only)
            ← rehydrated response     ←                        ←
```

The proxy keeps a single **in-memory** map for the life of the process:
`real value → placeholder` (and back). Every endpoint shares it, so the preview
the user approves and the request that actually reaches the cloud use identical
placeholders, and the reply can be rehydrated unambiguously. The map is never
written to disk; restarting the proxy clears it.

`stop.py` kills the proxy process and restores the original `ANTHROPIC_BASE_URL`
(reading the actual port back from `proxy.conf`).

---

## Endpoints

| Method / path | Purpose | Touches cloud? |
|---|---|---|
| `GET /health` | Liveness check (`{"status":"ok"}`). Used by `start.py` and the skill. | No |
| `POST /redact` | **Preview.** Body `{"text": "..."}` (or a full request). Runs the local model, updates the session map, and returns the entities that will be masked. | **No — local only** |
| `POST /v1/messages` | **Load-bearing.** Redacts `system` + every message + tool blocks, forwards only the masked body to the cloud, then rehydrates placeholders in the response. | Yes (masked) |
| anything else | Passed through unchanged (e.g. `GET /v1/models`). | Yes |

`/redact` response shape:
```json
{
  "ok": true,
  "masked": "My SSN is [SSN_1], email [EMAIL_1] — draft a dispute letter.",
  "entities": [
    {"type": "SSN",   "original": "123-45-6789", "placeholder": "[SSN_1]"},
    {"type": "EMAIL", "original": "a@b.com",      "placeholder": "[EMAIL_1]"}
  ]
}
```

Call it yourself from a terminal to confirm the masking without touching the
cloud (see the "Verify it's working" section in SKILL.md).

---

## Proxy log and verification

The proxy writes to `~/.claude/skills/local-ai-privacy/proxy.log`.

```bash
tail -f ~/.claude/skills/local-ai-privacy/proxy.log
```

Representative log lines:
```
[proxy] 10:23:01 Preview: 2 entities in submitted text
[proxy] 10:23:04 Forwarding redacted request (2 known entities)
```

Confirm the proxy is running:
```bash
curl -s http://localhost:8317/health
# → {"status":"ok"}
```

Try a preview without leaving your machine:
```bash
curl -s http://localhost:8317/redact -H 'Content-Type: application/json' \
  -d '{"text":"My SSN is 123-45-6789 and email a@b.com"}' | python3 -m json.tool
```

Check `settings.json` was patched:
```bash
python3 -c "import json,os; d=json.load(open(os.path.expanduser('~/.claude/settings.json'))); print(d['env']['ANTHROPIC_BASE_URL'])"
# → http://127.0.0.1:8317
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot reach Lemonade Server` at start | Server not running | `lemonade serve` |
| `ModuleNotFoundError: requests` | Missing dependency | `pip install requests` |
| Claude Code gets connection refused | Proxy died or not started | Re-run `start.py`, check `proxy.log` |
| Claude Code gets 502 (`Cloud error`) | Cloud endpoint unreachable | Check network; verify original URL in `proxy.conf` |
| Claude Code gets 502 (`redaction could not be completed`) | Lemonade unreachable, or the model kept returning unparseable output across all retries. The proxy **fails closed** — it blocks rather than sending unredacted data to the cloud | Check Lemonade is up and see `proxy.log`; retry. For persistent parse failures, raise `redaction_retries` or set a stronger `redaction_model` (e.g. `Qwen3-4B-GGUF`) in `proxy.conf` |
| Response shows `[SSN_1]` literally | Cloud model altered the placeholder so rehydration missed it | Usually transient; the injected system note tells the model to preserve tokens. Check `proxy.log` |
| A `[SSN_1]` shows literally right after a proxy restart | The in-memory map was cleared, so a placeholder with no raw value left in the transcript can't be rehydrated | Usually self-heals: your raw text is still in the transcript and is re-masked to the same placeholder on the next request. If not, start a fresh message |
| `Address already in use` at start | Another service on 8317 | `start.py` now auto-picks a free port; no action needed |
| Proxy slow on first request after restart | Model cold-load by Lemonade | Normal — subsequent requests are fast once the model is loaded |

If the proxy dies unexpectedly:
```bash
tail -20 ~/.claude/skills/local-ai-privacy/proxy.log
python3 ~/.claude/skills/local-ai-privacy/scripts/start.py
```

---

## Proxy options

`proxy.conf` controls all proxy behaviour. Edit it to change defaults without
re-running `start.py`:

```json
{
  "cloud_url":            "https://api.anthropic.com",
  "lemonade_url":         "http://localhost:13305",
  "redaction_model":      "Qwen3.6-35B-A3B-NoThinking",
  "proxy_port":           8317,
  "redaction_retries":    3,
  "redaction_max_tokens": 8192
}
```

`redaction_retries` (default `3`) is how many times the proxy re-asks the
local model when it returns unparseable output before giving up. Small models
are non-deterministic, so a couple of retries drives the failure rate near
zero. If all attempts fail the proxy **fails closed** — it returns a 502 and
blocks the request rather than forwarding unredacted text to the cloud.

`redaction_max_tokens` (default `8192`) caps the local model's reply. Keep it
generous so a large or reasoning model can't get its JSON truncated (which would
fail closed). Raise it (e.g. `16384`) for a thinking model.

After editing, restart the proxy:
```bash
python3 ~/.claude/skills/local-ai-privacy/scripts/stop.py
python3 ~/.claude/skills/local-ai-privacy/scripts/start.py
```

---

## Model picker

| Model | Approx size | Notes |
|---|---|---|
| `Qwen3.6-35B-A3B-NoThinking` | ~18 GB | **Default.** 35B MoE with ~3B active params — strong PII coverage at near-small-model speed, and no reasoning overhead so it returns clean JSON. |
| `Qwen3.6-35B-A3B-ThinkingCoder` | ~18 GB | Same base, but emits `<think>` reasoning first. Higher latency on every request and needs a larger `redaction_max_tokens`; usually not worth it for extraction. |
| `Qwen3-1.7B-GGUF` | ~1.1 GB | Lightweight fallback. Fastest and lowest RAM, but weaker coverage of unusual/edge-case PII. |

Switch by setting `redaction_model` in `proxy.conf` (or `LOCALAI_REDACTION_MODEL=... python3 start.py`), then restart the proxy. Pull first if needed: `lemonade pull <model>`. For a thinking model, also raise `redaction_max_tokens` (e.g. `16384`) so reasoning can't truncate the JSON.

---

## What gets detected

Detection is **open-ended**, not a fixed list. The model is asked to flag
anything sensitive, private, confidential, or proprietary, and to pick its own
short label. The labels below are just common examples of what it returns:

`NAME`, `EMAIL`, `PHONE`, `ADDRESS`, `SSN` / `GOVERNMENT_ID`, `DOB`, `CARD`,
`BANK`, `PASSWORD`, `API_KEY`, `PRIVATE_KEY`, `INTERNAL_HOST`, `DB_CONN`,
`MEDICAL_ID`, `PROJECT` (unreleased product/project names), `SOURCE`
(proprietary code), `SECRET` (anything else).

Because the label is freeform, the proxy also catches things a fixed PII list
would miss — internal hostnames, unreleased codenames, trade secrets,
confidential figures — as long as the local model recognises them as sensitive.

---

## Redaction system prompt

Stored in `data/redaction-prompt.txt` inside the skill folder. The proxy reads
it on every request; edit it to widen or narrow what counts as sensitive — no
code change, no restart needed.

The model returns a JSON array of `{"text": <exact span>, "label": <TYPE>}`.
The proxy assigns the placeholder numbers itself, so the model's own numbering
does not matter, and it skips any span shorter than 3 characters (a 1–2 char
match would smear across unrelated text).

---

## API key handling

The proxy forwards all request headers from Claude Code to the cloud
unchanged — `x-api-key`, `anthropic-version`, custom headers. No auth changes
are needed.

If `LEMONADE_API_KEY` is set in your environment, the proxy adds it when
calling Lemonade.

---

## Placeholder format and the session map

Format: `[TYPE_N]` where TYPE is the entity category in caps and N is a
**1-based** counter unique per type. Placeholders are assigned by the proxy (not
the model) and are **stable for the whole proxy session**: the same real value
always maps to the same placeholder across every request (and the `/redact`
endpoint), and a new value gets the next index for its type.

Because the map is shared, PII discovered once (in the current or an earlier
turn, or via `/redact`) is masked in every later request even if the model does
not re-flag it — so repeated values cannot slip through on a later turn. The
proxy also refuses to redact its **own** placeholder tokens (e.g. a `[SSN_1]`
that reappears in a tool result), which prevents runaway nesting like
`[SSN_1]` → `[SSN_2]` → `[SSN_3]`.

Discovery runs the local model over `system`, all message content, and tool
blocks, in line-aligned chunks. Chunks already scanned in a previous turn are
cached by hash and skipped, so the whole transcript is not re-scanned every
request.

On non-streaming responses the proxy rehydrates all `content` blocks before
returning to Claude Code. On SSE streaming responses it **buffers text across
`text_delta` events** and only rehydrates at safe boundaries, so a placeholder
split across streaming chunks (`[SSN` then `_1]`) is still restored correctly.
It also injects a short system note instructing the cloud model to preserve
`[TYPE_N]` tokens verbatim.

---

## Limitations

- Secrets encoded in base64, ROT13, or custom obfuscation are not detected
- Implicit references ("the key I mentioned earlier") with no literal value present
- Non-text content blocks (images, binary data) are passed through unredacted
- Very long single lines exceeding the model's context window (~32K tokens for 1.7B)
- The in-memory map is cleared when the proxy restarts, but self-heals: your raw
  text stays in Claude Code's transcript and is re-discovered (and re-mapped to
  the same placeholder) on the next request
- Placeholder rehydration is a literal string swap, so if the cloud model
  rewrites a token (e.g. `[SSN_1]` → `SSN #1`) that value stays masked in the
  reply

For strict compliance (HIPAA, PCI-DSS), layer a deterministic PII scanner
on top.
