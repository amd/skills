---
name: lemonade-router-builder
description: >-
  Turns a natural-language description of routing intent into a valid Lemonade
  `collection.router` policy JSON. The skill generates and validates the JSON
  only - it does not register it or call the live server.
  Use when the user wants to route requests between models ("route sensitive
  queries to X and everything else to Y"), generate a router/hybrid-router
  config or policy, author a collection.router JSON, split traffic between a
  small local model and a big/cloud model, add PII/jailbreak/topic classifiers
  to routing, or mentions Lemonade Router, routing rules, routing.router,
  candidates/default_model, keywords_any, semantic_similarity, or LLM-as-router.
  Fills every field the user did not specify with safe defaults.
---

# Lemonade Router Config Generator

Generate a **`collection.router` policy JSON** from a plain-English description
of how requests should be routed. The skill produces and validates the JSON
only - it does not call the live server, register the policy, or run requests
through it. The JSON is accepted by the strict server-side parser on the first
try and stays editable in the desktop app's Hybrid Router editor.

## Prerequisites

- **Lemonade Server v10.1.0+** running locally (`lemonade server start`).
  Required only to register and test the generated policy - the skill itself
  (JSON generation + offline validation) works without a live server.
- **No GPU or ROCm dependency** for authoring. The router policy is a JSON
  document; no hardware is needed to generate or validate it.
- **Python** (any 3.x) in PATH - used by the bundled offline validator
  (`scripts/validate.py`). No extra packages required.

The router picks one **candidate** model per request. Two authoring modes
exist, and choosing the right one is the first decision:

| Mode | JSON shape | When |
|------|-----------|------|
| **LLM-as-router** | `routing.router` block | The user describes intent only by *meaning* ("sensitive", "hard questions", "creative writing") with no concrete signals. A small LLM reads each prompt and picks the candidate. |
| **Rules** | `routing.rules` (+ optional `routing.classifiers`) | The user names any concrete signal: keywords, regex, length, tools, images, metadata, PII/topic classifiers, thresholds, "first match", fallback logic. Deterministic, no extra LLM call for simple conditions. |

`routing.router` is **mutually exclusive** with `routing.rules` and
`routing.classifiers` - never emit both.

## Step 1 - Extract from the user's words

- **Candidates**: the models that may *answer* requests. Verbatim model names
  (e.g. `Gemma-3-4b-it-GGUF`). If the user names none, ask - never invent
  model names. `lemonade list` or `GET /api/v1/models` shows what's available.
- **Default / fallback**: which candidate gets everything that matches nothing.
  If unstated, use the model the user framed as "local", "small", or "safe";
  otherwise the first candidate mentioned.
- **Signals**: every condition mentioned (keywords, patterns, length, images,
  tools, topics, PII, safety) and which model each one routes to.
- **Classifier models**: models named for *detection* rather than answering
  (BERT-style encoders, embedding models, an LLM used as judge).

## Step 2 - Scaffold

Always exactly this envelope (the parser rejects unknown or missing keys):

```json
{
  "version": "1",
  "model_name": "user.MyHybridRouter",
  "recipe": "collection.router",
  "components": [],
  "routing": { }
}
```

- `version` is the literal string `"1"`.
- `model_name` must start with `user.`; slug from the user's description if
  they gave a name (`user.<Name>` using only `[A-Za-z0-9._-]`). If they didn't
  name it, derive one from context instead of a fixed literal - e.g.
  `user.<slug-of-default-candidate>-Router` - so two different policies don't
  collide by default. **`/pull` is idempotent per `model_name`: registering a
  second policy under the same name silently overwrites the first.** If this
  conversation already produced an unnamed router, don't reuse the same
  derived name for the next one - ask, or pick a visibly different name.

## Step 3 - Candidates and default

```json
"candidates": ["<answering models>"],
"default_model": "<one of candidates>"
```

`default_model` MUST be listed in `candidates`. Candidates should be
chat-capable LLMs - not embedding, classification, or image models.

## Step 4 - Mode A: LLM-as-router

```json
"router": {
  "type": "llm",
  "model": "<small chat LLM>",
  "prompt": "You route user requests to the best model. <one sentence per candidate: when to pick it, using the exact model name>."
}
```

- `model` defaults to the smallest candidate (it may be a candidate; it also
  works as a separate small model).
- **Write intent only - never specify a reply format and never use imperative
  "Pick X" phrasing.** The engine unconditionally appends its own contract
  after your prompt: it lists the candidate names and demands a strict JSON
  reply `{"model": "<name>", "rationale": "<one sentence>"}`, then falls back
  to `default_model` on any deviation. A prompt that says "reply with ONLY the
  model name", "Pick Model-A", "respond with the model name", or similar is
  wrong about the wire format and causes weaker judge models to reply with a
  bare string that fails to parse - silently falling back to `default_model`
  on every request with no visible error.

  **Bad** (do not write): `"Pick Qwen3.5-9B-GGUF for sensitive queries, pick Qwen3.5-9B-NoThinking for everything else."`
  **Good**: `"Route to Qwen3.5-9B-GGUF when the request appears sensitive or contains personal information. Route to Qwen3.5-9B-NoThinking for all other requests."`

  Only describe *when* each candidate is appropriate. Never say "pick", "output", "reply with", or "respond with".
- **NEVER emit `rules` or `classifiers` in this mode.** The `routing` object
  in Mode A must contain exactly: `candidates`, `default_model`, and `router`.
  Adding `rules` or `classifiers` alongside `router` is a schema violation
  that the server parser rejects. If you catch yourself writing both, stop and
  remove `rules`/`classifiers` entirely.

## Step 5 - Mode B: classifiers

Only declare classifiers the rules actually reference. Three types:

```json
{ "id": "clf-1", "type": "classifier", "model": "<classification model>",
  "labels": ["PII", "Jailbreak"], "default_label": "PII", "on_error": "match_false" }

{ "id": "clf-2", "type": "semantic_similarity", "model": "<embedding model>",
  "reference_phrases": { "shopping": ["I want to shop for pants", "add to cart"] },
  "default_label": "shopping", "on_error": "match_false" }

{ "id": "clf-3", "type": "llm", "model": "<chat LLM>",
  "prompt": "Classify the request into only labels SAFE, RISKY",
  "labels": ["SAFE", "RISKY"], "default_label": "SAFE", "on_error": "match_false" }
```

Hard constraints (parser-enforced - see `reference.md` for the full matrix):

- `classifier` type: model should be a text-classification model (an
  `onnxruntime` encoder like `Bert-Phishing-ONNX`); `labels` must match the
  model's actual output labels. A chat LLM here is legal (LLM-as-classifier
  via chat) but prefer `type: "llm"` for that - it is explicit and prompted.
- `semantic_similarity`: `reference_phrases` is `{concept: [phrases...]}`,
  at least one concept, each with at least one phrase. Concept names ARE the
  labels - a `labels` key is **rejected** for this type. Model must be an
  embedding model. Give 3–5 varied phrases per concept when inventing them.
- `llm`: `prompt` AND non-empty `labels` are both required. **Write intent
  only - never tell the model how to format its reply.** The engine appends
  its own `{"model": "<chosen_label>", "rationale": "..."}` contract after
  your prompt (the same contract as `routing.router`). An authored line like
  "Reply with exactly one label: SAFE or RISKY" causes weaker models to output
  bare `SAFE`, which the parser rejects - the score comes back empty and the
  rule silently never fires. Describe what makes a request belong to each
  label; leave the reply format to the engine.
- `default_label`, when present, must be one of the labels/concepts.
- Defaults when unspecified: `id` = `clf-1`, `clf-2`, …; `on_error` =
  `"match_false"` (fail-open: a broken classifier doesn't match, so requests
  fall through - use `"match_true"` only when the user wants fail-closed
  safety); `default_label` = the first label.

## Step 6 - Mode B: rules

```json
"rules": [
  { "id": "rule-1", "match": { ... }, "route_to": "<candidate>",
    "outputs": { "reason": "<optional free-form>" } }
]
```

- **Order matters - first match wins.** Put the most specific /
  privacy-critical rules first (a "sensitive stays local" rule must precede a
  "code goes to the big model" rule, or coding prompts with PII leak).
- `route_to` MUST be a candidate. `id` uses only `[A-Za-z0-9._-]`; default
  `rule-1`, `rule-2`, ….
- No rule for the "everything else" case - that is `default_model`.

**Match conditions** - combine with `all` (AND), `any` (OR), `not`; one
condition per leaf object; nesting is allowed:

| Leaf | Example | Notes |
|------|---------|-------|
| `keywords_any` / `keywords_all` | `{ "keywords_any": ["SSN", "Email"] }` | case-insensitive substring - `"hi"` matches inside `"this"`, `"shipping"`, `"high"`, etc. Use `regex` with `\b...\b` when word-boundary precision is needed |
| `regex` | `{ "regex": "\\b\\d{3}-?\\d{2}-?\\d{4}\\b" }` | ECMAScript flavor |
| `min_chars` / `max_chars` | `{ "min_chars": 4000 }` | input length, UTF-8 bytes, non-negative integer |
| `has_tools` / `has_images` | `{ "has_images": true }` | booleans |
| `classifier` | `{ "classifier": "clf-1", "label": "PII", "min_score": 0.5 }` | band test; `min_score`/`max_score` in [0,1]; default `min_score` 0.5; omit `label` only if the classifier has `default_label` |
| `metadata` | `{ "metadata": { "key": "consent", "equals": "denied" } }` | exactly one of `equals` / `any` / `exists`; note: not editable in the desktop UI yet - use only when the user asks for metadata routing |

## Step 7 - Components

`components` = union of: all `candidates` + every classifier `model` + the
`router.model` (Mode A). Deduplicate, keep order stable. The parser rejects
any referenced model that is not declared here.

## Step 8 - Validate and output curl commands

These two actions are a single mandatory step. Do not stop between them.

**8a. Run the offline validator** before presenting anything to the user:

```bash
python scripts/validate.py router.json    # Windows
python3 scripts/validate.py router.json   # macOS/Linux
```

It exits 0 with `"ready": true` when there are no errors. If it reports
errors, fix the JSON and re-run. Do not present a policy that fails this
check.

**8b. Immediately after validation passes, print these three curl commands**
as plain text for the user to copy and run. This is not optional. Fill in
`<model-id>` and `<model_name>` from the policy, and a short `<test prompt>`
that should hit the first rule. Do not execute these with Bash or any tool —
print them as text only.

```bash
# 1. Check a model exists before registering
curl http://localhost:13305/api/v1/models/<model-id>

# 2. Register the policy (idempotent - re-POST to update)
curl -X POST http://localhost:13305/api/v1/pull \
     -H "Content-Type: application/json" --data-binary @router.json

# 3. Route a request and inspect the decision
curl -X POST http://localhost:13305/api/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "<model_name>", "route_trace": true,
          "messages": [{"role": "user", "content": "<test prompt>"}]}'
```

The `x-lemonade-route` response header carries the matched rule id (or
`default`). With `"route_trace": true` the body also carries
`x_lemonade_route`: `{ route_to, matched_rule, default_used, outputs,
trace[] }` - useful for verifying each rule fires as expected.

## Defaults summary

| Field | Default when the user doesn't say |
|-------|-----------------------------------|
| `model_name` | `user.<default-candidate-slug>-Router` (never reuse a name already used earlier in this conversation) |
| `default_model` | the "small/local/safe" candidate, else first mentioned |
| mode | rules if any concrete signal is named, else LLM-as-router |
| classifier `id` / rule `id` | `clf-N` / `rule-N` |
| `on_error` | `match_false` |
| `default_label` | first label / concept |
| `min_score` | `0.5` |
| `outputs` | omit |
| router prompt | intent only - no reply-format instruction (Step 4) |

Worked NL → JSON pairs live in `examples.md`; the full schema, parser error
matrix, and model-capability table live in `reference.md`; the offline
validator is `scripts/validate.py` (run it - see Step 8).
