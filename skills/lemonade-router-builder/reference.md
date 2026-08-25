# Lemonade Router Config: Reference

Detailed contract for the `lemonade-router-builder` skill. Read this when the
generation steps in `SKILL.md` leave a question open. The server-side parser
(`routing_policy_parser.cpp`) is strict: it rejects unknown keys at every
level, so never emit fields not listed here.

## Contents

- [Root document](#root-document)
- [routing block](#routing-block)
- [Classifier entries](#classifier-entries)
- [Rules and match expressions](#rules-and-match-expressions)
- [Model capability requirements](#model-capability-requirements)
- [Validation checklist](#validation-checklist)
- [Registering, invoking, tracing](#registering-invoking-tracing)
- [Desktop-editor compatibility](#desktop-editor-compatibility)

---

## Root document

Allowed keys: `version`, `model_name`, `recipe`, `components`, `models`,
`routing`.

| Key | Required | Constraint |
|-----|----------|-----------|
| `version` | yes | literal `"1"` (string, not number) |
| `model_name` | yes | non-empty; prefix `user.`; chars `[A-Za-z0-9._-]` after the prefix |
| `recipe` | yes | literal `"collection.router"` |
| `components` | yes | non-empty array; every model the policy references |
| `models` | no | embedded component definitions (produced by exports; don't generate) |
| `routing` | yes | object, see below |

## routing block

Allowed keys: `candidates`, `default_model`, `router`, `classifiers`, `rules`.

| Key | Required | Constraint |
|-----|----------|-----------|
| `candidates` | yes | non-empty array of unique names; each declared in `components` |
| `default_model` | yes | must be listed in `candidates` |
| `router` | either this or `rules` | `{ "type": "llm", "model": <component>, "prompt": <non-empty> }` - no other keys. Mutually exclusive with `rules` AND `classifiers`. Desugars server-side into one llm classifier + identity rules. |
| `classifiers` | no (rules mode only) | array; every entry must be referenced by some rule ideally |
| `rules` | either this or `router` | non-empty array; first match wins |

## Classifier entries

Allowed keys: `id`, `type`, `model`, `prompt`, `labels`, `default_label`,
`reference_phrases`, `on_error`.

| Field | `classifier` | `semantic_similarity` | `llm` |
|-------|--------------|----------------------|-------|
| `id` | required, unique | required, unique | required, unique |
| `model` | required | required (embedding model) | required (chat LLM) |
| `prompt` | - | - | **required**, non-empty |
| `labels` | optional (must match the model's real output labels) | **rejected** (concepts are the labels) | **required**, ≥ 1 entry |
| `reference_phrases` | - | **required**: `{concept: [phrase, ...]}`, ≥1 concept, ≥1 phrase each | - |
| `default_label` | optional; must be in `labels`; requires `labels` non-empty | optional; must be a concept name | optional; must be in `labels` |
| `on_error` | `"match_false"` (default) or `"match_true"` | same | same |

Notes:

- `on_error: match_false` = fail-open (classifier failure → condition doesn't
  match → request falls through, usually to `default_model`).
  `match_true` = fail-closed (failure counts as a match - for safety rules
  where "can't check" should route to the restricted/local path).
- A label-less `classifier` entry is legal (single-score models); its rule
  leaves then must not name a `label`.
- **`labels` on a `classifier` entry cannot be checked offline.**
  `scripts/validate.py` only verifies internal consistency (rules reference
  labels you declared) - it has no way to confirm those labels match the
  model's real output categories, since that requires calling the live model.
  A mismatched label doesn't error at registration or at request time; it
  silently scores `0.0` on every request, and with the default
  `on_error: match_false` the rule permanently falls through with no visible
  error. Verify by sending a test prompt designed to clearly hit the label and
  checking the trace `score` - stuck at `0.0` on an obvious match means the
  label name is wrong, not that the input didn't match.
- Scores are `{label: score}` with each score in `[0, 1]`.
- **`llm` classifier wire contract** - the engine appends its own JSON reply
  contract after every `llm` classifier prompt, exactly as it does for
  `routing.router`. The model must reply `{"model": "<chosen_label>",
  "rationale": "<one sentence>"}`. The engine then scores the chosen label
  `1.0` and all other labels `0.0`. **Never tell the model how to format its
  reply** - an authored instruction like "Reply with exactly one label: X or Y"
  causes weaker models to output bare `X`, the parser rejects it, the score
  comes back empty, and the rule silently never fires. Describe classification
  intent only (what makes a request COMPLEX vs SIMPLE, SAFE vs RISKY, etc.).
  If it still never fires after that, this is the same judge-capability limit
  as `routing.router` (SKILL.md Step 4) - swap in a stronger model.

## Rules and match expressions

Rule keys: `id`, `match`, `route_to`, `outputs`.

- `id`: required, unique, charset `[A-Za-z0-9._-]`.
- `route_to`: required, must be a candidate.
- `outputs`: optional object, copied verbatim into the routing decision - the
  engine never interprets it. Use for human-readable reasons or downstream
  tags (`{"reason": "privacy"}`, `{"route_category": "cloud"}`).
- `match`: a match expression.

Match-expression grammar:

- Logical node: exactly ONE of `all: [...]`, `any: [...]`, `not: {...}` and
  nothing else in that object. Arrays non-empty. Nesting allowed (depth ≤ 64).
- Leaf node: one or more condition keys (multiple keys in one leaf are an
  implicit AND, but prefer one condition per leaf - see
  [Desktop-editor compatibility](#desktop-editor-compatibility)).

| Condition | Value | Semantics |
|-----------|-------|-----------|
| `keywords_any` | non-empty array of non-empty strings | case-insensitive substring, any - `"hi"` also matches inside `"this"`, `"shipping"`, `"high"`, etc. Use `regex` `\b...\b` for word-boundary matching |
| `keywords_all` | same | all must appear; same substring semantics as `keywords_any` |
| `regex` | non-empty string | ECMAScript regex over the input text |
| `min_chars` / `max_chars` | integer ≥ 0 | input length in **UTF-8 bytes** |
| `has_tools` / `has_images` | boolean | request carries `tools[]` / image parts. `false` is legal and means "must NOT have"; `{"not": {"has_tools": true}}` is the equivalent preferred spelling |
| `classifier` | classifier id (string) | band test on that classifier's score |
| `label` | string | only with `classifier`; must exist on that classifier; may be omitted only if the classifier has `default_label` |
| `min_score` / `max_score` | number in [0,1], min ≤ max | only with `classifier`; omitting both defaults to `min_score: 0.5` |
| `metadata` | `{ "key": <string>, ... }` | plus exactly one of `equals` (string), `any` (non-empty string array), `exists` (boolean); evaluated over the request's OpenAI `metadata` object. Comparisons are case-sensitive. `any` comma-decodes the metadata value before matching: a caller sending `{"dept": "legal,compliance"}` (scalar) matches `any: ["legal", "compliance"]` - same as if two separate values were in the array. |

The routed input text is the last user message (chat), the `prompt`
(completions), or the `input` (responses).

## Model capability requirements

Checked at registration time - a mismatch fails the `/pull`:

| Role | Needs | Typical models |
|------|-------|----------------|
| candidate / `router.model` / `llm` classifier | chat-capable LLM | `*-GGUF` chat models, cloud models |
| `semantic_similarity` model | embeddings | `nomic-embed-text-*-GGUF` |
| `classifier` model | classification output (onnxruntime encoder) - or a chat LLM (LLM-as-classifier via chat; prefer `type: "llm"`) | `Bert-Phishing-ONNX`, `Phishing-Email-Detection-ONNX` |

Cloud candidates (`fireworks.<id>` etc.) are valid but only exist after the
provider is installed and authenticated - tell the user to run
`lemonade cloud install <provider>` + auth **before** registering the policy.

## Validation checklist

Every generated policy must pass all of these before it is shown to the user.
Items 1–9 are mechanically enforced by `scripts/validate.py` (see `SKILL.md`
Step 8) - run it rather than re-deriving these by eye. Items 10–12 need
authoring judgment and aren't (fully) checkable by a script.

1. Root has exactly `version "1"`, `model_name` (`user.` prefix), `recipe
   "collection.router"`, non-empty `components`, `routing`.
2. `routing` has `candidates` (non-empty, unique) and `default_model` ∈
   `candidates`.
3. Exactly one of `router` / `rules` present; `classifiers` only with `rules`.
4. `components` ⊇ candidates ∪ classifier models ∪ router model.
5. Every rule: unique safe `id`, `route_to` ∈ candidates, non-empty `match`.
6. Every logical node has a single key; every leaf key is in the condition
   table; no invented keys anywhere.
7. Every `classifier` leaf references a declared classifier id; its `label`
   exists on that classifier, or is omitted and the classifier has
   `default_label` (or is label-less).
8. `llm` classifiers have `prompt` + `labels`; `semantic_similarity` have
   `reference_phrases` and NO `labels`; `default_label` ∈ labels/concepts.
9. Scores in [0,1] with min ≤ max; char counts are non-negative integers
   (not a float, not a bool - Python/JSON `true`/`false` are not integers);
   keyword arrays contain only non-empty strings.
10. Privacy/safety rules are ordered before broader routing rules.
11. `model_name` isn't a name already used earlier in this conversation
    (`/pull` is idempotent per `model_name` - a collision silently overwrites
    the earlier registration, not an error).
12. Every `llm` prompt (both `routing.router.prompt` and `type: "llm"`
    classifier `prompt`) describes intent only - it never tells the model how
    to format its reply. The engine appends its own JSON `{model, rationale}`
    contract unconditionally to both. An authored instruction like "reply with
    only the model name" or "reply with exactly one label: X or Y" contradicts
    it: a weaker model obeys the authored line, outputs a bare string, the
    parser rejects it, and the router silently falls through to `default_model`
    on every request with no visible error.

## Registering, invoking, tracing

```bash
# register (idempotent - re-POST to update)
curl -X POST http://localhost:13305/api/v1/pull \
     -H "Content-Type: application/json" --data-binary @router.json

# route: just name the collection as the model
curl -X POST http://localhost:13305/api/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "user.MyHybridRouter", "route_trace": true,
          "messages": [{"role": "user", "content": "My SSN is 123-45-6789"}]}'

# remove
curl -X POST http://localhost:13305/api/v1/delete \
     -H "Content-Type: application/json" -d '{"model_name": "user.MyHybridRouter"}'
```

Every routed response carries the `x-lemonade-route` header (matched rule id
or `default`). With `"route_trace": true` the body also carries
`x_lemonade_route`: `{ version, route_to, matched_rule, default_used, outputs,
trace[] }` - use it to demonstrate each rule to the user. `version` is always
`"1"`. `trace[]` entries include `score` for classifier conditions and omit it
for keyword/metadata conditions. Registration errors come back as descriptive
parser messages (e.g. `routing.default_model 'X' must be listed in
routing.candidates`); fix the field it names and re-POST.

`default_used: true` + `matched_rule: ""` is the only reliable fallback
signal - an empty `rationale` alone is not, since a successful Mode A route to
a non-first candidate commonly returns one too. In Mode A, a non-fallback
`matched_rule` is a synthetic id (`__route_0`, `__route_1`, … - one per
`routing.candidates` entry, in declaration order), not an author-given one;
seeing one of these is expected, not an error. For classifier-backed rules,
also check the per-condition `score` in `trace[]` - stuck at `0.0` across
clearly-matching test inputs signals a mismatched `labels` entry, not a
non-match (see [Classifier entries](#classifier-entries)).

## Desktop-editor compatibility

Users may open the generated policy in the Lemonade desktop app's Hybrid
Router editor (gear icon on the collection). To keep the JSON fully editable
there:

- One condition per leaf object. Compound leaves (`{"min_chars": 5,
  "has_tools": true}`) are valid server-side but the editor cannot display
  them and will warn it must drop them on save.
- `metadata` conditions are JSON-only for now - the editor shows a lossy-edit
  warning for rules containing them. Generate them only when the user
  explicitly wants metadata routing, and say so in your reply.
- Everything else in this reference (nested `all`/`any`/`not`, all three
  classifier types, negated booleans) round-trips cleanly.
