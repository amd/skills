# Lemonade Router Config: Reference

Detailed contract for the `lemonade-router-builder` skill. Read this when the
generation steps in `SKILL.md` leave a question open. The server-side parser
(`routing_policy_parser.cpp`) is strict: it rejects unknown keys at every
level, so never emit fields not listed here.

The machine-readable source of truth is `scripts/route_policy.schema.json`, a
verbatim copy of the server's
[`route_policy.schema.json`](https://github.com/lemonade-sdk/lemonade/blob/main/src/cpp/resources/schemas/route_policy.schema.json);
`scripts/route_policy.schema.source.json` records the ref it was copied from.
`scripts/validate.py` derives its key sets from it, so the validator follows
the schema automatically after `python scripts/sync_schema.py`. The tables
below are hand-written. When a sync reports new keys or types, update them too.
`python scripts/sync_schema.py --check` exits 1 when the copy is stale.

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
`reference_phrases`, `question`, `on_error`, `on_truncation`.

| Field | `classifier` | `zero_shot` | `systemone` | `semantic_similarity` | `llm` |
|-------|--------------|-------------|-------------|----------------------|-------|
| `id` | required, unique | required, unique | required, unique | required, unique | required, unique |
| `model` | required (fixed-label classifier) | required (zero-shot router or decision model) | required (decision model) | required (embedding model) | required (chat LLM) |
| `prompt` | - | **rejected** | **rejected** | - | **required**, non-empty |
| `labels` | optional (must match the model's real output labels) | **required**, ≥ 1; sent to the model per request | **rejected** (the question's options are the labels) | **rejected** (concepts are the labels) | **required**, ≥ 1 entry |
| `reference_phrases` | - | **rejected** | **rejected** | **required**: `{concept: [phrase, ...]}`, ≥1 concept, ≥1 phrase each | - |
| `question` | rejected | rejected | **required**, see below | rejected | rejected |
| `default_label` | optional; must be in `labels`; requires `labels` non-empty | optional; in `labels` | optional; one of the question's options | optional; must be a concept name | optional; must be in `labels` |
| `on_error` | `"match_false"` (default) or `"match_true"` | same | same | same | same |
| `on_truncation` | `"score"` (default) or `"fail"` | same | same | **rejected** | **rejected** |

Every `labels` list (all types) must be unique, with no blank entries and no
control characters - the same rules `POST /v1/classify` applies. How many
labels a `zero_shot` model accepts is the model's token window's call, not a
policy limit; the model names the number in its error.

### `systemone` question

```json
"question": {
  "type": "choice" | "noul" | "score",
  "instructions": "<non-blank text, or a non-empty object/array>",
  "criteria": ...
}
```

Only those three keys. What `criteria` is, and which labels it yields:

| `type` | `criteria` | Labels (what rules test) | Score per label |
|---|---|---|---|
| `choice` | **array**, ≥ 2 distinct options, read in order: `["billing", {"shipping": "where an order is"}]` - each entry an option name or a one-key `{option: description}` (description = text, object, array, or null) | the option names | probability of that option |
| `noul` | optional `{"true": <desc>, "false": <desc>}` | `"true"`, `"false"` | `true` = p(yes), `false` = 1 − p(yes) |
| `score` | array of 2–10 distinct non-blank level strings, lowest first | the level strings | probability of that level |

- `choice` criteria as an **object is refused**: a stored policy sorts object
  keys, and option order changes the scores. Note the asymmetry: the raw
  `POST /v1/systemone` endpoint takes `choice` criteria as an object
  `{"billing": null, ...}` - convert when moving between the two.
- A `score` rule tests one level's probability. "At least *today*" is
  `{"any": [{"classifier": "u", "label": "today"}, {"classifier": "u", "label": "right now"}]}`
  (approximate - each level is tested separately), or ask a `noul` question.
- Answers depend on the model, the instruction wording, and option order -
  test with `/v1/systemone` before relying on a threshold.

### Truncation (`on_truncation`)

`classifier`, `zero_shot` and `systemone` read a token window. When the input
overflows it, the backend cuts the text (never the labels) and still returns
scores - a success, so `on_error` never fires. What the cut loses:

| Type / backend | Keeps | Loses |
|---|---|---|
| `zero_shot` on ort-server (`LFM2.5-Encoder-350M-Prompt-Router-ONNX`, 2,048-token default ≈ 8,000 chars) | head + tail | the middle |
| `classifier` (onnx encoder) | the head | **the end** |
| `systemone` / `zero_shot` on a decision model (Laya, Julia-1, `vllm-d3`, `d3-*-FLM`) | refuses an over-long text instead (4,096 tokens by default for encoder decision models) → that's an error, `on_error` applies | - |

`"on_truncation": "fail"` turns a reported cut - or a backend that cannot
report cuts (ort-server < 0.4.0, `lfm2-d1-omni`) - into a failure, so
`on_error` decides. For a privacy rule pair it with `"on_error": "match_true"`;
that protects the request only if the rule routes to a **local** model, sits
**above every cloud rule**, and the sensitive data is in the **last user
message** (the routing input). A `classifier` backed by a chat LLM refuses
`on_truncation` at `/pull` (a chat model reports no cut).

Raise the window server-side when needed: `--onnxruntime-args "--max-length 4096"`
(ONNX zero-shot; memory grows with the square), `--llamacpp-args "-b 8192 -ub 8192"`
(encoder decision models).

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
| `min_chars` / `max_chars` | integer ≥ 0 | length of the routing input (latest user turn) in **UTF-8 bytes** |
| `min_total_chars` / `max_total_chars` | integer ≥ 0 | length of **all** text in the request in UTF-8 bytes: every chat message of every role (system/user/assistant/tool), the `/completions` `prompt`, or every Responses `input` item. Images/audio count as 0. Fires on a long history even when the last turn is `"go on"`, which `min_chars` sees as 5 bytes. Estimate ~4 bytes per token when sizing against a context window. Server v11.8.0+ |
| `has_tools` / `has_images` | boolean | request carries `tools[]` / image parts. `false` is legal and means "must NOT have"; `{"not": {"has_tools": true}}` is the equivalent preferred spelling |
| `classifier` | classifier id (string) | band test on that classifier's score |
| `label` | string | only with `classifier`; must exist on that classifier; may be omitted only if the classifier has `default_label` |
| `min_score` / `max_score` | number in [0,1], min ≤ max | only with `classifier`; omitting both defaults to `min_score: 0.5` |
| `metadata` | `{ "key": <string>, ... }` | plus exactly one of `equals` (string), `any` (non-empty string array), `exists` (boolean); evaluated over the request's OpenAI `metadata` object. Comparisons are case-sensitive. `any` comma-decodes the metadata value before matching: a caller sending `{"dept": "legal,compliance"}` (scalar) matches `any: ["legal", "compliance"]` - same as if two separate values were in the array. |

The routed input text is the last user message (`/chat/completions`), the
`prompt` (`/completions`), or the `input` (`/responses`). All three endpoints
route a `collection.router` model on every supported server (v11.5.0+).

## Model capability requirements

Checked at registration time - a mismatch fails the `/pull`:

| Role | Needs | Typical models |
|------|-------|----------------|
| candidate / `router.model` / `llm` classifier | chat-capable LLM | `*-GGUF` chat models, cloud models |
| `semantic_similarity` model | embeddings | `nomic-embed-text-*-GGUF` |
| `classifier` model | classification output (onnxruntime encoder) - or a chat LLM (LLM-as-classifier via chat; prefer `type: "llm"`). **Refused**: a `zero-shot` model or a decision model | `Bert-Phishing-ONNX`, `Phishing-Email-Detection-ONNX` |
| `zero_shot` model | registry label `zero-shot` (onnxruntime), or a decision model (labels `classification` + `systemone`). No chat-LLM fallback | `LFM2.5-Encoder-350M-Prompt-Router-ONNX`; any decision model below |
| `systemone` model | decision model: labels `classification` + `systemone` on `llamacpp` or `flm`. An ONNX classifier is refused, as is a GGUF whose decision type the bundled llama.cpp can't load | `Laya-GGUF` (421M), `Julia-1-GGUF` (144M) on CPU/GPU; `d3-lite-FLM` (0.8B), `d3-nano-FLM` (2B), `d3-mini-FLM` (4B), `d3-turbo-FLM` / `d3-ultra-FLM` (27B) on the NPU; user-added `ggml-org/vllm-d3-*-GGUF` (pull with `"labels": ["classification", "systemone"]`) |

Decision models answer `/v1/systemone` (typed questions) and `/v1/classify`
with `labels` (zero-shot: one `choice` question whose options are the labels)
in one forward pass, without generating tokens - so one decision model can
back both a `systemone` and a `zero_shot` classifier in the same policy.

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
   `reference_phrases` and NO `labels`; `zero_shot` have `labels` and no
   `prompt`/`reference_phrases`; `systemone` have a valid `question` (choice
   criteria an array of ≥ 2, score 2–10 levels, noul only `true`/`false`) and
   no `labels`/`prompt`/`reference_phrases`; `question` only on `systemone`,
   `on_truncation` only on `classifier`/`zero_shot`/`systemone`; labels
   unique with no blanks/control characters; `default_label` ∈
   labels/concepts/options.
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
curl -X POST http://localhost:13305/api/v1/routing/validate \
     -H "Content-Type: application/json" \
     -d '{"policy": {...}, "prompt": "My SSN is 123-45-6789",
          "has_tools": false, "has_images": false, "metadata": {"consent": "denied"}}'

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
for keyword/metadata conditions. A `classifier`, `zero_shot` or `systemone`
entry also carries `error` (the backend's reason) when the condition failed -
without it, a failure and a low score both read `"result": false` - and
`truncated` (`true` = the input was cut to fit the token window, `false` =
scored whole, absent = the backend can't tell). Registration errors come back as descriptive
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
- `metadata` and `min_total_chars` / `max_total_chars` conditions are
  JSON-only in released builds - the editor shows a lossy-edit warning for
  rules containing them. (The editor on Lemonade `main` edits `metadata` but
  refuses to import `*_total_chars`.) Generate them only when the user's
  intent needs them, and say so in your reply.
- `zero_shot` and `systemone` classifiers and the `on_truncation` field are
  **not importable** in the editor (it refuses the whole policy: "The Lemonade
  App can safely edit classifier, semantic_similarity, and llm classifiers
  only"). Policies using them are JSON-only - register with `/pull` and edit
  the file. Say so in your reply when you generate one.
- Everything else in this reference (nested `all`/`any`/`not`, the
  `classifier` / `semantic_similarity` / `llm` types, negated booleans)
  round-trips cleanly.
