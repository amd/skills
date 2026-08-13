# NL → router policy examples

Each pair shows a user's natural-language request and the exact JSON the skill
should produce. Note what was defaulted: names, ids, thresholds, `on_error`,
`default_label`, and `model_name` all come from the defaults table in
`SKILL.md` when the user doesn't specify them - including `model_name`, which
is derived from the default candidate rather than a fixed literal, so the
examples below each get a distinct name.


## 1. Pure intent, no concrete signals → LLM-as-router

> I want to route my sensitive queries to Gemma-3-4b-it-GGUF and everything
> else to Gemma-4-E4B-it-GGUF

"Sensitive" is meaning, not a mechanical signal → Mode A. The small candidate
doubles as the router model and as `default_model` (fail-safe: on any router
hiccup, requests stay on the model the user trusts with sensitive data).

```json
{
  "version": "1",
  "model_name": "user.Gemma-3-4b-it-GGUF-Router",
  "recipe": "collection.router",
  "components": ["Gemma-3-4b-it-GGUF", "Gemma-4-E4B-it-GGUF"],
  "routing": {
    "candidates": ["Gemma-3-4b-it-GGUF", "Gemma-4-E4B-it-GGUF"],
    "default_model": "Gemma-3-4b-it-GGUF",
    "router": {
      "type": "llm",
      "model": "Gemma-3-4b-it-GGUF",
      "prompt": "You route user requests to the best model. Prompts with sensitive information route to Gemma-3-4b-it-GGUF and everything else to Gemma-4-E4B-it-GGUF."
    }
  }
}
```

Note what's *not* in the prompt: no "reply with only the model name" or any
other reply-format instruction. The engine appends its own strict JSON
contract (`{"model": ..., "rationale": ...}`, listing the exact candidate
names) after whatever the author writes - an authored format instruction
would only contradict it. See `reference.md`'s
[Validation checklist](reference.md#validation-checklist) item 12.

## 2. Concrete signals, no classifiers → deterministic rules

> Send coding questions or anything longer than 4000 characters to
> Qwen3-32B-GGUF, requests with images stay on Qwen3-8B-GGUF, default to
> Qwen3-8B-GGUF

Keywords/length/images are mechanical → Mode B, no classifiers needed. The
images rule is placed first (more specific; keeps image requests local even
when they are long or mention code).

```json
{
  "version": "1",
  "model_name": "user.Qwen3-8B-GGUF-Router",
  "recipe": "collection.router",
  "components": ["Qwen3-8B-GGUF", "Qwen3-32B-GGUF"],
  "routing": {
    "candidates": ["Qwen3-8B-GGUF", "Qwen3-32B-GGUF"],
    "default_model": "Qwen3-8B-GGUF",
    "rules": [
      {
        "id": "rule-1",
        "match": { "has_images": true },
        "route_to": "Qwen3-8B-GGUF",
        "outputs": { "reason": "images-stay-local" }
      },
      {
        "id": "rule-2",
        "match": {
          "any": [
            { "keywords_any": ["def ", "function", "stack trace", "compile"] },
            { "min_chars": 4000 }
          ]
        },
        "route_to": "Qwen3-32B-GGUF",
        "outputs": { "reason": "coding-or-long" }
      }
    ]
  }
}
```

## 3. Classifiers + nested conditions → full rules mode

> Route requests containing personally identifiable information (PII), such as
> Social Security numbers or email addresses, to Gemma-3-4b-it-GGUF. Detect
> PII using both classification and pattern matching where appropriate. Also
> send requests that involve tools or images together with PII to the same
> model. Use Bert-Phishing-ONNX to identify PII and jailbreak attempts. Use
> nomic-embed-text-v2-moe-GGUF to recognize shopping- and apparel-related
> requests through semantic similarity. For requests related to clothing or
> apparel that include images, use the semantic classifier together with
> additional context, such as an LLM safety assessment or the length of the
> request, before routing. Use Gemma-3-4b-it-GGUF as the LLM classifier for
> SAFE and RISKY classifications. If none of the routing rules apply, or if a
> classifier fails, fall back to Gemma-3-4b-it-GGUF.

All three classifier types, nested `any` inside `all`, an SSN regex, and
"classifier fails → fall back" mapping to `on_error: match_false` +
`default_model`. Components include the classifier models even though they
never answer requests.

```json
{
  "version": "1",
  "model_name": "user.Gemma-3-4b-it-GGUF-PII-Router",
  "recipe": "collection.router",
  "components": [
    "Gemma-3-4b-it-GGUF",
    "Gemma-4-E4B-it-GGUF",
    "Bert-Phishing-ONNX",
    "nomic-embed-text-v2-moe-GGUF"
  ],
  "routing": {
    "candidates": ["Gemma-3-4b-it-GGUF", "Gemma-4-E4B-it-GGUF"],
    "default_model": "Gemma-3-4b-it-GGUF",
    "classifiers": [
      {
        "id": "clf-1",
        "type": "classifier",
        "model": "Bert-Phishing-ONNX",
        "labels": ["PII", "Jailbreak"],
        "default_label": "PII",
        "on_error": "match_false"
      },
      {
        "id": "clf-2",
        "type": "semantic_similarity",
        "model": "nomic-embed-text-v2-moe-GGUF",
        "reference_phrases": {
          "shopping": [
            "I want to shop for pants",
            "find me a jacket in medium",
            "add these shoes to my cart"
          ]
        },
        "default_label": "shopping",
        "on_error": "match_false"
      },
      {
        "id": "clf-3",
        "type": "llm",
        "model": "Gemma-3-4b-it-GGUF",
        "prompt": "Classify the request into only labels SAFE, RISKY",
        "labels": ["SAFE", "RISKY"],
        "default_label": "SAFE",
        "on_error": "match_false"
      }
    ],
    "rules": [
      {
        "id": "rule-1",
        "match": {
          "all": [
            { "classifier": "clf-1", "min_score": 0.5 },
            { "keywords_any": ["SSN", "Email"] },
            { "has_tools": true },
            {
              "any": [
                { "regex": "\\b\\d{3}-?\\d{2}-?\\d{4}\\b" },
                { "has_images": true }
              ]
            }
          ]
        },
        "route_to": "Gemma-3-4b-it-GGUF"
      },
      {
        "id": "rule-2",
        "match": {
          "all": [
            { "classifier": "clf-2", "min_score": 0.5 },
            { "keywords_any": ["apparel", "clothes", "clothing"] },
            { "has_images": true },
            {
              "any": [
                { "classifier": "clf-3", "min_score": 0.5 },
                { "min_chars": 500 }
              ]
            }
          ]
        },
        "route_to": "Gemma-3-4b-it-GGUF"
      }
    ]
  }
}
```

## 4. Negation and metadata opt-out

> Anything without tool calls goes to Phi-4-mini-GGUF; if the request metadata
> has consent=denied it must stay on Phi-4-mini-GGUF no matter what; the rest
> to Qwen3-32B-GGUF

```json
{
  "version": "1",
  "model_name": "user.Phi-4-mini-GGUF-Router",
  "recipe": "collection.router",
  "components": ["Phi-4-mini-GGUF", "Qwen3-32B-GGUF"],
  "routing": {
    "candidates": ["Phi-4-mini-GGUF", "Qwen3-32B-GGUF"],
    "default_model": "Qwen3-32B-GGUF",
    "rules": [
      {
        "id": "rule-1",
        "match": { "metadata": { "key": "consent", "equals": "denied" } },
        "route_to": "Phi-4-mini-GGUF",
        "outputs": { "reason": "privacy" }
      },
      {
        "id": "rule-2",
        "match": { "not": { "has_tools": true } },
        "route_to": "Phi-4-mini-GGUF"
      }
    ]
  }
}
```

Tell the user: the metadata rule is honored by the server but is not yet
editable in the desktop Hybrid Router editor (it will warn about a lossy edit
if they open this policy there).

---

## 5. HR chatbot - LLM-as-router with a privacy-first prompt

> Build an HR assistant router. Any request with PII - names with salaries,
> SSNs, bank accounts, equity details, dates of birth - must stay on the local
> model. Everything else can go to the cloud model. When in doubt, keep it
> local.

"PII" is a meaning judgment, not a regex - the boundary is fuzzy and context-
dependent (a name alone is fine; a name + salary is not). That ambiguity is
exactly what Mode A is designed for: a small LLM reads each request and
decides. The router model doubles as `default_model` so a router hiccup never
leaks to the cloud.

The prompt describes *when* to pick each candidate. It does **not** tell the
model how to format its reply - the engine appends its own `{"model": ...,
"rationale": ...}` contract.

```json
{
  "version": "1",
  "model_name": "user.HR-Admin-Router",
  "recipe": "collection.router",
  "components": ["Qwen3.5-9B-GGUF", "fireworks.kimi-k2p6"],
  "routing": {
    "candidates": ["Qwen3.5-9B-GGUF", "fireworks.kimi-k2p6"],
    "default_model": "Qwen3.5-9B-GGUF",
    "router": {
      "type": "llm",
      "model": "Qwen3.5-9B-GGUF",
      "prompt": "You are a routing assistant for an AI company. Your job is to choose which model should handle each request.\n\nUse Qwen3.5-9B-GGUF (local, private) when:\n- The request contains personally identifiable information (PII), such as names with salaries, Social Security numbers (SSNs), bank account numbers, email addresses, compensation data, equity details, or dates of birth.\n- Data privacy is paramount - anything that should never leave the local machine.\n\nUse fireworks.kimi-k2p6 (cloud, powerful) for all other requests.\n\nIf the request is ambiguous, default to Qwen3.5-9B-GGUF. When in doubt, prioritize privacy over capability."
    }
  }
}
```

**When to prefer Mode A over regex for PII**: a regex catches `123-45-6789`
but misses "Alice's salary is sixty thousand". If the domain has natural-
language PII, Mode A catches it; if the domain has structured PII (form
inputs, database exports), regex is more reliable and cheaper (no extra LLM
call per request). Use both together when both forms appear - put regex rules
first (cheaper, no latency), then fall through to an LLM router or classifier
for the fuzzy residual.

---

## 6. Benefits chatbot - three-tier routing with max_chars and rich outputs

> Route a benefits chatbot. Any request containing PII (email, salary, SSN,
> equity, compensation, bank account, date of birth) must stay local on
> `Qwen3.5-9B-GGUF`. Complex analysis (comparison, benchmarking,
> optimization, or anything longer than 800 characters) goes to
> `fireworks.kimi-k2p6`. Short, simple benefit lookups (401k, PTO, healthcare,
> open enrollment, etc. under 400 characters) also stay local. Default to
> cloud for anything else.

Three tiers: **PII fence** first (privacy-critical, `on_error: match_true`
would be appropriate but this config uses the default fail-open for
simplicity), **complexity escalation** second, **domain fast-path** third.
`outputs` carries two fields (`reason` + `data_class`/`tier`) - useful for
downstream logging and observability.

New patterns not shown in examples 1–4:
- `max_chars` to cap a rule to *short* prompts only
- An `all` combining `max_chars` + `keywords_any` (domain keyword on a short
  prompt = cheap local RAG; same keyword on a long prompt falls through to
  cloud)
- Multiple key/value pairs in `outputs` for structured tagging

```json
{
  "version": "1",
  "model_name": "user.Benefits-Router",
  "recipe": "collection.router",
  "components": ["Qwen3.5-9B-GGUF", "fireworks.kimi-k2p6"],
  "routing": {
    "candidates": ["Qwen3.5-9B-GGUF", "fireworks.kimi-k2p6"],
    "default_model": "fireworks.kimi-k2p6",
    "rules": [
      {
        "id": "pii-stays-local",
        "match": {
          "any": [
            { "regex": "[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}" },
            { "regex": "\\$[0-9,]+(K|k|M|m)?\\b" },
            { "regex": "\\b[0-9]{3}-[0-9]{2}-[0-9]{4}\\b" },
            { "keywords_any": ["salary", "equity", "ssn", "social security", "bank account", "date of birth", "compensation"] }
          ]
        },
        "route_to": "Qwen3.5-9B-GGUF",
        "outputs": { "reason": "pii-detected", "data_class": "sensitive" }
      },
      {
        "id": "complex-benefits-analysis",
        "match": {
          "any": [
            { "keywords_any": ["compare", "benchmark", "analyze", "optimize", "recommend", "industry standard", "strategy"] },
            { "min_chars": 800 }
          ]
        },
        "route_to": "fireworks.kimi-k2p6",
        "outputs": { "reason": "complex-benefits-analysis", "tier": "cloud" }
      },
      {
        "id": "simple-benefits-rag",
        "match": {
          "all": [
            { "max_chars": 400 },
            { "keywords_any": ["401k", "401(k)", "vesting", "parental leave", "PTO", "vacation", "healthcare", "dental", "vision", "FSA", "HSA", "COBRA", "open enrollment", "qualifying life event", "copay", "deductible", "premium", "handbook", "policy", "onboard"] }
          ]
        },
        "route_to": "Qwen3.5-9B-GGUF",
        "outputs": { "reason": "simple-benefits-rag", "tier": "local-fast" }
      }
    ]
  }
}
```

**Rule ordering matters here**: `pii-stays-local` fires first. A message like
"what's the copay on my plan - my salary is $95K" would match both rule 1
(salary keyword) and rule 3 (copay + short). Because rule 1 comes first, it
routes local - correct. Reordering would leak PII to the cloud.

---

## 7. Finance chatbot - semantic similarity + LLM classifier in tandem

> Build a finance assistant router for a startup. PII-flavored finance queries
> (individual salaries, equity by person, payroll details) stay on the local
> model. Deep modeling requests (Monte Carlo, cap table, cohort forecasting,
> sensitivity analysis) go to the cloud. Simple metric lookups (burn rate, MRR,
> runway, ARR) stay local. As a fallback, use a local LLM to judge whether a
> request is COMPLEX or SIMPLE, routing COMPLEX to cloud. Default to local.

The most advanced pattern: **two classifiers working in tandem** - a fast
embedding classifier (`semantic_similarity`) fires first to catch known
patterns, then an LLM classifier acts as a catch-all judge for requests that
didn't match any semantic bucket. Four rules, two classifier types, three
score thresholds tuned independently per rule.

What's unique here not shown elsewhere:
- Two classifiers declared; rules reference each independently
- `semantic_similarity` with three distinct concept buckets (not just two)
- An `llm` classifier used as a safety net *after* semantic matching, not as
  the primary signal - keeps latency low for the common case
- Per-rule score thresholds tuned to the domain (0.65 for PII, 0.72 for deep
  modeling, 0.60 for simple lookups) rather than the default 0.5
- `outputs` carries a `classifier` field to tell downstream code *which*
  classifier fired

```json
{
  "version": "1",
  "model_name": "user.Finance-Router",
  "recipe": "collection.router",
  "components": [
    "fireworks.kimi-k2p6",
    "Qwen3.5-9B-GGUF",
    "embeddinggemma-300m-qat-q8_0-GGUF-Q8_0"
  ],
  "routing": {
    "candidates": ["Qwen3.5-9B-GGUF", "fireworks.kimi-k2p6"],
    "default_model": "Qwen3.5-9B-GGUF",
    "classifiers": [
      {
        "id": "finance-topic",
        "type": "semantic_similarity",
        "model": "embeddinggemma-300m-qat-q8_0-GGUF-Q8_0",
        "reference_phrases": {
          "pii-finance": [
            "show me the salary breakdown by employee",
            "what is the equity package for this specific person",
            "list all compensation by individual",
            "employee bank account for payroll",
            "individual 401k contribution amounts",
            "personal compensation details for staff member"
          ],
          "simple-lookup": [
            "what is our current burn rate",
            "what is the MRR this month",
            "how much runway do we have left",
            "what is the current ARR",
            "what is today's headcount",
            "how much did we spend last quarter"
          ],
          "deep-modeling": [
            "run a Monte Carlo simulation on our runway",
            "model the cap table after the next funding round",
            "forecast revenue using cohort analysis",
            "calculate waterfall distribution for an exit scenario",
            "build a sensitivity analysis for burn rate assumptions",
            "recommend an intervention strategy for high churn and retention",
            "analyze cap table dilution under different term sheet scenarios"
          ]
        }
      },
      {
        "id": "complexity",
        "type": "llm",
        "model": "Qwen3.5-9B-GGUF",
        "prompt": "You are a financial complexity classifier. Assess whether this finance request requires deep multi-step modeling and cloud-level reasoning, or is a simple metric lookup that a local model can handle.\n\nClassify as COMPLEX if the request involves: statistical modeling, simulations, multi-variable forecasting, cohort analysis, cap table calculations, or synthesizing multiple data sources.\n\nClassify as SIMPLE if the request is a direct data lookup, a single metric question, or a short factual query.",
        "labels": ["COMPLEX", "SIMPLE"],
        "default_label": "SIMPLE",
        "on_error": "match_false"
      }
    ],
    "rules": [
      {
        "id": "pii-finance-semantic",
        "match": { "classifier": "finance-topic", "label": "pii-finance", "min_score": 0.65 },
        "route_to": "Qwen3.5-9B-GGUF",
        "outputs": { "reason": "pii-semantic-finance", "data_class": "sensitive", "classifier": "semantic_similarity" }
      },
      {
        "id": "deep-model-semantic",
        "match": { "classifier": "finance-topic", "label": "deep-modeling", "min_score": 0.72 },
        "route_to": "fireworks.kimi-k2p6",
        "outputs": { "reason": "deep-finance-model-semantic", "tier": "cloud", "classifier": "semantic_similarity" }
      },
      {
        "id": "deep-model-llm-judge",
        "match": { "classifier": "complexity", "label": "COMPLEX", "min_score": 0.5 },
        "route_to": "fireworks.kimi-k2p6",
        "outputs": { "reason": "llm-judged-complex-finance", "tier": "cloud", "classifier": "llm" }
      },
      {
        "id": "simple-metric-lookup",
        "match": { "classifier": "finance-topic", "label": "simple-lookup", "min_score": 0.60 },
        "route_to": "Qwen3.5-9B-GGUF",
        "outputs": { "reason": "simple-metric-lookup", "tier": "local-fast", "classifier": "semantic_similarity" }
      }
    ]
  }
}
```

**Tuning guidance for `reference_phrases`**: aim for 5–8 varied phrases per
concept covering different vocabulary, register, and specificity. Phrases that
are too similar to each other (all formal, all the same length) produce a
tight cluster that fails on paraphrases. Include at least one informal and one
technical phrasing per concept.
