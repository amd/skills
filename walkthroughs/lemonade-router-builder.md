# AMD Skills walkthroughs: `lemonade-router-builder`

The goal of this skill is to teach your AI agent to turn a plain-English
description of routing intent into a valid Lemonade `collection.router` policy
JSON, ready for you to register and use.
See the [Lemonade Router policy documentation](https://lemonade-server.ai/docs/dev/router-policy/).

## Prerequisites

Before you start, confirm the following.

- Claude Code installed and authenticated.

## Step 1: Install and start Lemonade Server

Install Lemonade Server **v11.5.0 or later** from
<https://lemonade-server.ai/docs/guide/install/>, and download
at least two chat-capable models:

```bash
lemonade status
lemonade list
```

Note the exact model names shown by `lemonade list`. You will use them in the prompts below. The examples use `<LARGE_MODEL>`, `<CLOUD_MODEL>`, and `<SMALL_MODEL>` as placeholders; substitute the names of your two installed models (for example, Gemma-4-31B-it-GGUF for large, Gemma-3-4b-it-GGUF for small, and kimi-k2p6 for cloud).

## Step 2: Confirm the skill is visible

```bash
claude "Which skills can you see?" --model sonnet
```

You should see `lemonade-router-builder` in the list. If not, install it from your terminal, not inside Claude:

```bash
npx skills add amd/skills --skill lemonade-router-builder --agent claude-code
```

## Step 3: Generate a keyword router

Open Claude and run:

```
Route coding questions - anything mentioning functions, bugs, or stack traces -
to <LARGE_MODEL>. Everything else goes to <SMALL_MODEL>.

Example: Route coding questions - anything mentioning functions, bugs, or stack traces - to Gemma-4-31B-it-GGUF. Everything else goes to Gemma-3-4b-it-GGUF.
```

The agent should:
1. Ask which models you want if you haven't specified them (or proceed if you
   have).
2. Generate a `collection.router` JSON with a `keywords_any` rule.
3. Run `scripts/validate.py` against the JSON and confirm it passes.
4. Output the JSON in a fenced block, plus curl commands for you to register
   and test it.

## Step 4: Register and test the router yourself

The agent will save the policy as `router.json`. Note the full path it reports,
then run the curl commands **from the same directory** (or use the absolute path).
The `@` prefix in `--data-binary` tells curl to read the body from a file. It
resolves relative to your current working directory, not where the file was saved.

```bash
# Register (replace /path/to/ with wherever the agent saved router.json)
curl -X POST http://localhost:13305/api/v1/pull \
     -H "Content-Type: application/json" --data-binary @/path/to/router.json

# Test. Claude should give you the exact command; if not, adapt this one (update <RouterName> and "messages")
curl -i -X POST http://localhost:13305/api/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "user.<RouterName>", "route_trace": true,
          "messages": [{"role": "user", "content": "How do I fix this bug?"}]}'
```

Check the `x-lemonade-route` response header. It should show the matched rule
id. With `"route_trace": true` the body also contains `x_lemonade_route` with
the full per-condition trace.

## Step 5: Try a personally identifiable information (PII) privacy router

On a new Claude session, run:

```
Any message containing a Social Security number or email address must stay on <SMALL_MODEL>. Everything else can go to <CLOUD_MODEL>.

Example: Any message containing a Social Security number or email address must stay on Gemma-3-4b-it-GGUF. Everything else can go to fireworks.kimi-k2p6.
```

The agent should produce a rules-mode policy with two `regex` conditions and
place the PII rule first. Validate by sending a test message with a fake SSN
(`123-45-6789`) and confirming the header shows that rule's id. Expect the
default `rule-1` unless you asked the agent to name it something else.

If Claude refuses to send a test message containing a fake SSN, drop the SSN
condition and test with the email address only instead:

```
Any message containing an email address must stay on <SMALL_MODEL>. Everything else can go to <CLOUD_MODEL>.
```

## Step 6: Try a large language model (LLM) as router (intent-only)

On a new Claude session, run:

```
I want sensitive queries to stay on <LARGE_MODEL> and everything else to go to <CLOUD_MODEL>. Use the local model as the router.

Example: I want sensitive queries to stay on Gemma-4-31B-it-GGUF and everything else to go to fireworks.kimi-k2p6. Use the local model as the router.
```

The agent should choose Mode A (`routing.router` block) rather than rules,
because "sensitive" is a meaning judgment with no concrete signal. The
generated prompt should describe routing intent only, with no reply-format
instructions.

## Step 7: (Optional) Try to get things done without the skill

Remove the skill and ask the same routing questions. Without the skill, the
agent is likely to produce JSON that fails the server-side parser on the first
try, get the mode exclusivity wrong (`router` + `rules` together), or omit
required fields like `components`. This demonstrates the value of the skill's
structured generation and offline validation step.
