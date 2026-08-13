# AMD Skills Walkthroughs: `lemonade-router-builder`

The goal of this skill is to teach your AI agent to turn a plain-English
description of routing intent into a valid Lemonade `collection.router` policy
JSON, ready for you to register and use.

## Prerequisites

- Claude Code installed and authenticated.

## Step 1 - Install and start Lemonade Server

Install Lemonade Server from <https://lemonade-server.ai/docs/guide/install/>,
then start it and download at least two chat-capable models:

```bash
lemonade server start
lemonade list
```

## Step 2 - Confirm the skill is visible

```bash
claude "Which skills can you see?" --model sonnet
```

You should see `lemonade-router-builder` in the list. If not, install it from your terminal, not inside Claude:

```bash
npx skills add amd/skills --skill lemonade-router-builder --agent claude-code
```

## Step 3 - Generate a simple keyword router

Open Claude and run:

```
Route coding questions - anything mentioning functions, bugs, or stack traces -
to Qwen3.5-9B-GGUF. Everything else goes to Qwen3.5-2B-GGUF.
```

The agent should:
1. Ask which models you want if you haven't specified them (or proceed if you
   have).
2. Generate a `collection.router` JSON with a `keywords_any` rule.
3. Run `scripts/validate.py` against the JSON and confirm it passes.
4. Output the JSON in a fenced block, plus curl commands for you to register
   and test it.

## Step 4 - Register and test the router yourself

Copy the curl commands the agent produced and run them in your terminal:

```bash
# Register
curl -X POST http://localhost:13305/api/v1/pull \
     -H "Content-Type: application/json" --data-binary @router.json

# Test - should match the keyword rule
curl -X POST http://localhost:13305/api/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "user.<RouterName>", "route_trace": true,
          "messages": [{"role": "user", "content": "How do I fix this bug?"}]}'
```

Check the `x-lemonade-route` response header - it should show the matched rule
id. With `"route_trace": true` the body also contains `x_lemonade_route` with
the full per-condition trace.

## Step 5 - Try a PII privacy router

```
Any message containing a Social Security number or email address must stay on
Qwen3.5-9B-GGUF. Everything else can go to Qwen3.5-9B-NoThinking.
```

The agent should produce a rules-mode policy with two `regex` conditions and
place the PII rule first. Validate by sending a test message with a fake SSN
(`123-45-6789`) and confirming the header shows `pii-stays-local`.

## Step 6 - Try an LLM-as-router (intent-only)

```
I want sensitive queries to stay on Qwen3.5-9B-GGUF and everything else to go
to Qwen3.5-9B-NoThinking. Use the local model as the router.
```

The agent should choose Mode A (`routing.router` block) rather than rules,
because "sensitive" is a meaning judgment with no concrete signal. The
generated prompt should describe routing intent only - no reply-format
instructions.

## Step 7 - (Optional) Try to get things done without the skill

Remove the skill and ask the same routing questions. Without the skill, the
agent is likely to produce JSON that fails the server-side parser on the first
try, get the mode exclusivity wrong (`router` + `rules` together), or omit
required fields like `components`. This demonstrates the value of the skill's
structured generation and offline validation step.
