# AMD Skills walkthroughs: `local-ai-use`

The goal of this skill is to teach your AI agent to use image generation, text generation, and text to speech locally.

## Prerequisites

- Claude Code installed and authenticated.

## Step 1: Check which skills are available

Confirm that local AI skills are not yet visible to your agent.

* Run `claude "Which skills can you see?" --model sonnet`. You should see a list of skills that should not include anything related to local large language model (LLM) usage.
* Make sure there is no `AGENTS.md` file on your local folder.

## Step 2: Enable Claude to see `local-ai-use`

* Install the skill with the [`skills` CLI](https://github.com/vercel-labs/skills). Run this in your terminal, not inside Claude:

```bash
npx skills add amd/skills --skill local-ai-use --agent claude-code
```

* Run `claude "Which skills can you see?" --model sonnet`. You should see a list of skills that includes `local-ai-use`.

## Step 3: Run the skill

Open Claude and run the prompt:
```
Learn how to do image generation locally
```

Followed by
```
Generate the image of a cat
```

Claude should install Lemonade locally on your device and allow you to generate images locally after the first setup run.

## Step 4: (Optional) Go beyond

The `local-ai-use` skill can also help you with text to speech and speech to text locally. Ask Claude for help there.

## Step 5: (Optional) Try to get things done without AMD Skills

Remove the added skills from `.claude/skills/` and rerun the experiment above. This should lead to a high variance in execution length and token usage.

Common outcomes without the skill include:

* Model being successful after significant token usage.
* Model providing a knowledge article instead of actually learning how to do it.
* Model attempting to come up with a custom strategy to generate images locally, resulting in very low-quality assets.
