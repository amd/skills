# AMD Skills Walkthroughs: `local-ai-use`

The goal of this skill is to teach your AI agent to use image generation, text generation, and text to speech locally.


## Step 1 - Understanding which skills are available

* Run `claude "Which skills can you see?" --model sonnet`. You should see a list of skills that should not include anythink related to local LLM usage.
* Make sure there is no `AFENTS.md` file on your local folder.

## Step 2 - Enabling claude to see `local-ai-use`

In the future this will be enabled directly through claude's marketplace. For now, we have to manually add it.

* Clone `https://github.com/amd/skills`
* Move the `local-ai-use` skill from the repo to `.claude/skills/`
* Run `claude "Which skills can you see?" --model sonnet`. You should see a list of skills that includes `local-ai-use`.

## Step 3 - Running the skill

Open Claude and run the prompt:
```
Learn how to do image generation locally
```

Followed by
```
Generate the image of a cat
```

Claude should install Lemonade locally on your device and allow you to generate images locally after the first setup run.

## Step 4 - (Optional) Going beyond

The `local-ai-use` can also help you with text to speech and speech to text locally. Simply ask claude for help there.

## Step 5 - (Optional) Try to get things done without AMD Skills

Remove the added skills from `.claude/skills/` and rerun the experiment above. This should lead to a high variance in execution length and token usage.
* Model being successful after significant token usage.
* Model providing a knowledge article instead of actually learning how to do it.
* Model attempting to come up with a custom strategy to generate images locally, resulting in very low-quality assets.