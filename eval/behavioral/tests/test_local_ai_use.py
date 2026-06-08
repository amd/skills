"""Behavioral tests for the `local-ai-use` skill.

Run locally (needs the `claude` CLI authenticated and a reachable Lemonade
Server -- otherwise the suite skips):

    pytest eval/behavioral/tests/test_local_ai_use.py -s

Each `should` / `should_not` is graded individually; the conftest finalizer
prints a per-item pass/fail table, writes a results JSON under
`eval/behavioral/results/`, and fails the test if any item failed.
"""

from harness import (
    agents_md_contains,
    file_is_png,
    model_downloaded,
    model_newly_downloaded,
    prompt,
    tool_used,
    transcript_matches,
)


def test_generate_image_of_a_cat():
    run = prompt("Use local AI in this workspace, then generate an image of a cat and save it to out.png.")

    # Should: an image model is available, and the image was produced locally.
    run.should(model_downloaded("SD-Turbo"))
    run.should(file_is_png("out.png"))
    run.should(agents_md_contains("amd-skills:local-ai-use"))
    run.should(transcript_matches(r"(localhost|127\.0\.0\.1):13305/api/v1/images/generations"))

    # Should not: pull unrelated modalities for an image-only task.
    run.should_not(model_newly_downloaded("kokoro-v1"))
    run.should_not(model_newly_downloaded("Whisper-Tiny"))

    # Should not: reach for a cloud image path instead of local Lemonade.
    run.should_not(tool_used("GenerateImage"))
    run.should_not(transcript_matches(r"openai\.com|dall-?e|midjourney|stability\.ai"))
