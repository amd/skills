# AMD Skills

AMD Skills package the knowledge, scripts, and conventions needed to be productive with AMD hardware and software — ROCm, HIP, MIGraphX, ROCm-aware PyTorch and JAX, Instinct GPUs, Ryzen AI, and the broader AMD developer stack — and deliver them in a form that AI coding agents can load on demand.

The goal: when a developer asks an agent to "set up ROCm in this container," "port this CUDA kernel to HIP," or "tune this model for an MI300X," the agent can pull in an authoritative AMD-authored skill instead of guessing.

Skills in this repository follow the standardized [Agent Skills](https://github.com/anthropics/skills) format and are designed to interoperate with the major coding agents, including Cursor, Claude Code, OpenAI Codex, and Gemini CLI.

## What is a skill?

A skill is a self-contained folder that bundles everything an agent needs to perform a focused task: instructions, helper scripts, prompts, templates, and references. At its core is a `SKILL.md` file with YAML frontmatter — a `name` and a short `description` that tells the agent *when* the skill should activate — followed by the guidance the agent reads while the skill is in use.

```
skills/
  rocm-setup/
    SKILL.md
    scripts/
    references/
```

When an agent decides a skill is relevant (or you invoke it explicitly), it loads that `SKILL.md` and follows the instructions inside.

> If your agent of choice does not yet support the Agent Skills standard, you can fall back to the consolidated `agents/AGENTS.md` bundle that this repo will publish.

## Why AMD Skills?

Working effectively with the AMD stack often means knowing:

- Which ROCm version pairs with which kernel, distro, and PyTorch build.
- The right HIP idioms when porting CUDA, and where the abstractions diverge.
- How to pick a GPU target (`gfx942`, `gfx90a`, `gfx1100`, …) and matching compiler flags.
- Which container images, environment variables, and driver checks actually unblock a setup.
- How to profile and tune workloads with tools like `rocprof`, `omniperf`, and `omnitrace`.

Skills capture that hard-won institutional knowledge once, in a format agents can apply consistently across teams and repos.

## Installation

Detailed install steps for each supported agent will land alongside the first published skills. The general flow will be:

### Cursor

Install the AMD plugin from this repository through the Cursor plugin flow. The repo will ship a `.cursor-plugin/plugin.json` and an `.mcp.json` so skills are discoverable as soon as the plugin is enabled.

### Claude Code

Register this repository as a plugin marketplace, then install individual skills:

```bash
/plugin marketplace add amd/skills
/plugin install <skill-name>@amd/skills
```

### OpenAI Codex

Copy or symlink the desired folders from `skills/` into one of Codex's standard skill locations (for example `$REPO_ROOT/.agents/skills` or `$HOME/.agents/skills`). Codex will discover the `SKILL.md` files and load them when relevant.

### Gemini CLI

A `gemini-extension.json` will be provided so the repo can be installed as a Gemini CLI extension:

```bash
gemini extensions install https://github.com/amd/skills.git --consent
```

## Using a skill

Once a skill is installed, just reference it in plain language while talking to your agent. For example:

- "Use the ROCm setup skill to get PyTorch running on this MI300X node."
- "Use the HIP porting skill to convert these CUDA kernels and flag anything that needs manual review."
- "Use the MIGraphX skill to compile this ONNX model for `gfx942` and benchmark it."
- "Use the ROCm profiling skill to capture an `omniperf` trace for this training step."

The agent will load the matching `SKILL.md` and any helper scripts, then carry out the task.

## Repository layout

```
skills/             # Individual skill folders (SKILL.md + assets)
agents/             # Aggregated AGENTS.md fallback for agents without skill support
.cursor-plugin/     # Cursor plugin manifest
.claude-plugin/     # Claude Code marketplace manifest
.github/workflows/  # CI for validating skills and manifests
scripts/            # Tooling for publishing and regenerating manifests
```

## Contributing a skill

We welcome contributions from AMD engineers, partners, and the community.

1. Copy an existing skill folder under `skills/` as a starting point and rename it.
2. Update the `SKILL.md` frontmatter so the `name` and `description` clearly explain *what* the skill does and *when* an agent should reach for it.
3. Add the supporting scripts, templates, and reference docs your instructions point to. Keep skills focused — one well-scoped task per skill is better than one mega-skill.
4. Register the skill in `.claude-plugin/marketplace.json` with a human-readable description.
5. Run the publishing script to validate and regenerate the metadata:

   ```bash
   ./scripts/publish.sh
   ```

6. Open a pull request. CI will verify that names, descriptions, and paths are consistent across `SKILL.md` files and the marketplace manifest.

### Writing tips

- Optimize the `description` for *agent routing*, not marketing. It is what decides whether the skill gets loaded.
- Be explicit about prerequisites (ROCm version, kernel, GPU arch, container image).
- Prefer scripts and runnable commands over prose where possible.
- Call out known pitfalls — driver mismatches, unsupported architectures, environment variables that silently change behavior.

## Status

This repository is in its early days. Skills, manifests, and CI are being built out; expect rapid iteration. File an issue if there is a workflow you want covered, or open a PR with a skill you have been wanting to share.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
