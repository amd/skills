# Skill Card

## Description

Diagnoses ROCm / HIP SDK / PyTorch / llama.cpp failures on AMD GPUs (Linux and
Windows) against a closed list of known misconfigurations, and applies a
low-risk fix with consent or routes the user to the right upstream channel. A
thin driver over the `rocm` CLI (`rocm examine` / `rocm diagnose` / `rocm fix`) —
the probe, catalog, and fixes live in the CLI, versioned with the binary.

## Owner

rocm-cli team (AMD)

## License

MIT
