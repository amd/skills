# Skill Card

## Description

Entry point that connects an agent to AMD Quark's own quantization skills. On
consent it fetches Quark's skill tree from the Quark repository into a cache
directory outside the user's workspace, then routes the agent to one of three
starting points: PyTorch / HuggingFace PTQ, ONNX PTQ, or Quark environment
setup. It quantizes nothing itself and installs nothing as a skill — the
quantization knowledge stays upstream in Quark, versioned with the product.

## Owner

AMD

## License

MIT
