# Skill Card

## Description

Validate Quark ONNX quantization output using four lightweight checks: auxiliary file copy alignment, expected non-quantized initializer MD5 byte-identity (inline `raw_data` + external-data byte ranges), model metadata equality after stripping quantization-only opset entries / Quark domains, and fuzzy node-pattern + o…

## Owner

amd (federated from [amd/Quark](https://github.com/amd/Quark))

## License

MIT
