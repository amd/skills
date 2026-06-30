# Skill Card

## Description

Multi-instance vLLM benchmark on AMD EPYC CPU: runs N vLLM instances (each pinned to a range of physical cores) behind an NGINX load balancer, drives load with guidellm via ansible, and reports peak aggregate memory (podman stats) plus end-to-end throughput/latency across models, concurrency rates, and instance counts. The benchmark harness is vendored with the skill; nothing external is required beyond podman + ansible.

## Owner

AMD

## License

MIT
