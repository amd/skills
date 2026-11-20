#!/usr/bin/env bash

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
# Thin entrypoint for the cluster environment check. The implementation is environment_check.py
# (importable by other lanes: it exposes image_exists() and the individual check functions, and a
# stable --json verdict). This wrapper keeps the documented `bash .../environment-check.sh` path
# working and forwards all arguments unchanged.
exec python3 "$(cd "$(dirname "$0")" && pwd)/environment_check.py" "$@"
