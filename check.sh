#!/usr/bin/env bash

set -uo pipefail

PING_URL="https://huggingface.co/spaces/HariniPrithiyangara/trustchain-env"
REPO_DIR="."

printf "Pinging HF Space...\n"
curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$PING_URL/reset"

printf "\nRunning openenv validate...\n"
openenv validate
