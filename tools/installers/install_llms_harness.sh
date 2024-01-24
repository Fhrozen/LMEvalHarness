#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

pip install gekko bitsandbytes

[ -d lmeval_harness ] && rm -rf lmeval_harness
git clone https://github.com/EleutherAI/lm-evaluation-harness lmeval_harness
cd lmeval_harness
pip install -e ".[gptq,anthropic,ifeval,math,multilingual,promptsource]"
cd ..
