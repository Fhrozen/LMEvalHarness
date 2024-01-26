#!/usr/bin/env bash

TASK="jcommonsenseqa-1.1-0.2,"
TASK+="jnli-1.1-0.2,"
TASK+="marc_ja-1.1-0.2,"
TASK+="jsquad-1.1-0.2,"
TASK+="jaqket_v2-0.2-0.2,"
TASK+="xlsum_ja,"
TASK+="xwinograd_ja"
TASK+="mgsm"

./run.sh --model_id rinna/japanese-gpt-neox-small \
        --tasks $TASK \
        --num_fewshots "3 3 3 2 1 1 0 5"