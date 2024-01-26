#!/usr/bin/env bash

TASK="jcommonsenseqa_1.1-0.2,"
TASK+="jnli_1.3-0.2,"
TASK+="marc_ja_1.1-0.2,"
TASK+="jsquad_1.1-0.2,"
TASK+="jaqket_v2_0.2-0.2,"
TASK+="xlsum_ja_1.0-0.0,"  # Version 0.2 is missing (?)
TASK+="xwinograd_jp,"
TASK+="mgsm_1.0-0.0"

./run.sh --model_id cyberagent/open-calm-3b \
        --tasks $TASK \
        --num_fewshots "3 3 3 2 1 1 0 5"
