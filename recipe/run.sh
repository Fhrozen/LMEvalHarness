#!/usr/bin/env bash
set -e
set -u
set -o pipefail

. ./path.sh

SECONDS=0

model_id=microsoft/phi-2
ngpu=1
cache_dir=models
tasks=mathqa
nj=12
verbosity=INFO

log "$0 $*"
. parse_options.sh

# Download a model:
# ```python
# from huggingface_hub import snapshot_download
# snapshot_download(model_id, cache_dir=cache_dir)
# ```

export NUMEXPR_MAX_THREADS=${nj}

models_args="cache_dir=${cache_dir},"
models_args+="pretrained=${model_id},"
models_args+="parallelize=True,"
models_args+="load_in_4bit=True,"
models_args+="trust_remote_code=True,"

working_dir=${model_id#*/}
working_dir="exp/${working_dir//-/_}"

log "LM Evaluation started... log: '${working_dir}/initial_eval.log'"
# run.pl --gpu ${ngpu} ${working_dir}/initial_eval.log \
        python -m lmeval_add.bin.run --model hf \
            --model_args ${models_args} \
            --tasks ${tasks} \
            --output_path ${working_dir}/init_evals \
            --batch_size auto \
            --log_samples \
            --verbosity ${verbosity} \
            --write_out \
            --device cuda 
            # --limit 10 

log "Successfully finished. [elapsed=${SECONDS}s]"
