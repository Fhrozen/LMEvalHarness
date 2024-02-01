#!/usr/bin/env bash
set -e
set -u
set -o pipefail

. ./path.sh

SECONDS=0

model_id=microsoft/phi-2
ngpu=2
cache_dir=models
tasks=mathqa
nj=12
verbosity=INFO
num_fewshots=""

use_mlflow=false
experiment_name=lm_harness
run_name=
track_url="sqlite:///$(readlink -e ./)/mlruns.sql"

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

model_id=${model_id#*/}
model_id=${model_id//-/_}
working_dir="exp/output_${model_id}"

log "LM Evaluation started... log: '${working_dir}/run.log'"

run_args=""
if [ -n "${num_fewshots}" ]; then
    run_args="--num_fewshot ${num_fewshots}"
fi

run.pl --gpu ${ngpu} ${working_dir}/run.log \
        python -m lmeval_add.bin.run --model hf \
            --model_args ${models_args} \
            --tasks ${tasks} \
            --output_path ${working_dir} \
            --batch_size auto \
            --device cuda \
            --verbosity ${verbosity} \
            --write_out \
            --log_samples \
            ${run_args}

if ${use_mlflow} ;then
    if [ -z "${run_name}" ]; then
        run_name=${model_id}
    fi

    log "Reporting into mlflow run_name: ${run_name}"
    python -m lmeval_add.bin.mlflow_report \
        --experiment_name ${experiment_name} \
        --run_name ${run_name} \
        --track_url ${track_url} \
        --results_dir ${working_dir} \
        --config_details ${models_args}

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
