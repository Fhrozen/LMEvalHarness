#!/usr/bin/env bash
set -euo pipefail
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $# -ne 3 ]; then
    log "Usage: $0 <use_conda| true or false> <torch_version> <cuda_version>"
    exit 1
elif [ $# -eq 3 ]; then
    use_conda="$1"
    if [ "${use_conda}" != false ] && [ "${use_conda}" != true ]; then
        log "[ERROR] <use_conda> must be true or false, but ${use_conda} is given."
        log "Usage: $0 <use_conda| true or false> <torch_version> <cuda_version>"
        exit 1
    fi
    torch_version="$2"
    cuda_version="$3"
fi
if [ "${cuda_version}" = cpu ] || [ "${cuda_version}" = CPU ]; then
    cuda_version=
fi


python_version=$(python3 -c "import sys; print(sys.version.split()[0])")
cuda_version_without_dot="${cuda_version/\./}"


python_plus(){
    python3 <<EOF
from packaging.version import parse as L
if L('$python_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}
pytorch_plus(){
    python3 <<EOF
from packaging.version import parse as L
if L('$torch_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}
install_torch(){
# Usage: install_torch <torchaudio-version> <default-cuda-version-for-pip-install-torch>
    if "${use_conda}"; then
        log conda install -y "pytorch=${torch_version}" "torchaudio=$1" "pytorch-cuda=${cuda_version} torchvision" -c pytorch -c nvidia
        conda install -y "pytorch=${torch_version}" "torchaudio=$1" "pytorch-cuda=${cuda_version}" "torchvision" -c pytorch -c nvidia
    else
        if [ -z "${cuda_version}" ]; then
            log python3 -m pip install "torch==${torch_version}+cpu" "torchaudio==$1+cpu" -f https://download.pytorch.org/whl/torch_stable.html
            python3 -m pip install "torch==${torch_version}+cpu" "torchaudio==$1+cpu" -f https://download.pytorch.org/whl/torch_stable.html
        else
            if [ "${cuda_version}" = "$2" ]; then
                log python3 -m pip install "torch==${torch_version}" "torchaudio==$1"
                python3 -m pip install "torch==${torch_version}" "torchaudio==$1"
            else
                log python3 -m pip install "torch==${torch_version}+cu${cuda_version_without_dot}" "torchaudio==$1+cu${cuda_version_without_dot}" -f "https://download.pytorch.org/whl/torch_stable.html"
                python3 -m pip install "torch==${torch_version}+cu${cuda_version_without_dot}" "torchaudio==$1+cu${cuda_version_without_dot}" -f "https://download.pytorch.org/whl/torch_stable.html"
            fi
        fi
    fi
}
check_python_version(){
    if $(python_plus $1) || ! $(python_plus 3.6); then
        log "[ERROR] pytorch=${torch_version} doesn't provide binary build for python>=$1,<3.6, but your python is ${python_version}"
        exit 1
    fi
}
check_cuda_version(){
    supported=false
    for v in "" $@; do
        [ "${cuda_version}" = "${v}" ] && supported=true
    done
    if ! "${supported}"; then
        # See https://anaconda.org/pytorch/pytorch/files to provided version
        log "[ERROR] Pytorch=${torch_version} binary for CUDA=${cuda_version} is not provided. $@ are supported."
        exit 1
    fi
}


log "[INFO] python_version=${python_version}"
log "[INFO] torch_version=${torch_version}"
log "[INFO] cuda_version=${cuda_version}"

if ! python -c "import packaging.version" &> /dev/null; then
    python3 -m pip install packaging
fi

if $(pytorch_plus 2.1.3); then
    log "[ERROR] This script doesn't support pytorch=${torch_version}"
    exit 1

elif $(pytorch_plus 2.1.2); then
    check_python_version 3.11  # Error if python>=<number>
    check_cuda_version 11.8 12.1  # Error if cuda_version doesn't match with any given numbers
    install_torch 2.1.2 12.1  # install_torch <torch-audio-ver> <default-cuda-version-for-pip-install-torch>

elif $(pytorch_plus 2.0.0); then
    check_python_version 3.11  # Error if python>=<number>
    check_cuda_version 11.7 11.8  # Error if cuda_version doesn't match with any given numbers
    install_torch 2.0.0 11.7  # install_torch <torch-audio-ver> <default-cuda-version-for-pip-install-torch>
else
    log "[ERROR] This script doesn't support pytorch=${torch_version}"
    exit 1
fi
