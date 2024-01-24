TH_VERSION := 2.1.2

# Use pip for pytorch installation even if you have anaconda
ifneq ($(shell test -f ./activate_python.sh && grep 'conda activate' ./activate_python.sh),)
USE_CONDA := 1
else
USE_CONDA :=
endif

# Set if install binaries on CPU mode e.g. make CPU_ONLY=0
# If you don't have nvcc, this value will be set automatically
ifneq ($(shell which nvcc 2>/dev/null),)
CPU_ONLY :=
# Derive CUDA version from nvcc
CUDA_VERSION := $(shell nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
CUDA_VERSION_WITHOUT_DOT := $(strip $(subst .,,$(CUDA_VERSION)))

else
CPU_ONLY := 0
CUDA_VERSION :=
CUDA_VERSION_WITHOUT_DOT :=
endif
WITH_OMP=ON

.PHONY: all clean

all: showenv conda_packages.done python

python: activate_python.sh pytorch.done torch_optimizer.done llms.done llm_harness.done

activate_python.sh:
	test -f activate_python.sh || { echo "Error: Run ./setup_python.sh or ./setup_anaconda.sh"; exit 1; }

################ Logging ################
showenv: activate_python.sh
ifeq ($(strip $(CPU_ONLY)),)
	@echo CUDA_VERSION=$(CUDA_VERSION)
else
	@echo Perform on CPU mode: CPU_ONLY=$(CPU_ONLY)
endif
	@echo PYTHON=$(shell . ./activate_python.sh && command -v python3)
	@echo PYTHON_VERSION=$(shell . ./activate_python.sh && python3 --version)
	@echo USE_CONDA=$(USE_CONDA)
	@echo TH_VERSION=$(TH_VERSION)
	@echo WITH_OMP=$(WITH_OMP)

#########################################

bc.done: activate_python.sh
	. ./activate_python.sh && { command -v bc || conda install -y bc -c conda-forge; }
	touch bc.done
cmake.done: activate_python.sh
	. ./activate_python.sh && { command -v cmake || conda install -y cmake; }
	touch cmake.done

ifneq ($(strip $(USE_CONDA)),)
conda_packages.done: bc.done cmake.done
else
conda_packages.done:
endif
	touch conda_packages.done

pytorch.done: activate_python.sh
ifeq ($(strip $(USE_CONDA)),)
	. ./activate_python.sh && pip install numpy
	. ./activate_python.sh && ./installers/install_torch.sh "false" "${TH_VERSION}" "${CUDA_VERSION}"
else
	. ./activate_python.sh && conda install -y "numpy" "numba" -c numba
	. ./activate_python.sh && ./installers/install_torch.sh "true" "${TH_VERSION}" "${CUDA_VERSION}"
endif
	touch pytorch.done

torch_optimizer.done: pytorch.done
	. ./activate_python.sh && ./installers/install_torch_optimizer.sh
	touch torch_optimizer.done

# Install editable mode by default
llms.done: pytorch.done conda_packages.done
	. ./activate_python.sh && conda install scikit-learn -y
	. ./activate_python.sh && conda install matplotlib -y
	. ./activate_python.sh && python3 -m pip install -e "..[ja]"
	touch llms.done

llm_harness.done: llms.done
	. ./activate_python.sh && ./installers/install_llms_harness.sh
	touch llm_harness.done

clean:
	rm -rf *.done
	find . -iname "*.pyc" -delete
	rm -rf lmeval_harness
