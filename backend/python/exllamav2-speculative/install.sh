#!/bin/bash
set -e

LIMIT_TARGETS="cublas"
EXTRA_PIP_INSTALL_FLAGS="--no-build-isolation"
EXLLAMA2_VERSION=6a2d8311408aa23af34e8ec32e28085ea68dada7

backend_dir=$(dirname $0)
if [ -d $backend_dir/common ]; then
    source $backend_dir/common/libbackend.sh
else
    source $backend_dir/../common/libbackend.sh
fi

# Check for SM_120 GPUs (RTX 5090/5080/5070/6000 PRO)
USE_PYTORCH_NIGHTLY=false
if nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | grep -q "12\.0"; then
    echo "Detected SM_120 GPU (RTX 5090/5080/5070/6000 PRO), using PyTorch nightly"
    USE_PYTORCH_NIGHTLY=true
fi

installRequirements

# Install PyTorch nightly for SM_120 support if needed
if [ "$USE_PYTORCH_NIGHTLY" = true ]; then
    echo "Installing PyTorch nightly with SM_120 support..."
    uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
fi

git clone https://github.com/turboderp/exllamav2 $MY_DIR/source
pushd ${MY_DIR}/source && git checkout -b build ${EXLLAMA2_VERSION} && popd

# This installs exllamav2 in JIT mode so it will compile the appropriate torch extension at runtime
EXLLAMA_NOCOMPILE= uv pip install ${EXTRA_PIP_INSTALL_FLAGS} ${MY_DIR}/source/
