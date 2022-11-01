#!/usr/bin/env bash

set -e
set -u

__check_cuda() {
    ret=0
    nvcc -v >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        ret=1
    else
        ret=0
    fi
    echo $ret
    return $?
}

cuda_state=$(__check_cuda)

if [ ${cuda_state} -eq 0 ]; then
    echo "CUDA 没安装，安装pytorch-cpu :)"
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
else
    echo "CUDA 已经安装了，安装pytorch-cu :)"
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
fi

pip install numpy Pillow tensorboard
