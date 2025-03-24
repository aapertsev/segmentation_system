#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
pip install -r "${PROJECT_ROOT}/requirements.txt"
git submodule update --init --recursive
cd "${PROJECT_ROOT}/nnUNet"
pip install -e .
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
export nnUNet_raw="${PROJECT_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/nnUNet_results"
cd "${PROJECT_ROOT}"
