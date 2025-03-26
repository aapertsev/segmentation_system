#!/usr/bin/env bash

# Определяем директорию, в которой лежит скрипт
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Теперь поднимаемся на уровень выше
PROJECT_ROOT="$(cd "$SCRIPT_DIR"/.. && pwd)"
echo "Project root = $PROJECT_ROOT"
echo "Installing Python requirements..."
pip install -r "${PROJECT_ROOT}/requirements.txt"
echo "Installing nnUNet in editable mode..."
cd "${PROJECT_ROOT}"
git clone https://github.com/aapertsev/nnUNet.git
cd "${PROJECT_ROOT}/nnUNet"
pip install -e .
echo "Installing hiddenlayer from GitHub..."
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
echo "Exporting nnU-Net environment variables..."
export nnUNet_raw="${PROJECT_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/nnUNet_results"
cd "${PROJECT_ROOT}"
pip install streamlit
echo "Done!"