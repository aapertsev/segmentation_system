@echo off
set "PROJECT_ROOT=%~dp0"
pip install -r "%PROJECT_ROOT%requirements.txt"
pip install PyQt5
git submodule update --init --recursive
cd "%PROJECT_ROOT%nnUNet"
pip install -e .
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
set "nnUNet_raw=%PROJECT_ROOT%nnUNet_raw"
set "nnUNet_preprocessed=%PROJECT_ROOT%nnUNet_preprocessed"
set "nnUNet_results=%PROJECT_ROOT%nnUNet_results"
cd "%PROJECT_ROOT%"
pause
