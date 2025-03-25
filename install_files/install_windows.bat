@echo off
REM Переходим на один уровень выше директории, где лежит скрипт, и записываем путь в PROJECT_ROOT
set "PROJECT_ROOT=%~dp0..\"
echo Project root = %PROJECT_ROOT%
echo Installing Python requirements...
pip install -r "%PROJECT_ROOT%requirements.txt"
echo Updating git submodules...
git submodule update --init --recursive
echo Installing nnUNet in editable mode...
cd "%PROJECT_ROOT%nnUNet"
pip install -e .
echo Installing hiddenlayer from GitHub...
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
echo Exporting nnU-Net environment variables (local session only)...
set "nnUNet_raw=%PROJECT_ROOT%nnUNet_raw"
set "nnUNet_preprocessed=%PROJECT_ROOT%nnUNet_preprocessed"
set "nnUNet_results=%PROJECT_ROOT%nnUNet_results"
cd "%PROJECT_ROOT%"
echo Installing PyQt5...
pip install PyQt5
echo Done!
pause
