@echo off
set "PROJECT_ROOT=%~dp0"
pip install -r "%PROJECT_ROOT%requirements.txt"
git submodule update --init --recursive
cd "%PROJECT_ROOT%nnUNet"
pip install -e .
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
setx "nnUNet_raw=%PROJECT_ROOT%nnUNet_raw"
setx "nnUNet_preprocessed=%PROJECT_ROOT%nnUNet_preprocessed"
setx "nnUNet_results=%PROJECT_ROOT%nnUNet_results"
cd "%PROJECT_ROOT%nnUNet_results\Dataset001_MyDataset\nnUNetTrainer__nnUNetPlans__2d\fold_4"
curl -L "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/IU7EjyQU11UVog" -o checkpoint_best.pth
cd "%PROJECT_ROOT%"
pause
