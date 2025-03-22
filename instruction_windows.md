
# Инструкция по установке для Windows

### 1) Склонировать репозиторий к себе на компьютер
```
git clone https://github.com/aapertsev/segmentation_system.git
```
### 2) Скачать модель (вместо path_to_segmentation_system указать свой путь к склонированному репозитори
```
cd path_to_segmentation_system/nnUNet_results/Dataset001_MyDataset/nnUNetTrainer__nnUNetPlans__2d/fold_4
curl -L "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/IU7EjyQU11UVog" -o checkpoint_best.pth
cd path_to_segmatetion_system
```
### 3) Запустить скрипт install_windows.bat
```
install_windows.bat
```
### 4) Запустить систему
```
python interface/funcs.py
```
### (***) Если система начинает выдавать ошибки на инференсе, нужно заново выполнить пункт 3