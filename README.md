# Инструкция для модели nnU-net.
## Склонировать репозиторий с сохраненной моделью
```
git clone https://git.miem.hse.ru/2057/2057.git
```
## (!!!) Сначала обязательно установить PyTorch
```
pip install --upgrade torch torchvision torchaudio
```
## Установить библиотеки
```
pip install -r requirements.txt
```
## Установить фреймворк nnU-net.
```
git submodule update --init --recursive 
cd nnUNet
pip install -e .
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

## Установить переменные среды в соответствии с путями к склонированным файлам
```
export nnUNet_raw="/media/fabian/nnUNet_raw"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export nnUNet_results="/media/fabian/nnUNet_results"
```
## Загрузить модель
```
cd ../nnUNet_results/Dataset001_MyDataset/nnUNetTrainer__nnUNetPlans__2d/fold_4
wget -L "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/IU7EjyQU11UVog" -O checkpoint_best.pth

```
## Перед началом инференса модели, убедитесь, что снимки в формате .nii.gz !!! Также они должны быть названы аналогично тому, как сделано в директории inputs.

## Для тестирования можно пользоваться файлами inputs

## Создайте директорию, куда хотите сохранять предсказания, например predictions

## Запуск (не забудьте указать свои пути для входа и выхода)
```
nnUNetv2_predict -i /Users/aleksandr/Downloads/2057/inputs -o /Users/aleksandr/Downloads/2057/predictions -d 1 -c 2d -device cpu -chk checkpoint_best.pth -f 4
```

## Если возникает ошибка ImportError: cannot import name 'GradScaler' from 'torch', идем в файл 
## ***nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py*** 
## и меняем 43 строчку с этой:
```
from torch import GradScaler
```
## на эту:
```
from torch.cuda.amp import GradScaler
```
