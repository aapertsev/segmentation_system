# 2057

# Инструкция для модели nnU-net.
## Склонировать репозиторий с сохраненной моделью
```
git clone https://git.miem.hse.ru/2057/2057.git
```
## (!!!) Сначала обязательно установить PyTorch
```
pip install --upgrade torch torchvision torchaudio
```
## Установить фреймворк nnU-net.
```
git submodule add https://github.com/MIC-DKFZ/nnUNet.git  
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
## Перед началом инференса модели, убедитесь, что снимки в формате .nii.gz !!! Также они должны быть названы аналогично тому, как сделано в директории inputs.

## Для тестирования можно пользоваться файлами inputs

## Создайте директорию, куда хотите сохранять предсказания, например predictions

## Запуск (не забудьте указать свои пути для входа и выхода)
```
nnUNetv2_predict -i /Users/aleksandr/Downloads/2057/inputs -o /Users/aleksandr/Downloads/2057/predictions -d 1 -c 2d -device cpu -chk checkpoint_best.pth -f 4
```


