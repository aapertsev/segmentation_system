
# Инструкция по установке и использованию для Windows

### 1) Склонировать репозиторий.
```
git clone https://github.com/aapertsev/segmentation_system.git
```

### 2) В директории с проектом создать виртуальную среду и активировать ее.
```
python -m venv venv
.\venv\Scripts\activate
```
### 3) Загрузить модель в директорию (вместо your_project_path - путь к склонированному репозиторию).

```
cd nnUNet_results/Dataset001_MyDataset/nnUNetTrainer__nnUNetPlans__2d/fold_4
curl -L "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/IU7EjyQU11UVog" -o checkpoint_best.pth
cd your_project_path
```

### 4) Запустить скрипт install.bat (далее этот скрипт необходимо запускать при каждом новом сеансе).
```
install_files/install_windows.bat
pip install PyQt5
```

### 5) Запустить приложение.
```
python interface/main.py 
```


