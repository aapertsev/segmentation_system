
# Инструкция по установке и использованию для Linux
### 0) Предварительные требования:
Необходимо установить Python версии 3.10 и выше, а так же пакет с заголовками Python и инструменты сборки.  
На Fedora это можно сделать командой:
```
sudo dnf install python3-devel gcc
```

### 1) Склонировать репозиторий.
```
git clone https://github.com/aapertsev/segmentation_system.git
cd segmentation_system
```

### 2) В директории с проектом создать виртуальную среду и активировать ее.
```
python3 -m venv venv
source venv/bin/activate
```
### 3) Загрузить модель в директорию (вместо your_project_path - путь к склонированному репозиторию).

```
cd nnUNet_results/Dataset001_MyDataset/nnUNetTrainer__nnUNetPlans__2d/fold_4
curl -L "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/IU7EjyQU11UVog" -o checkpoint_best.pth
cd your_project_path
```

### 4) Запустить скрипт install_linux.sh (далее этот скрипт необходимо запускать при каждом новом сеансе).
```
cd install_files
chmod +x install_linux.sh
source install_linux.sh
```

### 5) Запустить приложение.
```
streamlit run web/streamlit_app.py
```
