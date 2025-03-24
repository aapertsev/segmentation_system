# Проект №2057 «Система сегментации сердца и аорты»

## Аннотация

&nbsp;&nbsp;&nbsp;В рамках проекта разработана система сегментации сердца и аорты на снимках компьютерной томографии (КТ), которая состоит из конвейера для предобработки изображений, нейросетевой модели для сегментации целевых анатомических структур и алгоритма визуализации результатов. Для обучения модели сегментации использовались размеченные снимки КТ, предоставленные Медицинским институтом, ФГБО ВО Орловского государственного университета им. И.С. Тургенева. 


## Команда проекта

1. **Перцев Александр Анатольевич** (БПМ222)  
   *Email:* aapertsev@edu.hse.ru    
   *Роль:* Лидер проекта, ML-разработчик

2. **Нам Виктория Сергеевна** (БПМ222)  
   *Email:* vsnam@edu.hse.ru  
   *Роль:* ML-разработчик

3. **Матиев Магомед Мусаевич** (БИБ233)  
   *Email:* mmmatiev@edu.hse.ru  
   *Роль:* Стажер

## Структура репозитория
```
2057/
├── interface/
│   ├── main.py         # Точка входа в приложение (GUI)
│   ├── gui.py          # Классы MainWindow и VisualizationWindow
│   └──  utils.py        # Утилитарные функции (конвертация JPG->NIfTI)
│
├── nnUNet_results/
│   └── Dataset001_MyDataset/
│       └── nnUNetTrainer__nnUNetPlans__2d/
│           └── fold_4/
│               └── checkpoint_best.pth # Модель сегментации
│   
├── install_instructions/
│   ├── install_linux.sh   # Инструкция для Linux/macOS
│   └── install_windows.bat # Инструкция для Windows
│
├── install_files/
│   ├── linux_install.md    # Установочный файл для Linux/macOS
│   └── windows_install.md  # Установочный файл для Windows
│ 
├── README.md          # Описание проекта (текущий файл)
└── ...
```

## Инструкции по установке и использованию
&nbsp;&nbsp;&nbsp;Проект разработан для использования на Windows, Linux и macOS. Инструкции по использованию и установке различаются для разных ОС.  
&nbsp;&nbsp;&nbsp;Чтобы установить на Linux/macOS используйте [данную инструкцию](install_instructions/linux_install.md).  
&nbsp;&nbsp;&nbsp;Чтобы установить на Windows используйте [данную инструкцию](install_instructions/windows_install.md). 
