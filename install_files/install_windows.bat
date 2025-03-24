@echo off
REM Включаем "delayed expansion" для корректной обработки переменных при IF
setlocal enabledelayedexpansion

REM 1) Проверяем, установлен ли pip
where pip >nul 2>nul
if %errorlevel% neq 0 (
    echo Ошибка: pip не найден в PATH. Установите Python/pip или добавьте в PATH.
    exit /b 1
)

REM 2) Определяем корневую директорию (папка с этим скриптом)
set "PROJECT_ROOT=%~dp0"

REM 3) Проверяем наличие файла requirements.txt
if not exist "%PROJECT_ROOT%requirements.txt" (
    echo Ошибка: не найден файл requirements.txt в %PROJECT_ROOT%.
    exit /b 1
)

REM 4) Устанавливаем зависимости
echo Устанавливаем зависимости из requirements.txt ...
pip install -r "%PROJECT_ROOT%requirements.txt"
if %errorlevel% neq 0 (
    echo Ошибка: pip install не смог установить зависимости.
    exit /b 1
)
echo Зависимости успешно установлены.

REM 5) Инициализируем сабмодули
echo Инициализируем сабмодули...
git submodule update --init --recursive
if %errorlevel% neq 0 (
    echo Ошибка: не удалось инициализировать сабмодули (git submodule).
    exit /b 1
)
echo Сабмодули успешно инициализированы.

REM 6) Переходим в папку nnUNet и устанавливаем в режиме editable
cd "%PROJECT_ROOT%nnUNet"
if %errorlevel% neq 0 (
    echo Ошибка: не найдена директория %PROJECT_ROOT%nnUNet
    exit /b 1
)

echo Установка nnUNet (editable mode)...
pip install -e .
if %errorlevel% neq 0 (
    echo Ошибка: не удалось установить nnUNet в editable mode.
    exit /b 1
)

echo Установка/обновление hiddenlayer...
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
if %errorlevel% neq 0 (
    echo Ошибка: не удалось установить/обновить hiddenlayer.
    exit /b 1
)
echo nnUNet и hiddenlayer успешно установлены/обновлены.

REM 7) Устанавливаем переменные среды (только в текущем сеансе cmd)
echo Устанавливаем переменные окружения nnUNet_*, будут действовать только в этом сеансе...
set "nnUNet_raw=%PROJECT_ROOT%nnUNet_raw"
set "nnUNet_preprocessed=%PROJECT_ROOT%nnUNet_preprocessed"
set "nnUNet_results=%PROJECT_ROOT%nnUNet_results"

REM 8) Возвращаемся в корневую директорию
cd "%PROJECT_ROOT%"
if %errorlevel% neq 0 (
    echo Ошибка: не удалось вернуться в директорию %PROJECT_ROOT%.
    exit /b 1
)

echo Установка завершена без ошибок!
echo Внимание: переменные среды nnUNet_* действуют только в текущем окне cmd.
echo Если нужно сохранить их постоянно, используйте setx или пропишите в .bat/.ps1
pause
