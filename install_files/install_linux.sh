#!/usr/bin/env bash

# 1) Проверяем, есть ли pip
if ! command -v pip >/dev/null 2>&1; then
  echo "Ошибка: pip не установлен или не находится в PATH."
  echo "Установите pip и повторите запуск."
  exit 1
fi

# 2) Определяем корневую директорию проекта
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 3) Проверяем, есть ли файл requirements.txt
REQ_FILE="${PROJECT_ROOT}/requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
  echo "Ошибка: не найден файл requirements.txt в директории ${PROJECT_ROOT}."
  exit 1
fi

# 4) Устанавливаем зависимости из requirements.txt
echo "Устанавливаем зависимости из requirements.txt..."
pip install -r "$REQ_FILE"
echo "Зависимости успешно установлены."

# 5) Инициализируем и обновляем сабмодули (nnUNet и другие)
echo "Инициализация сабмодулей..."
git submodule update --init --recursive
echo "Сабмодули успешно инициализированы."

# 6) Переходим в папку nnUNet и ставим её в режиме editable
cd "${PROJECT_ROOT}/nnUNet"
echo "Установка nnUNet (editable mode)..."
pip install -e .
echo "nnUNet успешно установлена."

# 7) Установка hiddenlayer из GitHub
echo "Установка/обновление hiddenlayer..."
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
echo "hiddenlayer успешно установлена/обновлена."

# 8) Экспорт переменных окружения (только для текущего сеанса)
echo "Экспорт переменных среды nnUNet..."
export nnUNet_raw="${PROJECT_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/nnUNet_results"

cd "${PROJECT_ROOT}"

echo "Все шаги скрипта выполнены успешно!"
echo "Внимание: переменные среды nnUNet_* действуют только в текущей сессии терминала."
echo "Если нужно, чтобы они были постоянными — пропишите их в ~/.bashrc"
