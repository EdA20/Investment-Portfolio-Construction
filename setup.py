#!/usr/bin/env python3
import os
import sys
import subprocess

VENV_DIR = "..venv"
REQUIREMENTS = "requirements.txt"


def main():
    # Проверяем наличие Python
    if sys.version_info < (3, 3):
        sys.exit("Требуется Python версии 3.3 или выше")

    # Создаем виртуальное окружение
    try:
        subprocess.run([sys.executable, "-m", ".venv", VENV_DIR], check=True)
    except subprocess.CalledProcessError:
        sys.exit("Ошибка при создании виртуального окружения")

    # Определяем путь к pip
    if sys.platform == "win32":
        pip_path = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        pip_path = os.path.join(VENV_DIR, "bin", "pip")

    # Устанавливаем зависимости
    try:
        print('Запуск установки необходимых библиотек')
        subprocess.run([pip_path, "install", "--upgrade", "--force-reinstall", "-r", REQUIREMENTS], check=True)
    except FileNotFoundError:
        sys.exit(f"Файл {REQUIREMENTS} не найден")
    except subprocess.CalledProcessError:
        sys.exit("Ошибка при установке зависимостей")

    print(f"Успешно! Виртуальное окружение создано в {VENV_DIR}")
    print("Для активации выполните:")
    if sys.platform == "win32":
        print(f"{VENV_DIR}\\Scripts\\Activate.ps1")
    else:
        print(f"source {VENV_DIR}/bin/activate")


if __name__ == "__main__":
    main()