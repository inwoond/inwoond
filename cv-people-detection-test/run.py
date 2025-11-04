"""
Точка входа в приложение Crowd Detection.

Этот скрипт запускает Streamlit веб-интерфейс для детекции людей на видео.
"""

import subprocess
import sys
from pathlib import Path
from loguru import logger


def main():
    """
    Запускает Streamlit приложение.
    
    Проверяет наличие файла приложения и запускает его через Streamlit.
    """
    logger.info("Запуск Crowd Detection приложения...")
    
    # Путь к Streamlit приложению
    app_path = Path(__file__).parent / "app" / "streamlit_app.py"
    
    # Проверка существования файла
    if not app_path.exists():
        logger.error(f"Файл приложения не найден: {app_path}")
        print(f" Ошибка: файл {app_path} не найден")
        sys.exit(1)
    
    logger.info(f"Запуск Streamlit: {app_path}")
    
    try:
        # Запуск Streamlit с нужными параметрами
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка запуска Streamlit: {e}")
        print(f" Ошибка запуска приложения: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Приложение остановлено пользователем")
        print("\n Приложение остановлено")
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        print(f" Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("Crowd Detection - Детекция людей на видео")
    print("=" * 60)
    print()
    print("Запуск веб-интерфейса...")
    print("После запуска откройте в браузере: http://localhost:8501")
    print()
    print("Для остановки нажмите Ctrl+C")
    print("=" * 60)
    print()
    
    main()
