"""
Вспомогательные утилиты для проекта.

Содержит функции для валидации файлов, работы со статистикой,
форматирования данных и других общих операций.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List
import os
import pandas as pd
from loguru import logger


# Константы
MAX_VIDEO_SIZE_MB = 500  # Максимальный размер видео в MB
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']


def validate_video_file(file_path: str) -> Tuple[bool, str]:
    """
    Проверяет корректность видеофайла.
    
    Args:
        file_path: Путь к видеофайлу.
        
    Returns:
        Кортеж (is_valid, error_message):
            - is_valid: True если файл корректен
            - error_message: Сообщение об ошибке (пустая строка если файл корректен)
            
    Example:
        >>> is_valid, error = validate_video_file('data/input/crowd.mp4')
        >>> if not is_valid:
        ...     print(f"Ошибка: {error}")
    """
    path = Path(file_path)
    
    # Проверка существования файла
    if not path.exists():
        return False, f"Файл не найден: {file_path}"
    
    # Проверка что это файл, а не директория
    if not path.is_file():
        return False, f"Путь не является файлом: {file_path}"
    
    # Проверка расширения файла
    if path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        supported = ', '.join(SUPPORTED_VIDEO_FORMATS)
        return False, f"Неподдерживаемый формат видео. Поддерживаются: {supported}"
    
    # Проверка размера файла
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_VIDEO_SIZE_MB:
        return False, f"Файл слишком большой ({file_size_mb:.1f} MB). Максимум: {MAX_VIDEO_SIZE_MB} MB"
    
    # Проверка прав на чтение
    if not os.access(file_path, os.R_OK):
        return False, f"Нет прав на чтение файла: {file_path}"
    
    logger.info(f"Валидация файла {file_path} успешна ({file_size_mb:.1f} MB)")
    return True, ""


def validate_output_directory(dir_path: str) -> Tuple[bool, str]:
    """
    Проверяет и создает выходную директорию если необходимо.
    
    Args:
        dir_path: Путь к директории.
        
    Returns:
        Кортеж (is_valid, error_message).
    """
    path = Path(dir_path)
    
    try:
        # Создаем директорию если не существует
        path.mkdir(parents=True, exist_ok=True)
        
        # Проверка прав на запись
        if not os.access(dir_path, os.W_OK):
            return False, f"Нет прав на запись в директорию: {dir_path}"
        
        logger.info(f"Выходная директория готова: {dir_path}")
        return True, ""
        
    except Exception as e:
        return False, f"Ошибка создания директории {dir_path}: {e}"


def format_time(seconds: float) -> str:
    """
    Форматирует время в читаемый формат.
    
    Args:
        seconds: Время в секундах.
        
    Returns:
        Отформатированная строка (например, "1h 23m 45s" или "45.3s").
        
    Example:
        >>> format_time(3665.5)
        '1h 1m 5.5s'
        >>> format_time(45.3)
        '45.3s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def calculate_processing_fps(frames_count: int, elapsed_time: float) -> float:
    """
    Вычисляет скорость обработки видео в FPS.
    
    Args:
        frames_count: Количество обработанных кадров.
        elapsed_time: Время обработки в секундах.
        
    Returns:
        Скорость обработки в кадрах в секунду.
        
    Example:
        >>> fps = calculate_processing_fps(300, 45.5)
        >>> print(f"Обработка: {fps:.2f} FPS")
    """
    if elapsed_time <= 0:
        return 0.0
    return frames_count / elapsed_time


def save_statistics_to_csv(
    frame_stats: List[Dict[str, Any]],
    output_path: str
) -> bool:
    """
    Сохраняет статистику по кадрам в CSV файл.
    
    Args:
        frame_stats: Список словарей со статистикой по каждому кадру.
        output_path: Путь для сохранения CSV файла.
        
    Returns:
        True если сохранение успешно, False иначе.
        
    Example:
        >>> frame_stats = [
        ...     {'frame_number': 0, 'timestamp': 0.0, 'detections_count': 5},
        ...     {'frame_number': 1, 'timestamp': 0.033, 'detections_count': 4}
        ... ]
        >>> save_statistics_to_csv(frame_stats, 'data/reports/stats.csv')
    """
    if not frame_stats:
        logger.warning("Нет данных для сохранения в CSV")
        return False
    
    try:
        # Создаем DataFrame
        df = pd.DataFrame(frame_stats)
        
        # Сортируем по номеру кадра
        if 'frame_number' in df.columns:
            df = df.sort_values('frame_number')
        
        # Сохраняем в CSV
        df.to_csv(output_path, index=False)
        
        logger.info(f"Статистика сохранена в {output_path} ({len(df)} строк)")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении CSV: {e}")
        return False


def save_detections_to_json(
    data: Dict[str, Any],
    output_path: str
) -> bool:
    """
    Сохраняет полные результаты детекции в JSON файл.
    
    Args:
        data: Словарь с результатами (video_info, statistics, frame_results).
        output_path: Путь для сохранения JSON файла.
        
    Returns:
        True если сохранение успешно, False иначе.
    """
    import json
    
    if not data:
        logger.warning("Нет данных для сохранения в JSON")
        return False
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Результаты сохранены в {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении JSON: {e}")
        return False


def generate_output_filename(
    input_path: str,
    suffix: str,
    output_dir: str = None
) -> str:
    """
    Генерирует имя выходного файла на основе входного.
    
    Args:
        input_path: Путь к входному файлу.
        suffix: Суффикс для добавления (например, 'processed', 'stats').
        output_dir: Директория для сохранения (опционально).
        
    Returns:
        Полный путь к выходному файлу.
        
    Example:
        >>> generate_output_filename('data/input/crowd.mp4', 'processed', 'data/output')
        'data/output/crowd_processed.mp4'
    """
    input_path_obj = Path(input_path)
    
    # Получаем имя файла без расширения
    stem = input_path_obj.stem
    
    # Получаем расширение
    extension = input_path_obj.suffix
    
    # Формируем новое имя
    new_name = f"{stem}_{suffix}{extension}"
    
    # Определяем директорию
    if output_dir:
        output_path = Path(output_dir) / new_name
    else:
        output_path = input_path_obj.parent / new_name
    
    return str(output_path)


def calculate_confidence_distribution(
    detections_list: List[List[Dict[str, Any]]]
) -> Dict[str, int]:
    """
    Вычисляет распределение детекций по уровням уверенности.
    
    Args:
        detections_list: Список списков детекций для каждого кадра.
        
    Returns:
        Словарь с количеством детекций для каждого уровня уверенности.
        
    Example:
        >>> dist = calculate_confidence_distribution(all_detections)
        >>> print(f"Высокая уверенность: {dist['high (≥0.8)']}")
    """
    distribution = {
        'high (≥0.8)': 0,
        'medium (0.5-0.8)': 0,
        'low (<0.5)': 0
    }
    
    for frame_detections in detections_list:
        for detection in frame_detections:
            conf = detection.get('confidence', 0)
            
            if conf >= 0.8:
                distribution['high (≥0.8)'] += 1
            elif conf >= 0.5:
                distribution['medium (0.5-0.8)'] += 1
            else:
                distribution['low (<0.5)'] += 1
    
    return distribution


def get_file_size_mb(file_path: str) -> float:
    """
    Возвращает размер файла в мегабайтах.
    
    Args:
        file_path: Путь к файлу.
        
    Returns:
        Размер файла в MB.
    """
    try:
        size_bytes = Path(file_path).stat().st_size
        return size_bytes / (1024 * 1024)
    except Exception:
        return 0.0


if __name__ == "__main__":
    # Тестирование утилит
    print("=== Тестирование утилит ===\n")
    
    # Тест форматирования времени
    print("Форматирование времени:")
    print(f"  45.3 сек -> {format_time(45.3)}")
    print(f"  125.5 сек -> {format_time(125.5)}")
    print(f"  3665.5 сек -> {format_time(3665.5)}")
    
    # Тест расчета FPS
    print("\nРасчет FPS обработки:")
    fps = calculate_processing_fps(300, 45.5)
    print(f"  300 кадров за 45.5 сек -> {fps:.2f} FPS")
    
    # Тест генерации имени файла
    print("\nГенерация имени файла:")
    output = generate_output_filename('data/input/crowd.mp4', 'processed', 'data/output')
    print(f"  {output}")
    
    print("\nВсе тесты завершены")
