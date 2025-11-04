"""
Тесты для модуля utils.py.
"""

import pytest
from pathlib import Path
import tempfile
from src.utils import (
    validate_video_file,
    validate_output_directory,
    format_time,
    calculate_processing_fps,
    generate_output_filename,
    calculate_confidence_distribution,
    get_file_size_mb
)


def test_format_time():
    """Тест форматирования времени."""
    assert format_time(45.3) == "45.3s"
    assert format_time(125.5) == "2m 5.5s"
    assert format_time(3665.5) == "1h 1m 5.5s"


def test_calculate_processing_fps():
    """Тест расчета FPS обработки."""
    fps = calculate_processing_fps(300, 45.5)
    assert pytest.approx(fps, rel=0.01) == 6.59
    
    # Нулевое время
    fps = calculate_processing_fps(100, 0)
    assert fps == 0.0


def test_validate_output_directory():
    """Тест валидации выходной директории."""
    # Создание временной директории
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test_output"
        
        is_valid, error = validate_output_directory(str(test_dir))
        
        assert is_valid is True
        assert error == ""
        assert test_dir.exists()


def test_generate_output_filename():
    """Тест генерации имени выходного файла."""
    input_path = "data/input/crowd.mp4"
    
    output = generate_output_filename(input_path, "processed", "data/output")
    
    assert "crowd_processed.mp4" in output
    assert "data/output" in output or "data\\output" in output


def test_calculate_confidence_distribution():
    """Тест расчета распределения уверенности."""
    detections_list = [
        [
            {'confidence': 0.95},
            {'confidence': 0.85},
            {'confidence': 0.65}
        ],
        [
            {'confidence': 0.45},
            {'confidence': 0.75}
        ]
    ]
    
    dist = calculate_confidence_distribution(detections_list)
    
    assert dist['high (≥0.8)'] == 2  # 0.95, 0.85
    assert dist['medium (0.5-0.8)'] == 2  # 0.65, 0.75
    assert dist['low (<0.5)'] == 1  # 0.45


def test_get_file_size_mb():
    """Тест получения размера файла."""
    # Создаем временный файл
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"x" * (1024 * 1024))  # 1 MB
        tmp_path = tmp.name
    
    try:
        size_mb = get_file_size_mb(tmp_path)
        assert pytest.approx(size_mb, rel=0.01) == 1.0
    finally:
        Path(tmp_path).unlink()


def test_validate_video_file_nonexistent():
    """Тест валидации несуществующего файла."""
    is_valid, error = validate_video_file("nonexistent_video.mp4")
    
    assert is_valid is False
    assert "не найден" in error.lower()
