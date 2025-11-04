"""
Тесты для модуля detector.py.
"""

import pytest
import numpy as np
from src.detector import PeopleDetector


def test_detector_initialization():
    """Тест инициализации детектора."""
    detector = PeopleDetector(model_name="yolov8n.pt", confidence_threshold=0.5)
    
    assert detector.model_name == "yolov8n.pt"
    assert detector.confidence_threshold == 0.5
    assert detector.model is not None


def test_detector_invalid_confidence():
    """Тест с некорректным порогом уверенности."""
    with pytest.raises(ValueError):
        PeopleDetector(confidence_threshold=1.5)
    
    with pytest.raises(ValueError):
        PeopleDetector(confidence_threshold=-0.1)


def test_set_confidence_threshold():
    """Тест изменения порога уверенности."""
    detector = PeopleDetector()
    
    detector.set_confidence_threshold(0.7)
    assert detector.confidence_threshold == 0.7
    
    with pytest.raises(ValueError):
        detector.set_confidence_threshold(2.0)


def test_detect_with_empty_frame():
    """Тест детекции с пустым кадром."""
    detector = PeopleDetector()
    
    empty_frame = np.array([])
    
    with pytest.raises(ValueError):
        detector.detect(empty_frame)


def test_detect_with_invalid_frame():
    """Тест детекции с некорректным форматом кадра."""
    detector = PeopleDetector()
    
    # 2D массив вместо 3D
    invalid_frame = np.zeros((100, 100), dtype=np.uint8)
    
    with pytest.raises(ValueError):
        detector.detect(invalid_frame)


def test_detect_with_valid_frame():
    """Тест детекции с корректным кадром."""
    detector = PeopleDetector()
    
    # Создаем тестовый кадр (черное изображение)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Детекция должна вернуть список (возможно пустой)
    detections = detector.detect(test_frame)
    
    assert isinstance(detections, list)


def test_get_model_info():
    """Тест получения информации о модели."""
    detector = PeopleDetector()
    
    info = detector.get_model_info()
    
    assert 'model_name' in info
    assert 'confidence_threshold' in info
    assert 'is_loaded' in info
    assert info['is_loaded'] is True
