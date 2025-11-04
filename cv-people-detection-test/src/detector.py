"""
Модуль для детекции людей на изображениях с использованием YOLO.

Этот модуль содержит класс PeopleDetector, который загружает предобученную
модель YOLOv8 и выполняет детекцию людей на изображениях/кадрах видео.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO
from loguru import logger


# Константы
DEFAULT_MODEL = "yolov8n.pt"  # Nano модель для CPU
CONFIDENCE_THRESHOLD = 0.5  # Порог уверенности
PERSON_CLASS_ID = 0  # ID класса "person" в COCO dataset


class PeopleDetector:
    """
    Класс для детекции людей на изображениях с использованием YOLOv8.
    
    Attributes:
        model_name (str): Название файла модели YOLO.
        confidence_threshold (float): Минимальный порог уверенности для детекций.
        model (YOLO): Загруженная модель YOLO.
        
    Example:
        >>> detector = PeopleDetector()
        >>> detections = detector.detect(frame)
        >>> print(f"Найдено людей: {len(detections)}")
    """
    
    def __init__(
        self, 
        model_name: str = DEFAULT_MODEL,
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ):
        """
        Инициализация детектора людей.
        
        Args:
            model_name: Название модели YOLO (yolov8n.pt, yolov8s.pt, etc.).
            confidence_threshold: Порог уверенности для фильтрации детекций (0.0-1.0).
            
        Raises:
            ValueError: Если confidence_threshold не в диапазоне [0, 1].
        """
        if not 0 <= confidence_threshold <= 1:
            raise ValueError(
                f"confidence_threshold должен быть между 0 и 1, получено: {confidence_threshold}"
            )
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None
        
        logger.info(f"Инициализация PeopleDetector с моделью {model_name}")
        self.load_model()
    
    def load_model(self) -> None:
        """
        Загружает модель YOLO из файла или скачивает её автоматически.
        
        Модель сохраняется в директорию models/ и кешируется для
        последующих запусков. При первом запуске модель скачивается
        автоматически из репозитория Ultralytics.
        
        Raises:
            RuntimeError: Если не удалось загрузить модель.
        """
        try:
            logger.info(f"Загрузка модели {self.model_name}...")
            
            # Ultralytics автоматически скачает модель, если её нет локально
            self.model = YOLO(self.model_name)
            
            logger.info(f"Модель {self.model_name} успешно загружена")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise RuntimeError(f"Не удалось загрузить модель {self.model_name}: {e}")
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Выполняет детекцию людей на одном кадре.
        
        Args:
            frame: Изображение в формате numpy array (BGR, OpenCV формат).
            
        Returns:
            Список словарей с информацией о каждой детекции:
                - bbox: [x1, y1, x2, y2] - координаты bounding box
                - confidence: float - уверенность модели (0.0-1.0)
                - class_id: int - ID класса (всегда 0 для person)
                - class_name: str - название класса (всегда 'person')
                
        Example:
            >>> frame = cv2.imread('image.jpg')
            >>> detections = detector.detect(frame)
            >>> for det in detections:
            ...     print(f"Person: confidence={det['confidence']:.2f}")
            
        Raises:
            ValueError: Если frame пустой или имеет неверный формат.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Получен пустой кадр для детекции")
        
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"frame должен быть numpy.ndarray, получено: {type(frame)}")
        
        if len(frame.shape) != 3:
            raise ValueError(f"frame должен быть цветным изображением (3D array), получено: {frame.shape}")
        
        try:
            # Запуск детекции (verbose=False отключает подробный вывод)
            results = self.model(frame, verbose=False)
            
            # Извлечение детекций
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Получаем ID класса
                    class_id = int(box.cls[0])
                    
                    # Фильтруем только людей (class_id == 0 в COCO)
                    if class_id != PERSON_CLASS_ID:
                        continue
                    
                    # Получаем уверенность
                    confidence = float(box.conf[0])
                    
                    # Фильтруем по порогу уверенности
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Получаем координаты bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': 'person'
                    }
                    
                    detections.append(detection)
            
            logger.debug(f"Обнаружено {len(detections)} человек(а) на кадре")
            return detections
            
        except Exception as e:
            logger.error(f"Ошибка при детекции: {e}")
            return []
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Устанавливает новый порог уверенности для детекций.
        
        Args:
            threshold: Новый порог уверенности (0.0-1.0).
            
        Raises:
            ValueError: Если threshold не в диапазоне [0, 1].
        """
        if not 0 <= threshold <= 1:
            raise ValueError(
                f"threshold должен быть между 0 и 1, получено: {threshold}"
            )
        
        self.confidence_threshold = threshold
        logger.info(f"Порог уверенности изменен на {threshold}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о загруженной модели.
        
        Returns:
            Словарь с информацией о модели:
                - model_name: название модели
                - confidence_threshold: текущий порог уверенности
                - is_loaded: загружена ли модель
        """
        return {
            'model_name': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'is_loaded': self.model is not None
        }


if __name__ == "__main__":
    # Пример использования
    import cv2
    
    # Создание детектора
    detector = PeopleDetector(model_name="yolov8n.pt", confidence_threshold=0.5)
    
    # Загрузка тестового изображения
    test_image_path = "data/input/test_image.jpg"
    if Path(test_image_path).exists():
        frame = cv2.imread(test_image_path)
        
        # Детекция
        detections = detector.detect(frame)
        
        print(f"Обнаружено людей: {len(detections)}")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. Уверенность: {det['confidence']:.2f}, BBox: {det['bbox']}")
    else:
        print(f"Тестовое изображение не найдено: {test_image_path}")
