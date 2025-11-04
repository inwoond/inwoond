"""
Модуль для визуализации результатов детекции людей.

Этот модуль содержит класс Visualizer, который отрисовывает bounding boxes,
текстовые метки и статистику на кадрах видео.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from loguru import logger


# Константы для визуализации
BOX_COLOR_HIGH_CONF = (0, 255, 0)      # Зеленый для confidence >= 0.8
BOX_COLOR_MEDIUM_CONF = (0, 165, 255)  # Оранжевый для 0.5 <= confidence < 0.8
BOX_COLOR_LOW_CONF = (0, 0, 255)       # Красный для confidence < 0.5
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
LABEL_PADDING = 5


class Visualizer:
    """
    Класс для визуализации результатов детекции на изображениях.
    
    Выполняет отрисовку bounding boxes, текстовых меток с именем класса
    и уверенностью, а также общей статистики на кадре.
    
    Example:
        >>> visualizer = Visualizer()
        >>> annotated_frame = visualizer.draw_detections(frame, detections)
        >>> cv2.imshow('Result', annotated_frame)
    """
    
    def __init__(
        self,
        show_stats: bool = True,
        box_thickness: int = BOX_THICKNESS,
        font_scale: float = FONT_SCALE
    ):
        """
        Инициализация визуализатора.
        
        Args:
            show_stats: Показывать ли статистику на кадре (количество людей, FPS).
            box_thickness: Толщина линий bounding box.
            font_scale: Размер шрифта для текстовых меток.
        """
        self.show_stats = show_stats
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        
        logger.info("Инициализация Visualizer")
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Отрисовывает все детекции на кадре.
        
        Для каждой детекции рисует:
        - Bounding box (цвет зависит от уверенности)
        - Текстовую метку с именем класса и confidence
        
        Args:
            frame: Исходный кадр (numpy array, BGR формат).
            detections: Список детекций из PeopleDetector.detect().
            
        Returns:
            Аннотированный кадр с отрисованными детекциями.
            
        Example:
            >>> detections = [
            ...     {'bbox': [100, 100, 200, 300], 'confidence': 0.85, 'class_name': 'person'}
            ... ]
            >>> result = visualizer.draw_detections(frame, detections)
        """
        if frame is None or frame.size == 0:
            logger.warning("Получен пустой кадр для визуализации")
            return frame
        
        # Создаем копию кадра, чтобы не изменять оригинал
        annotated_frame = frame.copy()
        
        # Отрисовка каждой детекции
        for detection in detections:
            self._draw_single_detection(annotated_frame, detection)
        
        logger.debug(f"Отрисовано {len(detections)} детекций")
        return annotated_frame
    
    def _draw_single_detection(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any]
    ) -> None:
        """
        Отрисовывает одну детекцию на кадре.
        
        Args:
            frame: Кадр для рисования (изменяется in-place).
            detection: Словарь с информацией о детекции.
        """
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        
        # Преобразуем координаты в целые числа
        x1, y1, x2, y2 = map(int, bbox)
        
        # Выбор цвета в зависимости от уверенности
        color = self._get_color_by_confidence(confidence)
        
        # Отрисовка bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
        
        # Формирование текстовой метки: "person: 0.85"
        label = f"{class_name}: {confidence:.2f}"
        
        # Отрисовка метки
        self._draw_label(frame, label, (x1, y1), color)
    
    def _get_color_by_confidence(self, confidence: float) -> Tuple[int, int, int]:
        """
        Возвращает цвет bounding box в зависимости от уверенности.
        
        Args:
            confidence: Уверенность детекции (0.0-1.0).
            
        Returns:
            Кортеж (B, G, R) с цветом для OpenCV.
        """
        if confidence >= 0.8:
            return BOX_COLOR_HIGH_CONF  # Зеленый
        elif confidence >= 0.5:
            return BOX_COLOR_MEDIUM_CONF  # Оранжевый
        else:
            return BOX_COLOR_LOW_CONF  # Красный
    
    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int]
    ) -> None:
        """
        Отрисовывает текстовую метку с фоном над bounding box.
        
        Args:
            frame: Кадр для рисования (изменяется in-place).
            text: Текст метки.
            position: Позиция (x, y) левого верхнего угла bounding box.
            color: Цвет текста и фона.
        """
        x, y = position
        
        # Получаем размер текста
        (text_width, text_height), baseline = cv2.getTextSize(
            text, FONT, self.font_scale, FONT_THICKNESS
        )
        
        # Координаты прямоугольника для фона
        label_y = y - 10 if y - 10 > text_height else y + text_height + 10
        
        # Отрисовка фона для текста
        cv2.rectangle(
            frame,
            (x, label_y - text_height - LABEL_PADDING),
            (x + text_width + LABEL_PADDING * 2, label_y + LABEL_PADDING),
            color,
            -1  # Заполненный прямоугольник
        )
        
        # Отрисовка текста
        cv2.putText(
            frame,
            text,
            (x + LABEL_PADDING, label_y),
            FONT,
            self.font_scale,
            (255, 255, 255),  # Белый цвет текста
            FONT_THICKNESS,
            cv2.LINE_AA
        )
    
    def draw_statistics(
        self,
        frame: np.ndarray,
        stats: Dict[str, Any]
    ) -> np.ndarray:
        """
        Отрисовывает статистику в углу кадра.
        
        Args:
            frame: Исходный кадр.
            stats: Словарь со статистикой:
                - detections_count: количество людей
                - frame_number: номер кадра (опционально)
                - fps: FPS обработки (опционально)
                
        Returns:
            Кадр со статистикой.
            
        Example:
            >>> stats = {'detections_count': 5, 'frame_number': 42, 'fps': 6.5}
            >>> result = visualizer.draw_statistics(frame, stats)
        """
        if not self.show_stats:
            return frame
        
        if frame is None or frame.size == 0:
            return frame
        
        annotated_frame = frame.copy()
        
        # Формирование текста статистики
        stats_text = []
        
        if 'detections_count' in stats:
            stats_text.append(f"People: {stats['detections_count']}")
        
        if 'frame_number' in stats:
            stats_text.append(f"Frame: {stats['frame_number']}")
        
        if 'fps' in stats:
            stats_text.append(f"FPS: {stats['fps']:.1f}")
        
        # Отрисовка статистики в левом верхнем углу
        y_offset = 30
        for i, line in enumerate(stats_text):
            y_position = y_offset + i * 30
            
            # Фон для текста
            (text_width, text_height), _ = cv2.getTextSize(
                line, FONT, self.font_scale, FONT_THICKNESS
            )
            
            cv2.rectangle(
                annotated_frame,
                (10, y_position - text_height - 5),
                (20 + text_width, y_position + 5),
                (0, 0, 0),  # Черный фон
                -1
            )
            
            # Текст
            cv2.putText(
                annotated_frame,
                line,
                (15, y_position),
                FONT,
                self.font_scale,
                (255, 255, 255),  # Белый текст
                FONT_THICKNESS,
                cv2.LINE_AA
            )
        
        return annotated_frame
    
    def draw_all(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        stats: Dict[str, Any] = None
    ) -> np.ndarray:
        """
        Отрисовывает и детекции, и статистику на кадре.
        
        Удобная функция для одновременной отрисовки всего.
        
        Args:
            frame: Исходный кадр.
            detections: Список детекций.
            stats: Словарь со статистикой (опционально).
            
        Returns:
            Полностью аннотированный кадр.
            
        Example:
            >>> result = visualizer.draw_all(frame, detections, stats)
        """
        # Сначала отрисовываем детекции
        annotated_frame = self.draw_detections(frame, detections)
        
        # Затем статистику
        if stats is not None:
            annotated_frame = self.draw_statistics(annotated_frame, stats)
        
        return annotated_frame


if __name__ == "__main__":
    # Пример использования
    import cv2
    
    # Создание тестового изображения
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (50, 50, 50)  # Серый фон
    
    # Тестовые детекции
    test_detections = [
        {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'class_name': 'person'
        },
        {
            'bbox': [300, 150, 400, 350],
            'confidence': 0.65,
            'class_name': 'person'
        },
        {
            'bbox': [450, 80, 550, 280],
            'confidence': 0.45,
            'class_name': 'person'
        }
    ]
    
    # Тестовая статистика
    test_stats = {
        'detections_count': 3,
        'frame_number': 42,
        'fps': 6.5
    }
    
    # Визуализация
    visualizer = Visualizer(show_stats=True)
    result = visualizer.draw_all(test_frame, test_detections, test_stats)
    
    # Сохранение результата
    cv2.imwrite('test_visualization.jpg', result)
    print("Тестовая визуализация сохранена в test_visualization.jpg")
