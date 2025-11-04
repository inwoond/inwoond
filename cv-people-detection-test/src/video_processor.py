"""
Модуль для обработки видеофайлов с детекцией людей.

Содержит класс VideoProcessor, который оркестрирует весь процесс:
чтение видео, детекцию на каждом кадре, визуализацию и сохранение результата.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger

from .detector import PeopleDetector
from .visualizer import Visualizer
from .utils import (
    validate_video_file,
    validate_output_directory,
    format_time,
    calculate_processing_fps,
    save_statistics_to_csv,
    save_detections_to_json,
    generate_output_filename,
    calculate_confidence_distribution,
    get_file_size_mb
)


# Константы
OUTPUT_VIDEO_CODEC = 'mp4v'  # Кросс-платформенный кодек


class VideoProcessor:
    """
    Класс для обработки видео с детекцией людей.
    
    Выполняет покадровую обработку видео: детекцию людей, визуализацию
    результатов и сохранение обработанного видео с статистикой.
    
    Example:
        >>> processor = VideoProcessor('data/input/crowd.mp4')
        >>> processor.load_video()
        >>> detector = PeopleDetector()
        >>> visualizer = Visualizer()
        >>> results = processor.process_video(detector, visualizer)
        >>> print(f"Обработано кадров: {results['processed_frames']}")
    """
    
    def __init__(self, video_path: str, output_dir: str = 'data/output'):
        """
        Инициализация процессора видео.
        
        Args:
            video_path: Путь к входному видеофайлу.
            output_dir: Директория для сохранения результатов.
            
        Raises:
            ValueError: Если видеофайл некорректен.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        
        # Валидация входного файла
        is_valid, error_msg = validate_video_file(video_path)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Валидация выходной директории
        is_valid, error_msg = validate_output_directory(output_dir)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Переменные для видео
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_info: Dict[str, Any] = {}
        
        # Статистика
        self.frame_stats: List[Dict[str, Any]] = []
        self.all_detections: List[List[Dict[str, Any]]] = []
        
        logger.info(f"Инициализация VideoProcessor для {video_path}")
    
    def load_video(self) -> bool:
        """
        Загружает видеофайл и получает его метаданные.
        
        Returns:
            True если загрузка успешна, False иначе.
            
        Raises:
            RuntimeError: Если не удалось открыть видео.
        """
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Не удалось открыть видео: {self.video_path}")
            
            # Получение метаданных видео
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Явное вычисление длительности
            duration = frame_count / fps if fps > 0 else 0
            
            self.video_info = {
                'filename': Path(self.video_path).name,
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'file_size_mb': get_file_size_mb(self.video_path)
            }
            
            logger.info(
                f"Видео загружено: {width}x{height}, {fps:.2f} FPS, "
                f"{frame_count} кадров, {duration:.1f} сек"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки видео: {e}")
            raise RuntimeError(f"Не удалось загрузить видео: {e}")
    
    def _create_video_writer(self, output_path: str) -> cv2.VideoWriter:
        """
        Создает VideoWriter для сохранения обработанного видео.
        
        Args:
            output_path: Путь для сохранения видео.
            
        Returns:
            Объект VideoWriter.
        """
        fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
        
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.video_info['fps'],
            (self.video_info['width'], self.video_info['height'])
        )
        
        if not writer.isOpened():
            raise RuntimeError(f"Не удалось создать VideoWriter: {output_path}")
        
        logger.info(f"VideoWriter создан: {output_path}")
        return writer
    
    def process_video(
        self,
        detector: PeopleDetector,
        visualizer: Visualizer,
        save_video: bool = True,
        save_reports: bool = True
    ) -> Dict[str, Any]:
        """
        Обрабатывает видео: детекция, визуализация, сохранение.
        
        Это основная функция, которая выполняет весь pipeline обработки.
        
        Args:
            detector: Экземпляр PeopleDetector для детекции.
            visualizer: Экземпляр Visualizer для отрисовки.
            save_video: Сохранять ли обработанное видео.
            save_reports: Сохранять ли статистику и отчеты.
            
        Returns:
            Словарь с результатами обработки и статистикой.
            
        Example:
            >>> results = processor.process_video(detector, visualizer)
            >>> print(f"Всего детекций: {results['total_detections']}")
        """
        if self.cap is None:
            raise RuntimeError("Видео не загружено. Вызовите load_video() сначала.")
        
        logger.info("Начало обработки видео...")
        start_time = time.time()
        
        # Подготовка выходных путей
        output_video_path = None
        if save_video:
            output_video_path = generate_output_filename(
                self.video_path, 'processed', self.output_dir
            )
            self.video_writer = self._create_video_writer(output_video_path)
        
        # Сброс статистики
        self.frame_stats = []
        self.all_detections = []
        
        # Обработка кадров
        frame_number = 0
        processed_frames = 0
        failed_frames = 0
        
        # Прогресс-бар
        pbar = tqdm(
            total=self.video_info['frame_count'],
            desc="Обработка видео",
            unit="кадр"
        )
        
        while True:
            # Чтение кадра
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            try:
                # Временная метка кадра
                timestamp = frame_number / self.video_info['fps']
                
                # Детекция людей
                detections = detector.detect(frame)
                self.all_detections.append(detections)
                
                # Статистика по кадру
                frame_stat = self._calculate_frame_stats(
                    frame_number, timestamp, detections
                )
                self.frame_stats.append(frame_stat)
                
                # Визуализация
                if save_video:
                    stats_for_viz = {
                        'detections_count': len(detections),
                        'frame_number': frame_number,
                        'fps': calculate_processing_fps(
                            processed_frames + 1,
                            time.time() - start_time
                        )
                    }
                    
                    annotated_frame = visualizer.draw_all(
                        frame, detections, stats_for_viz
                    )
                    
                    # Запись кадра
                    self.video_writer.write(annotated_frame)
                
                processed_frames += 1
                
            except Exception as e:
                logger.error(f"Ошибка обработки кадра {frame_number}: {e}")
                failed_frames += 1
            
            frame_number += 1
            pbar.update(1)
        
        pbar.close()
        
        # Завершение обработки
        elapsed_time = time.time() - start_time
        
        # Освобождение ресурсов
        self.release_resources()
        
        # Вычисление итоговой статистики
        processing_stats = self._calculate_processing_statistics(
            processed_frames, failed_frames, elapsed_time
        )
        
        # Сохранение отчетов
        if save_reports:
            self._save_all_reports(processing_stats)
        
        logger.info(
            f"Обработка завершена: {processed_frames} кадров за {format_time(elapsed_time)}"
        )
        
        return processing_stats
    
    def _calculate_frame_stats(
        self,
        frame_number: int,
        timestamp: float,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Вычисляет статистику для одного кадра.
        
        Args:
            frame_number: Номер кадра.
            timestamp: Временная метка в видео.
            detections: Список детекций на кадре.
            
        Returns:
            Словарь со статистикой кадра.
        """
        if not detections:
            return {
                'frame_number': frame_number,
                'timestamp': round(timestamp, 3),
                'detections_count': 0,
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0
            }
        
        confidences = [d['confidence'] for d in detections]
        
        return {
            'frame_number': frame_number,
            'timestamp': round(timestamp, 3),
            'detections_count': len(detections),
            'avg_confidence': round(np.mean(confidences), 3),
            'max_confidence': round(max(confidences), 3),
            'min_confidence': round(min(confidences), 3)
        }
    
    def _calculate_processing_statistics(
        self,
        processed_frames: int,
        failed_frames: int,
        elapsed_time: float
    ) -> Dict[str, Any]:
        """
        Вычисляет итоговую статистику обработки.
        
        Args:
            processed_frames: Количество успешно обработанных кадров.
            failed_frames: Количество кадров с ошибками.
            elapsed_time: Время обработки в секундах.
            
        Returns:
            Словарь с полной статистикой.
        """
        # Подсчет детекций
        total_detections = sum(len(dets) for dets in self.all_detections)
        
        # Статистика по кадрам
        frames_with_detections = sum(
            1 for dets in self.all_detections if len(dets) > 0
        )
        frames_without_detections = len(self.all_detections) - frames_with_detections
        
        # Статистика по уверенности
        all_confidences = []
        for frame_dets in self.all_detections:
            all_confidences.extend([d['confidence'] for d in frame_dets])
        
        if all_confidences:
            avg_confidence = float(np.mean(all_confidences))
            median_confidence = float(np.median(all_confidences))
            std_confidence = float(np.std(all_confidences))
            min_confidence = float(min(all_confidences))
            max_confidence = float(max(all_confidences))
        else:
            avg_confidence = median_confidence = std_confidence = 0.0
            min_confidence = max_confidence = 0.0
        
        # Детекции по кадрам
        detections_per_frame = [len(dets) for dets in self.all_detections]
        avg_detections_per_frame = (
            np.mean(detections_per_frame) if detections_per_frame else 0
        )
        
        # Распределение по уверенности
        confidence_dist = calculate_confidence_distribution(self.all_detections)
        
        return {
            'video_info': self.video_info,
            'total_frames': self.video_info['frame_count'],
            'processed_frames': processed_frames,
            'failed_frames': failed_frames,
            'total_detections': total_detections,
            'avg_detections_per_frame': round(avg_detections_per_frame, 2),
            'max_detections_per_frame': max(detections_per_frame) if detections_per_frame else 0,
            'min_detections_per_frame': min(detections_per_frame) if detections_per_frame else 0,
            'frames_with_detections': frames_with_detections,
            'frames_without_detections': frames_without_detections,
            'avg_confidence': round(avg_confidence, 3),
            'median_confidence': round(median_confidence, 3),
            'std_confidence': round(std_confidence, 3),
            'min_confidence': round(min_confidence, 3),
            'max_confidence': round(max_confidence, 3),
            'confidence_distribution': confidence_dist,
            'processing_time_seconds': round(elapsed_time, 2),
            'processing_fps': round(calculate_processing_fps(processed_frames, elapsed_time), 2),
            'start_time': datetime.now().isoformat(),
            'end_time': datetime.now().isoformat()
        }
    
    def _save_all_reports(self, stats: Dict[str, Any]) -> None:
        """
        Сохраняет все отчеты: CSV, JSON, Markdown.
        
        Args:
            stats: Статистика обработки.
        """
        reports_dir = Path('data/reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(self.video_path).stem
        
        # CSV со статистикой по кадрам
        csv_path = reports_dir / f"{base_name}_stats.csv"
        save_statistics_to_csv(self.frame_stats, str(csv_path))
        
        # JSON с полными результатами
        json_data = {
            'video_info': self.video_info,
            'processing_statistics': stats,
            'frame_results': self.frame_stats
        }
        json_path = reports_dir / f"{base_name}_detections.json"
        save_detections_to_json(json_data, str(json_path))
        
        # Markdown отчет
        report_path = reports_dir / f"{base_name}_report.md"
        self._generate_markdown_report(stats, str(report_path))
        
        logger.info(f"Все отчеты сохранены в {reports_dir}")
    
    def _generate_markdown_report(self, stats: Dict[str, Any], output_path: str) -> None:
        """
        Генерирует Markdown отчет с анализом результатов.
        
        Args:
            stats: Статистика обработки.
            output_path: Путь для сохранения отчета.
        """
        report = f"""# Отчет по детекции людей

## Входное видео
- Файл: {stats['video_info']['filename']}
- Разрешение: {stats['video_info']['width']}x{stats['video_info']['height']}
- FPS: {stats['video_info']['fps']:.2f}
- Длительность: {format_time(stats['video_info']['duration'])}
- Размер файла: {stats['video_info']['file_size_mb']:.1f} MB

## Результаты обработки
- Обработано кадров: {stats['processed_frames']}/{stats['total_frames']}
- Кадров с ошибками: {stats['failed_frames']}
- Время обработки: {format_time(stats['processing_time_seconds'])}
- Скорость обработки: {stats['processing_fps']:.2f} FPS

## Статистика детекций
- Всего детекций: {stats['total_detections']}
- Среднее количество людей на кадр: {stats['avg_detections_per_frame']:.2f}
- Максимум людей на кадре: {stats['max_detections_per_frame']}
- Минимум людей на кадре: {stats['min_detections_per_frame']}
- Кадров с детекциями: {stats['frames_with_detections']}
- Кадров без детекций: {stats['frames_without_detections']}

## Анализ качества
### Уверенность детекций
- Средняя уверенность: {stats['avg_confidence']:.3f}
- Медианная уверенность: {stats['median_confidence']:.3f}
- Стандартное отклонение: {stats['std_confidence']:.3f}
- Минимум: {stats['min_confidence']:.3f}
- Максимум: {stats['max_confidence']:.3f}

### Распределение по уровням уверенности
- Высокая (≥0.8): {stats['confidence_distribution']['high (≥0.8)']} ({stats['confidence_distribution']['high (≥0.8)']/stats['total_detections']*100:.1f}%)
- Средняя (0.5-0.8): {stats['confidence_distribution']['medium (0.5-0.8)']} ({stats['confidence_distribution']['medium (0.5-0.8)']/stats['total_detections']*100:.1f}%)
- Низкая (<0.5): {stats['confidence_distribution']['low (<0.5)']} ({stats['confidence_distribution']['low (<0.5)']/stats['total_detections']*100:.1f}%)

## Выводы и рекомендации

### Качество детекции
{"Отлично" if stats['avg_confidence'] >= 0.75 else "Удовлетворительно" if stats['avg_confidence'] >= 0.6 else "Требует улучшения"}
- Средняя уверенность {stats['avg_confidence']:.3f} {"выше" if stats['avg_confidence'] >= 0.7 else "ниже"} рекомендуемого порога 0.7

### Производительность
{"Хорошая" if stats['processing_fps'] >= 5 else "Средняя" if stats['processing_fps'] >= 2 else "Низкая"}
- Скорость обработки: {stats['processing_fps']:.2f} FPS

### Рекомендации по улучшению
1. **Повышение точности детекции:**
   - Использовать YOLOv8m/l модели для лучшей точности
   - Fine-tuning на похожих данных (толпы людей)
   - Увеличить порог уверенности до 0.6-0.7

2. **Улучшение обработки перекрытий:**
   - Применить NMS (Non-Maximum Suppression) с оптимальным IoU
   - Использовать модели, обученные на dense crowds

3. **Оптимизация производительности:**
   - Уменьшить разрешение входного видео
   - Использовать GPU для ускорения (если доступен)
   - Обрабатывать каждый N-й кадр для real-time

4. **Дополнительные метрики:**
   - Добавить трекинг для подсчета уникальных людей
   - Реализовать анализ траекторий движения
   - Добавить heat map плотности людей

---
*Отчет сгенерирован: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Markdown отчет сохранен: {output_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения Markdown отчета: {e}")
    
    def release_resources(self) -> None:
        """
        Освобождает ресурсы (VideoCapture и VideoWriter).
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.debug("VideoCapture освобожден")
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            logger.debug("VideoWriter освобожден")
    
    def get_video_info(self) -> Dict[str, Any]:
        """
        Возвращает метаданные видео.
        
        Returns:
            Словарь с информацией о видео.
        """
        return self.video_info.copy()
    
    def __del__(self):
        """Деструктор для гарантированного освобождения ресурсов."""
        self.release_resources()


if __name__ == "__main__":
    # Пример использования
    from detector import PeopleDetector
    from visualizer import Visualizer
    
    # Путь к видео
    video_path = "data/input/crowd.mp4"
    
    if Path(video_path).exists():
        try:
            # Инициализация компонентов
            processor = VideoProcessor(video_path)
            processor.load_video()
            
            detector = PeopleDetector(model_name="yolov8n.pt")
            visualizer = Visualizer(show_stats=True)
            
            # Обработка видео
            results = processor.process_video(detector, visualizer)
            
            # Вывод результатов
            print("\n=== Результаты обработки ===")
            print(f"Обработано кадров: {results['processed_frames']}")
            print(f"Всего детекций: {results['total_detections']}")
            print(f"Среднее людей на кадр: {results['avg_detections_per_frame']:.2f}")
            print(f"Средняя уверенность: {results['avg_confidence']:.3f}")
            print(f"Время обработки: {format_time(results['processing_time_seconds'])}")
            print(f"Скорость: {results['processing_fps']:.2f} FPS")
            
        except Exception as e:
            logger.error(f"Ошибка: {e}")
    else:
        print(f"Видео не найдено: {video_path}")
        print("Положите файл crowd.mp4 в директорию data/input/")
