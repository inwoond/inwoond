"""
Веб-интерфейс на Streamlit для детекции людей на видео.

Предоставляет простой и интуитивный интерфейс для загрузки видео,
настройки параметров детекции и просмотра результатов.
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Добавляем src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from loguru import logger

from src.detector import PeopleDetector
from src.video_processor import VideoProcessor
from src.visualizer import Visualizer


# Конфигурация страницы
st.set_page_config(
    page_title="Crowd Detection - Детекция людей",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Константы
SUPPORTED_MODELS = {
    "YOLOv8 Nano (быстро, CPU)": "yolov8n.pt",
    "YOLOv8 Small (баланс)": "yolov8s.pt",
    "YOLOv8 Medium (точно, медленно)": "yolov8m.pt"
}


def initialize_session_state():
    """Инициализирует session state для хранения данных между перезагрузками."""
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None


def show_header():
    """Отображает заголовок приложения."""
    st.title("Crowd Detection - Детекция людей на видео")
    st.markdown("""
    Загрузите видео и получите:
    - Автоматическую детекцию людей
    - Детальную статистику
    - Анализ качества распознавания
    - Обработанное видео и отчеты
    """)
    st.divider()


def show_sidebar():
    """
    Отображает боковую панель с настройками.
    
    Returns:
        Кортеж (model_choice, confidence_threshold, show_stats).
    """
    with st.sidebar:
        st.header("Настройки")
        
        # Выбор модели
        st.subheader("Модель YOLO")
        model_display = st.selectbox(
            "Выберите модель",
            list(SUPPORTED_MODELS.keys()),
            index=0,
            help="Nano - быстрая на CPU, Medium - более точная но медленная"
        )
        model_choice = SUPPORTED_MODELS[model_display]
        
        st.divider()
        
        # Порог уверенности
        st.subheader("Параметры детекции")
        confidence_threshold = st.slider(
            "Порог уверенности",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Минимальная уверенность для отображения детекции"
        )
        
        st.divider()
        
        # Визуализация
        st.subheader("Визуализация")
        show_stats = st.checkbox(
            "Показывать статистику на видео",
            value=True,
            help="Отображать количество людей и FPS на кадрах"
        )
        
        st.divider()
        
        # Информация
        st.subheader("Информация")
        st.info("""
        **Формат видео:** MP4, AVI, MOV
        
        **Макс размер:** 500 MB
        
        **Класс детекции:** person (человек)
        """)
        
        # Версия
        st.caption("Version 1.0.0")
        
    return model_choice, confidence_threshold, show_stats


def upload_video_section():
    """
    Секция загрузки видео.
    
    Returns:
        Путь к загруженному видео или None.
    """
    st.header("Загрузка видео")
    
    uploaded_file = st.file_uploader(
        "Выберите видеофайл",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Поддерживаются форматы: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file is not None:
        # Показываем информацию о файле
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Имя файла", uploaded_file.name)
        with col2:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.metric("Размер", f"{file_size_mb:.1f} MB")
        
        # Сохраняем временно
        temp_dir = Path(tempfile.gettempdir()) / "crowd_detection"
        temp_dir.mkdir(exist_ok=True)
        
        temp_video_path = temp_dir / uploaded_file.name
        
        with open(temp_video_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"Видео загружено: {temp_video_path}")
        return str(temp_video_path)
    
    return None


def process_video_section(
    video_path: str,
    model_name: str,
    confidence_threshold: float,
    show_stats: bool
):
    """
    Секция обработки видео.
    
    Args:
        video_path: Путь к видео.
        model_name: Название модели YOLO.
        confidence_threshold: Порог уверенности.
        show_stats: Показывать ли статистику.
    """
    st.header("Обработка видео")
    
    # Кнопка запуска
    if st.button("Начать обработку", type="primary", use_container_width=True):
        
        with st.spinner("Инициализация моделей и загрузка видео..."):
            try:
                # Создание компонентов
                detector = PeopleDetector(
                    model_name=model_name,
                    confidence_threshold=confidence_threshold
                )
                
                visualizer = Visualizer(show_stats=show_stats)
                
                processor = VideoProcessor(
                    video_path=video_path,
                    output_dir='data/output'
                )
                
                processor.load_video()
                
                # Информация о видео
                video_info = processor.get_video_info()
                
                st.success("Видео успешно загружено!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Разрешение", f"{video_info['width']}x{video_info['height']}")
                with col2:
                    st.metric("FPS", f"{video_info['fps']:.2f}")
                with col3:
                    st.metric("Кадров", video_info['frame_count'])
                with col4:
                    st.metric("Длительность", f"{video_info['duration']:.1f}с")
                
            except Exception as e:
                st.error(f"Ошибка загрузки видео: {e}")
                logger.error(f"Ошибка загрузки видео: {e}")
                return
        
        # Прогресс обработки
        progress_bar = st.progress(0, text="Обработка видео...")
        status_text = st.empty()
        
        try:
            # Обработка видео
            st.info("⏳ Обработка видео... Это может занять некоторое время.")
            
            results = processor.process_video(
                detector=detector,
                visualizer=visualizer,
                save_video=True,
                save_reports=True
            )
            
            progress_bar.progress(100, text="Обработка завершена!")
            
            # Сохраняем результаты
            st.session_state.processed = True
            st.session_state.results = results
            st.session_state.video_path = video_path
            
            st.success("Видео успешно обработано!")
            st.balloons()
            
        except Exception as e:
            st.error(f"Ошибка при обработке видео: {e}")
            logger.error(f"Ошибка обработки: {e}")
            progress_bar.empty()
            return


def show_results_section():
    """Отображает секцию с результатами обработки."""
    if not st.session_state.processed or st.session_state.results is None:
        return
    
    st.header("Результаты обработки")
    
    results = st.session_state.results
    
    # Основные метрики
    st.subheader("Основные метрики")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Обработано кадров",
            f"{results['processed_frames']}/{results['total_frames']}"
        )
    
    with col2:
        st.metric(
            "Всего детекций",
            results['total_detections']
        )
    
    with col3:
        st.metric(
            "Среднее людей/кадр",
            f"{results['avg_detections_per_frame']:.2f}"
        )
    
    with col4:
        st.metric(
            "Скорость обработки",
            f"{results['processing_fps']:.2f} FPS"
        )
    
    st.divider()
    
    # Статистика детекций
    st.subheader("Статистика детекций")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Количество людей на кадрах:**")
        st.write(f"- Максимум: {results['max_detections_per_frame']}")
        st.write(f"- Минимум: {results['min_detections_per_frame']}")
        st.write(f"- Кадров с детекциями: {results['frames_with_detections']}")
        st.write(f"- Кадров без детекций: {results['frames_without_detections']}")
    
    with col2:
        st.markdown("**Уверенность детекций:**")
        st.write(f"- Средняя: {results['avg_confidence']:.3f}")
        st.write(f"- Медианная: {results['median_confidence']:.3f}")
        st.write(f"- Минимум: {results['min_confidence']:.3f}")
        st.write(f"- Максимум: {results['max_confidence']:.3f}")
    
    st.divider()
    
    # Распределение по уверенности
    st.subheader("Качество детекций")
    
    dist = results['confidence_distribution']
    total = results['total_detections']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_pct = (dist['high (≥0.8)'] / total * 100) if total > 0 else 0
        st.metric(
            "Высокая (≥0.8)",
            dist['high (≥0.8)'],
            f"{high_pct:.1f}%"
        )
    
    with col2:
        medium_pct = (dist['medium (0.5-0.8)'] / total * 100) if total > 0 else 0
        st.metric(
            "Средняя (0.5-0.8)",
            dist['medium (0.5-0.8)'],
            f"{medium_pct:.1f}%"
        )
    
    with col3:
        low_pct = (dist['low (<0.5)'] / total * 100) if total > 0 else 0
        st.metric(
            "Низкая (<0.5)",
            dist['low (<0.5)'],
            f"{low_pct:.1f}%"
        )
    
    # Оценка качества
    avg_conf = results['avg_confidence']
    if avg_conf >= 0.75:
        quality_emoji = "[OK]"
        quality_text = "Отлично"
        quality_color = "green"
    elif avg_conf >= 0.6:
        quality_emoji = "[!]"
        quality_text = "Удовлетворительно"
        quality_color = "orange"
    else:
        quality_emoji = "[X]"
        quality_text = "Требует улучшения"
        quality_color = "red"
    
    st.markdown(f"**Общая оценка качества:** {quality_emoji} {quality_text}")
    
    st.divider()
    
    # Время обработки
    st.subheader("Производительность")
    from src.utils import format_time
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Время обработки",
            format_time(results['processing_time_seconds'])
        )
    
    with col2:
        st.metric(
            "Скорость обработки",
            f"{results['processing_fps']:.2f} FPS"
        )


def show_download_section():
    """Отображает секцию загрузки результатов."""
    if not st.session_state.processed:
        return
    
    st.header("Скачать результаты")
    
    video_name = Path(st.session_state.video_path).stem
    
    col1, col2, col3 = st.columns(3)
    
    # Обработанное видео
    with col1:
        video_path = Path('data/output') / f"{video_name}_processed.mp4"
        if video_path.exists():
            with open(video_path, 'rb') as f:
                st.download_button(
                    label="Скачать видео",
                    data=f,
                    file_name=f"{video_name}_processed.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
    
    # CSV статистика
    with col2:
        csv_path = Path('data/reports') / f"{video_name}_stats.csv"
        if csv_path.exists():
            with open(csv_path, 'rb') as f:
                st.download_button(
                    label="Скачать CSV",
                    data=f,
                    file_name=f"{video_name}_stats.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # JSON результаты
    with col3:
        json_path = Path('data/reports') / f"{video_name}_detections.json"
        if json_path.exists():
            with open(json_path, 'rb') as f:
                st.download_button(
                    label="Скачать JSON",
                    data=f,
                    file_name=f"{video_name}_detections.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    # Markdown отчет
    report_path = Path('data/reports') / f"{video_name}_report.md"
    if report_path.exists():
        st.divider()
        
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        st.download_button(
            label="Скачать полный отчет (Markdown)",
            data=report_content,
            file_name=f"{video_name}_report.md",
            mime="text/markdown",
            use_container_width=True
        )


def show_footer():
    """Отображает подвал приложения."""
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>
            Crowd Detection v1.0.0<br>
            Детекция людей с использованием YOLOv8<br>
            <a href='https://github.com' target='_blank'>GitHub</a> | 
            <a href='#' target='_blank'>Документация</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Главная функция приложения."""
    # Инициализация
    initialize_session_state()
    
    # Заголовок
    show_header()
    
    # Боковая панель с настройками
    model_name, confidence_threshold, show_stats = show_sidebar()
    
    # Основной контент
    video_path = upload_video_section()
    
    if video_path:
        st.divider()
        process_video_section(video_path, model_name, confidence_threshold, show_stats)
    
    # Результаты (если есть)
    if st.session_state.processed:
        st.divider()
        show_results_section()
        
        st.divider()
        show_download_section()
    
    # Подвал
    show_footer()


if __name__ == "__main__":
    main()
