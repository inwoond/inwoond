"""
Streamlit приложение для анализа металлургических данных.

Интерактивный инструмент для:
- Загрузки CSV/XLSX файлов
- Выбора столбцов для анализа  
- Кластеризации с различными алгоритмами
- Визуализации результатов
- Интерпретации через DecisionTree
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Импорт наших модулей
import sys
import os
sys.path.append('src')

try:
    from src.preprocessing import MetallurgyPreprocessor
    from src.clustering import MetallurgyClustering, compare_clustering_methods
    from src.explain_tree import ClusterExplainer, explain_clustering_results
except ImportError as e:
    st.error(f"Ошибка импорта модулей: {e}")
    st.stop()


# Конфигурация страницы
st.set_page_config(
    page_title="Анализ данных",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация session_state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False
if 'clustering_done' not in st.session_state:
    st.session_state.clustering_done = False


def load_data():
    """Загрузка данных из файла."""
    st.subheader("Загрузка данных")
    
    uploaded_file = st.file_uploader(
        "Выберите файл с производственными данными",
        type=['csv', 'xlsx'],
        help="Поддерживаются CSV и Excel файлы"
    )
    
    if uploaded_file is not None:
        try:
            # Загрузка файла
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.raw_data = df
            st.session_state.data_loaded = True
            
            # Отображение информации о данных
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Количество строк", df.shape[0])
            with col2:
                st.metric("Количество столбцов", df.shape[1])
            with col3:
                st.metric("Пропущенных значений", df.isnull().sum().sum())
            
            # Превью данных
            st.subheader("Превью данных")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Типы данных
            st.subheader("Информация о столбцах")
            type_info = pd.DataFrame({
                'Столбец': df.columns,
                'Тип данных': df.dtypes.astype(str),
                'Уникальных значений': [df[col].nunique() for col in df.columns],
                'Пропусков': [df[col].isnull().sum() for col in df.columns]
            })
            st.dataframe(type_info, use_container_width=True)
            
            return True
            
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {e}")
            return False
    
    return False


def select_features():
    """Выбор признаков для анализа."""
    if not st.session_state.data_loaded:
        st.warning("Сначала загрузите данные")
        return False
    
    st.subheader("Выбор признаков для анализа")
    
    df = st.session_state.raw_data
    
    # Автоматическое определение типов
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Числовые признаки:**")
        selected_numeric = st.multiselect(
            "Выберите числовые столбцы",
            numeric_cols,
            default=numeric_cols[:5],  # По умолчанию первые 5
            key="numeric_features"
        )
    
    with col2:
        st.write("**Категориальные признаки:**")
        selected_categorical = st.multiselect(
            "Выберите категориальные столбцы",
            categorical_cols,
            default=categorical_cols[:3] if len(categorical_cols) > 0 else [],
            key="categorical_features"
        )
    
    # Объединение выбранных признаков
    selected_features = selected_numeric + selected_categorical
    
    if len(selected_features) == 0:
        st.warning("Выберите хотя бы один признак для анализа")
        return False
    
    # Сохранение выбранных признаков
    st.session_state.selected_features = selected_features
    st.session_state.selected_data = df[selected_features].copy()
    
    st.success(f"Выбрано {len(selected_features)} признаков для анализа")
    
    return True


def data_filtering():
    """Фильтрация данных по диапазонам значений."""
    if 'selected_data' not in st.session_state:
        return False
    
    st.subheader("Фильтрация данных")
    
    df = st.session_state.selected_data.copy()
    
    # Фильтры для числовых признаков
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        st.write("**Фильтры по диапазонам значений:**")
        
        for col in numeric_cols:
            col_min, col_max = float(df[col].min()), float(df[col].max())
            
            # Слайдер для каждого числового признака
            range_values = st.slider(
                f"{col}",
                min_value=col_min,
                max_value=col_max,
                value=(col_min, col_max),
                key=f"filter_{col}"
            )
            
            # Применение фильтра
            df = df[(df[col] >= range_values[0]) & (df[col] <= range_values[1])]
    
    # Обработка выбросов
    st.write("**Обработка выбросов:**")
    remove_outliers = st.checkbox("Удалить выбросы (IQR метод)", key="remove_outliers")
    
    if remove_outliers:
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    # Сохранение отфильтрованных данных
    st.session_state.filtered_data = df
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Исходное количество строк", len(st.session_state.selected_data))
    with col2:
        st.metric("После фильтрации", len(df))
    
    if len(df) < 10:
        st.warning("Слишком мало данных после фильтрации. Попробуйте изменить настройки.")
        return False
    
    return True


def preprocessing_settings():
    """Настройки предобработки данных."""
    st.subheader("Настройки предобработки")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Нормализация числовых признаков:**")
        scaler_type = st.selectbox(
            "Тип нормализации",
            ["standard", "minmax"],
            format_func=lambda x: "StandardScaler" if x == "standard" else "MinMaxScaler",
            key="scaler_type"
        )
    
    with col2:
        st.write("**Обработка категориальных признаков:**")
        rare_threshold = st.slider(
            "Порог для редких категорий (%)",
            min_value=1,
            max_value=20,
            value=5,
            help="Категории встречающиеся реже этого порога будут объединены в 'Other'",
            key="rare_threshold"
        ) / 100
    
    return scaler_type, rare_threshold


def run_preprocessing():
    """Выполнение предобработки данных."""
    if 'filtered_data' not in st.session_state:
        st.warning("Сначала выберите и отфильтруйте данные")
        return False
    
    scaler_type, rare_threshold = preprocessing_settings()
    
    if st.button("Выполнить предобработку", type="primary"):
        with st.spinner("Выполняется предобработка данных..."):
            try:
                # Инициализация препроцессора
                preprocessor = MetallurgyPreprocessor(
                    numerical_scaler=scaler_type,
                    rare_threshold=rare_threshold
                )
                
                # Предобработка данных
                df = st.session_state.filtered_data
                X_processed = preprocessor.fit_transform(df)
                
                # Сохранение результатов
                st.session_state.preprocessor = preprocessor
                st.session_state.X_processed = X_processed
                st.session_state.preprocessing_done = True
                
                # Отчет о предобработке
                report = preprocessor.get_preprocessing_report()
                
                st.success("Предобработка выполнена успешно!")
                
                # Отображение отчета
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Числовые признаки", len(report['numeric_features']))
                with col2:
                    st.metric("OneHot признаки", len(report['onehot_features']))
                with col3:
                    st.metric("Label признаки", len(report['label_features']))
                
                st.write("**Детали предобработки:**")
                st.json(report)
                
                return True
                
            except Exception as e:
                st.error(f"Ошибка при предобработке: {e}")
                return False
    
    return False


def clustering_settings():
    """Настройки кластеризации."""
    st.subheader("Настройки кластеризации")
    
    col1, col2 = st.columns(2)
    
    with col1:
        algorithm = st.selectbox(
            "Алгоритм кластеризации",
            ["agglomerative", "dbscan", "faiss_kmeans"],
            format_func=lambda x: {
                "agglomerative": "Agglomerative Clustering",
                "dbscan": "DBSCAN", 
                "faiss_kmeans": "FAISS K-means"
            }[x],
            key="clustering_algorithm"
        )
    
    with col2:
        auto_clusters = st.checkbox(
            "Автоматическое определение количества кластеров",
            value=True,
            key="auto_clusters"
        )
        
        if not auto_clusters:
            n_clusters = st.slider(
                "Количество кластеров",
                min_value=2,
                max_value=20,
                value=5,
                key="n_clusters_manual"
            )
        else:
            n_clusters = None
    
    # Дополнительные настройки в зависимости от алгоритма
    if algorithm == "agglomerative":
        linkage_method = st.selectbox(
            "Метод связи",
            ["ward", "complete", "average", "single"],
            key="linkage_method"
        )
        return algorithm, n_clusters, {"linkage": linkage_method}
    
    elif algorithm == "dbscan":
        st.write("Параметры будут определены автоматически")
        return algorithm, n_clusters, {}
    
    else:  # faiss_kmeans
        return algorithm, n_clusters, {}


def run_clustering():
    """Выполнение кластеризации."""
    if not st.session_state.preprocessing_done:
        st.warning("Сначала выполните предобработку данных")
        return False
    
    algorithm, n_clusters, params = clustering_settings()
    
    # Выбор метрик для оценки
    st.write("**Метрики качества:**")
    metrics_selection = st.multiselect(
        "Выберите метрики для вычисления",
        ["silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"],
        default=["silhouette_score", "davies_bouldin_score"],
        key="selected_metrics"
    )
    
    if st.button("Запустить анализ", type="primary"):
        with st.spinner(f"Выполняется кластеризация методом {algorithm.upper()}..."):
            try:
                # Получение данных
                X = st.session_state.X_processed
                
                # Инициализация и выполнение кластеризации
                clustering = MetallurgyClustering()
                
                if algorithm == "agglomerative":
                    clustering.fit_agglomerative(X, n_clusters=n_clusters, **params)
                elif algorithm == "dbscan":
                    clustering.fit_dbscan(X, **params)
                elif algorithm == "faiss_kmeans":
                    clustering.fit_faiss_kmeans(X, n_clusters=n_clusters, **params)
                
                # Вычисление метрик
                metrics = clustering.calculate_metrics()
                
                # Сохранение результатов
                st.session_state.clustering = clustering
                st.session_state.cluster_labels = clustering.labels_
                st.session_state.clustering_metrics = metrics
                st.session_state.clustering_done = True
                
                st.success(f"Кластеризация выполнена! Найдено {clustering.n_clusters} кластеров")
                
                # Отображение метрик
                st.write("**Метрики качества кластеризации:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'silhouette_score' in metrics:
                        st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
                
                with col2:
                    if 'davies_bouldin_score' in metrics:
                        st.metric("Davies-Bouldin Index", f"{metrics['davies_bouldin_score']:.3f}")
                
                with col3:
                    if 'calinski_harabasz_score' in metrics:
                        st.metric("Calinski-Harabasz Index", f"{metrics['calinski_harabasz_score']:.1f}")
                
                return True
                
            except Exception as e:
                st.error(f"Ошибка при кластеризации: {e}")
                return False
    
    return False


def visualize_results():
    """Визуализация результатов кластеризации."""
    if not st.session_state.clustering_done:
        st.warning("Сначала выполните кластеризацию")
        return
    
    st.subheader("Визуализация результатов")
    
    # Получение данных
    clustering = st.session_state.clustering
    X = st.session_state.X_processed
    labels = st.session_state.cluster_labels
    
    # Вкладки для различных визуализаций
    tab1, tab2, tab3, tab4 = st.tabs(["PCA Проекция", "Дендрограмма", "Характеристики кластеров", "Распределения"])
    
    with tab1:
        st.write("**PCA проекция кластеров**")
        fig = clustering.plot_clusters_pca()
        st.pyplot(fig)
    
    with tab2:
        if clustering.algorithm == "agglomerative":
            st.write("**Дендрограмма иерархической кластеризации**")
            if len(X) <= 1000:  # Ограничение для читаемости
                fig = clustering.plot_dendrogram(X)
                st.pyplot(fig)
            else:
                st.warning("Слишком много точек для отображения дендрограммы")
        else:
            st.info("Дендрограмма доступна только для Agglomerative Clustering")
    
    with tab3:
        st.write("**Размеры кластеров**")
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        
        # Исключаем выбросы (label = -1) из отображения
        if -1 in cluster_sizes.index:
            outliers = cluster_sizes[-1]
            cluster_sizes = cluster_sizes[cluster_sizes.index != -1]
            st.write(f"Выбросов (DBSCAN): {outliers}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        cluster_sizes.plot(kind='bar', ax=ax)
        ax.set_title("Размеры кластеров")
        ax.set_xlabel("Номер кластера")
        ax.set_ylabel("Количество точек")
        st.pyplot(fig)
    
    with tab4:
        if hasattr(st.session_state, 'filtered_data'):
            st.write("**Распределение исходных признаков по кластерам**")
            
            df_viz = st.session_state.filtered_data.copy()
            df_viz['cluster'] = labels
            
            # Выбор признака для визуализации
            numeric_features = df_viz.select_dtypes(include=[np.number]).columns
            if len(numeric_features) > 0:
                selected_feature = st.selectbox(
                    "Выберите признак для визуализации",
                    numeric_features,
                    key="viz_feature"
                )
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Исключаем выбросы из визуализации
                df_clean = df_viz[df_viz['cluster'] != -1] if -1 in df_viz['cluster'].values else df_viz
                
                sns.boxplot(data=df_clean, x='cluster', y=selected_feature, ax=ax)
                ax.set_title(f"Распределение {selected_feature} по кластерам")
                st.pyplot(fig)


def interpret_clusters():
    """Интерпретация кластеров через DecisionTree."""
    if not st.session_state.clustering_done:
        st.warning("Сначала выполните кластеризацию")
        return
    
    st.subheader("Интерпретация кластеров")
    
    if st.button("Построить дерево решений", type="primary"):
        with st.spinner("Построение дерева решений для интерпретации..."):
            try:
                # Получение данных
                X = st.session_state.filtered_data
                labels = st.session_state.cluster_labels
                
                # Создание объяснителя
                explainer = ClusterExplainer(max_depth=5)
                explainer.fit(X, labels, feature_names=X.columns.tolist())
                
                # Сохранение результатов
                st.session_state.explainer = explainer
                
                st.success("Дерево решений построено успешно!")
                
                # Важность признаков
                st.write("**Важность признаков:**")
                importance_df = explainer.get_feature_importance(top_n=10)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.dataframe(importance_df, use_container_width=True)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(data=importance_df.head(8), x='importance', y='feature', ax=ax)
                    ax.set_title("Топ-8 важных признаков")
                    st.pyplot(fig)
                
                # Правила решений
                st.write("**Правила разделения кластеров:**")
                rules = explainer.get_decision_rules()
                st.text(rules[:2000] + "..." if len(rules) > 2000 else rules)
                
                # Отчет об интерпретации
                report = explainer.generate_cluster_interpretation_report()
                
                st.write("**Сводка по кластерам:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Точность дерева", f"{report['model_accuracy']:.3f}")
                with col2:
                    st.metric("Глубина дерева", report['tree_depth'])
                with col3:
                    st.metric("Количество листьев", report['n_leaves'])
                
                st.write("**Топ-3 важных признака:**")
                st.write(", ".join(report['top_3_features']))
                
            except Exception as e:
                st.error(f"Ошибка при построении дерева решений: {e}")


def export_results():
    """Экспорт результатов анализа."""
    if not st.session_state.clustering_done:
        st.warning("Нет результатов для экспорта")
        return
    
    st.subheader("Экспорт результатов")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Экспорт данных с кластерами"):
            # Подготовка данных для экспорта
            df_export = st.session_state.filtered_data.copy()
            df_export['cluster'] = st.session_state.cluster_labels
            
            # Конвертация в CSV
            csv = df_export.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            
            st.download_button(
                label="Скачать CSV с результатами",
                data=csv,
                file_name="clustering_results.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Экспорт отчета"):
            # Создание отчета
            report = {
                "algorithm": st.session_state.clustering.algorithm,
                "n_clusters": st.session_state.clustering.n_clusters,
                "metrics": st.session_state.clustering_metrics,
                "data_shape": st.session_state.filtered_data.shape,
            }
            
            if hasattr(st.session_state, 'explainer'):
                interp_report = st.session_state.explainer.generate_cluster_interpretation_report()
                report["interpretation"] = interp_report
            
            # Конвертация в JSON
            import json
            json_str = json.dumps(report, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="Скачать отчет JSON",
                data=json_str,
                file_name="analysis_report.json",
                mime="application/json"
            )


def main():
    """Основная функция приложения."""
    st.title("Инструмент для анализа данных")
    st.markdown("---")
    st.markdown(
        "**Научно-прикладной прототип** для анализа производственных данных "
        "с выявлением зависимостей между технологическими параметрами и КПЭ"
    )
    
    # Боковая панель с навигацией
    with st.sidebar:
        st.header("Этапы анализа")
        
        # Индикаторы прогресса
        progress_indicators = [
            ("1. Загрузка данных", st.session_state.data_loaded),
            ("2. Предобработка", st.session_state.preprocessing_done),
            ("3. Кластеризация", st.session_state.clustering_done)
        ]
        
        for step, completed in progress_indicators:
            if completed:
                st.success(step + " ✅")
            else:
                st.info(step + " ⏳")
        
        st.markdown("---")
        st.markdown("**Поддерживаемые алгоритмы:**")
        st.markdown("• Agglomerative Clustering")
        st.markdown("• DBSCAN")  
        st.markdown("• FAISS K-means")
        
        st.markdown("**Метрики качества:**")
        st.markdown("• Silhouette Score")
        st.markdown("• Davies-Bouldin Index")
        st.markdown("• Calinski-Harabasz Index")
    
    # Основное содержание
    # 1. Загрузка данных
    if load_data():
        
        # 2. Выбор признаков
        if select_features():
            
            # 3. Фильтрация данных
            if data_filtering():
                
                # 4. Предобработка
                run_preprocessing()
                
                # 5. Кластеризация
                if st.session_state.preprocessing_done:
                    run_clustering()
                    
                    # 6. Визуализация результатов
                    if st.session_state.clustering_done:
                        visualize_results()
                        
                        # 7. Интерпретация
                        interpret_clusters()
                        
                        # 8. Экспорт результатов
                        export_results()


if __name__ == "__main__":
    main()