"""
Модуль интерпретации результатов кластеризации через DecisionTree.

Функции:
- Построение DecisionTree на результатах кластеризации
- Анализ важности признаков (feature importance)
- Визуализация дерева решений
- Генерация текстового отчета по ключевым разделениям
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class ClusterExplainer:
    """
    Класс для объяснения результатов кластеризации через DecisionTree.
    """
    
    def __init__(self, max_depth=5, min_samples_split=10, min_samples_leaf=5):
        """
        Инициализация объяснителя кластеров.
        
        Parameters:
        -----------
        max_depth : int, default=5
            Максимальная глубина дерева
        min_samples_split : int, default=10
            Минимальное количество образцов для разбиения узла
        min_samples_leaf : int, default=5
            Минимальное количество образцов в листе
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.decision_tree = None
        self.feature_names = None
        self.cluster_labels = None
        self.X_data = None
        self.is_fitted = False
        
    def fit(self, X, cluster_labels, feature_names=None):
        """
        Обучение DecisionTree для объяснения кластеров.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Исходные данные (признаки)
        cluster_labels : np.ndarray
            Метки кластеров (целевая переменная)
        feature_names : list, optional
            Названия признаков
            
        Returns:
        --------
        self
        """
        # Подготовка данных
        if isinstance(X, pd.DataFrame):
            self.X_data = X.values
            if feature_names is None:
                self.feature_names = X.columns.tolist()
        else:
            self.X_data = X
            if feature_names is None:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            else:
                self.feature_names = feature_names
        
        # Исключение выбросов (метки -1 для DBSCAN)
        mask = cluster_labels != -1
        X_clean = self.X_data[mask]
        labels_clean = cluster_labels[mask]
        
        if len(np.unique(labels_clean)) < 2:
            raise ValueError("Недостаточно кластеров для обучения дерева решений")
        
        self.cluster_labels = labels_clean
        
        # Создание и обучение DecisionTree
        self.decision_tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        
        self.decision_tree.fit(X_clean, labels_clean)
        self.is_fitted = True
        
        return self
    
    def get_feature_importance(self, top_n=None):
        """
        Получение важности признаков.
        
        Parameters:
        -----------
        top_n : int, optional
            Количество топ признаков для возврата
            
        Returns:
        --------
        pd.DataFrame : таблица с важностью признаков
        """
        if not self.is_fitted or self.decision_tree is None:
            raise ValueError("Модель не обучена")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.decision_tree.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def plot_feature_importance(self, top_n=15, figsize=(12, 8)):
        """
        Визуализация важности признаков.
        
        Parameters:
        -----------
        top_n : int, default=15
            Количество топ признаков для отображения
        figsize : tuple, default=(12, 8)
            Размер фигуры
            
        Returns:
        --------
        matplotlib figure
        """
        importance_df = self.get_feature_importance(top_n=top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Важность признаков для разделения кластеров (Топ-{top_n})')
        plt.xlabel('Важность признака')
        plt.ylabel('Признак')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()
    
    def plot_decision_tree(self, figsize=(20, 15), max_depth_plot=3):
        """
        Визуализация дерева решений.
        
        Parameters:
        -----------
        figsize : tuple, default=(20, 15)
            Размер фигуры
        max_depth_plot : int, default=3
            Максимальная глубина для отображения (для читаемости)
            
        Returns:
        --------
        matplotlib figure
        """
        if not self.is_fitted or self.decision_tree is None or self.cluster_labels is None:
            raise ValueError("Модель не обучена")
        
        plt.figure(figsize=figsize)
        
        plot_tree(
            self.decision_tree,
            feature_names=self.feature_names,
            class_names=[f'Кластер {i}' for i in np.unique(self.cluster_labels)],
            filled=True,
            rounded=True,
            fontsize=10,
            max_depth=max_depth_plot
        )
        
        plt.title('Дерево решений для объяснения кластеров')
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()
    
    def get_decision_rules(self, max_rules=10):
        """
        Извлечение правил решений в текстовом виде.
        
        Parameters:
        -----------
        max_rules : int, default=10
            Максимальное количество правил для отображения
            
        Returns:
        --------
        str : текстовые правила решений
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        
        tree_rules = export_text(
            self.decision_tree,
            feature_names=self.feature_names,
            max_depth=3  # Ограничиваем глубину для читаемости
        )
        
        return tree_rules
    
    def generate_cluster_interpretation_report(self):
        """
        Генерация подробного отчета об интерпретации кластеров.
        
        Returns:
        --------
        dict : словарь с отчетом
        """
        if not self.is_fitted:
            return {"error": "Модель не обучена"}
        
        # Основные метрики дерева
        accuracy = self.decision_tree.score(self.X_data[self.cluster_labels != -1], 
                                          self.cluster_labels[self.cluster_labels != -1])
        
        # Важность признаков
        feature_importance = self.get_feature_importance(top_n=10)
        
        # Статистика по кластерам
        cluster_stats = {}
        unique_clusters = np.unique(self.cluster_labels)
        
        for cluster_id in unique_clusters:
            mask = self.cluster_labels == cluster_id
            cluster_data = self.X_data[self.cluster_labels != -1][mask]
            
            cluster_stats[f'cluster_{cluster_id}'] = {
                'size': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(self.cluster_labels) * 100),
                'mean_values': cluster_data.mean(axis=0).tolist() if len(cluster_data) > 0 else []
            }
        
        # Правила решений
        decision_rules = self.get_decision_rules()
        
        report = {
            'model_accuracy': float(accuracy),
            'n_clusters': len(unique_clusters),
            'tree_depth': int(self.decision_tree.get_depth()),
            'n_leaves': int(self.decision_tree.get_n_leaves()),
            'feature_importance': feature_importance.to_dict('records'),
            'cluster_statistics': cluster_stats,
            'decision_rules': decision_rules,
            'top_3_features': feature_importance.head(3)['feature'].tolist()
        }
        
        return report
    
    def explain_sample_prediction(self, sample_idx):
        """
        Объяснение предсказания для конкретного образца.
        
        Parameters:
        -----------
        sample_idx : int
            Индекс образца для объяснения
            
        Returns:
        --------
        dict : объяснение предсказания
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        
        # Получение пути решения для образца
        sample = self.X_data[sample_idx:sample_idx+1]
        predicted_cluster = self.decision_tree.predict(sample)[0]
        
        # Путь в дереве
        leaf_id = self.decision_tree.decision_path(sample).toarray()[0]
        feature_path = []
        
        # Извлечение пути решения (упрощенная версия)
        tree_structure = self.decision_tree.tree_
        
        explanation = {
            'sample_index': sample_idx,
            'predicted_cluster': int(predicted_cluster),
            'confidence': float(np.max(self.decision_tree.predict_proba(sample))),
            'feature_values': dict(zip(self.feature_names, sample[0])),
            'decision_path_summary': f"Образец отнесен к кластеру {predicted_cluster}"
        }
        
        return explanation
    
    def plot_cluster_characteristics(self, figsize=(15, 10)):
        """
        Визуализация характеристик кластеров.
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Размер фигуры
            
        Returns:
        --------
        matplotlib figure
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        
        # Подготовка данных для визуализации
        df_viz = pd.DataFrame(self.X_data[self.cluster_labels != -1], columns=self.feature_names)
        df_viz['cluster'] = self.cluster_labels[self.cluster_labels != -1]
        
        # Топ признаков по важности
        top_features = self.get_feature_importance(top_n=6)['feature'].tolist()
        
        # Создание subplot'ов
        n_features = len(top_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, feature in enumerate(top_features):
            if i < len(axes):
                sns.boxplot(data=df_viz, x='cluster', y=feature, ax=axes[i])
                axes[i].set_title(f'Распределение {feature} по кластерам')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Скрытие лишних subplot'ов
        for j in range(len(top_features), len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Характеристики кластеров по ключевым признакам')
        plt.tight_layout()
        plt.show()
        
        return fig


def explain_clustering_results(X, cluster_labels, feature_names=None, 
                             max_depth=5, generate_report=True, plot_results=True):
    """
    Утилитарная функция для полного объяснения результатов кластеризации.
    
    Parameters:
    -----------
    X : np.ndarray or pd.DataFrame
        Исходные данные
    cluster_labels : np.ndarray
        Метки кластеров
    feature_names : list, optional
        Названия признаков
    max_depth : int, default=5
        Максимальная глубина дерева
    generate_report : bool, default=True
        Генерировать ли подробный отчет
    plot_results : bool, default=True
        Создавать ли визуализации
        
    Returns:
    --------
    dict : результаты объяснения
    """
    # Создание и обучение объяснителя
    explainer = ClusterExplainer(max_depth=max_depth)
    explainer.fit(X, cluster_labels, feature_names)
    
    results = {
        'explainer': explainer,
        'feature_importance': explainer.get_feature_importance(),
        'decision_rules': explainer.get_decision_rules()
    }
    
    if generate_report:
        results['interpretation_report'] = explainer.generate_cluster_interpretation_report()
    
    if plot_results:
        print("Создание визуализаций...")
        
        # График важности признаков
        explainer.plot_feature_importance()
        
        # Дерево решений
        if len(np.unique(cluster_labels[cluster_labels != -1])) <= 10:
            explainer.plot_decision_tree()
        else:
            print("Слишком много кластеров для визуализации дерева")
        
        # Характеристики кластеров
        explainer.plot_cluster_characteristics()
    
    return results


if __name__ == "__main__":
    # Пример использования
    print("Модуль интерпретации кластеров через DecisionTree")
    print("Возможности:")
    print("- Построение DecisionTree на результатах кластеризации")
    print("- Анализ важности признаков (feature importance)")
    print("- Визуализация дерева решений и характеристик кластеров")
    print("- Генерация текстового отчета по ключевым разделениям")
    print("- Объяснение предсказания для отдельных образцов")