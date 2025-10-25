"""
Модуль кластеризации для металлургических данных.

Поддерживает:
- Agglomerative Clustering с дендрограммами
- DBSCAN с обработкой выбросов (label = -1)
- FAISS для больших данных (до 100k строк)
- Автоматическое определение оптимального количества кластеров
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# FAISS для быстрой кластеризации
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# NMSLIB для поиска ближайших соседей  
try:
    import nmslib
    NMSLIB_AVAILABLE = True
except ImportError:
    NMSLIB_AVAILABLE = False


class MetallurgyClustering:
    """
    Класс для кластеризации металлургических данных с поддержкой различных алгоритмов.
    """
    
    def __init__(self):
        self.algorithm = None
        self.n_clusters = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.fitted_model = None
        self.X_fitted = None
        
    def find_optimal_clusters(self, X, method='silhouette', k_range=range(2, 11), algorithm='kmeans'):
        """
        Автоматическое определение оптимального количества кластеров.
        
        Parameters:
        -----------
        X : np.ndarray
            Данные для кластеризации
        method : str, default='silhouette'
            Метод определения: 'elbow', 'silhouette'
        k_range : range, default=range(2, 11)
            Диапазон для поиска оптимального k
        algorithm : str, default='kmeans'
            Алгоритм для определения оптимального k
            
        Returns:
        --------
        dict : результаты анализа оптимального количества кластеров
        """
        results = {'k_values': [], 'scores': [], 'method': method}
        
        for k in k_range:
            if algorithm == 'kmeans':
                if FAISS_AVAILABLE and X.shape[0] > 10000:
                    # Используем FAISS для больших данных
                    labels = self._faiss_kmeans(X, k)
                else:
                    # Используем sklearn KMeans
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X)
            elif algorithm == 'agglomerative':
                agg = AgglomerativeClustering(n_clusters=k)
                labels = agg.fit_predict(X)
            else:
                raise ValueError(f"Неподдерживаемый алгоритм: {algorithm}")
            
            if method == 'silhouette':
                if len(np.unique(labels)) > 1:  # Нужно минимум 2 кластера
                    score = silhouette_score(X, labels)
                    results['scores'].append(score)
                    results['k_values'].append(k)
            elif method == 'elbow':
                if algorithm == 'kmeans':
                    # Для метода локтя вычисляем WCSS (Within-Cluster Sum of Squares)
                    wcss = 0
                    for cluster_id in np.unique(labels):
                        cluster_points = X[labels == cluster_id]
                        if len(cluster_points) > 0:
                            cluster_center = np.mean(cluster_points, axis=0)
                            wcss += np.sum((cluster_points - cluster_center) ** 2)
                    results['scores'].append(wcss)
                    results['k_values'].append(k)
        
        # Определение оптимального k
        if method == 'silhouette':
            optimal_idx = np.argmax(results['scores'])
            optimal_k = results['k_values'][optimal_idx]
        elif method == 'elbow':
            # Простая эвристика для метода локтя
            scores = np.array(results['scores'])
            diffs = np.diff(scores)
            second_diffs = np.diff(diffs)
            if len(second_diffs) > 0:
                optimal_idx = np.argmax(second_diffs) + 2  # +2 из-за двойного diff
                optimal_k = results['k_values'][min(optimal_idx, len(results['k_values']) - 1)]
            else:
                optimal_k = results['k_values'][0]
        
        results['optimal_k'] = optimal_k
        results['optimal_score'] = results['scores'][results['k_values'].index(optimal_k)] if optimal_k in results['k_values'] else None
        
        return results
    
    def _faiss_kmeans(self, X, k, niter=20, verbose=False):
        """
        Кластеризация с помощью FAISS для больших данных.
        
        Parameters:
        -----------
        X : np.ndarray
            Данные для кластеризации
        k : int
            Количество кластеров
        niter : int, default=20
            Количество итераций
        verbose : bool, default=False
            Вывод отладочной информации
            
        Returns:
        --------
        np.ndarray : метки кластеров
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS не установлен. Используйте: pip install faiss-cpu")
        
        # Преобразование в float32 (требование FAISS)
        X_faiss = X.astype(np.float32)
        
        # Создание и обучение FAISS KMeans
        d = X_faiss.shape[1]  # размерность
        kmeans = faiss.Kmeans(d=d, k=k, niter=niter, verbose=verbose)
        kmeans.train(X_faiss)
        
        # Получение меток кластеров
        distances, labels = kmeans.index.search(X_faiss, 1)
        
        return labels.flatten()
    
    def fit_agglomerative(self, X, n_clusters=None, linkage='ward', **kwargs):
        """
        Агломеративная кластеризация.
        
        Parameters:
        -----------
        X : np.ndarray
            Данные для кластеризации
        n_clusters : int, optional
            Количество кластеров (автоопределение если None)
        linkage : str, default='ward'
            Метод связи: 'ward', 'complete', 'average', 'single'
        **kwargs : dict
            Дополнительные параметры для AgglomerativeClustering
            
        Returns:
        --------
        self
        """
        self.algorithm = 'agglomerative'
        
        # Автоматическое определение количества кластеров
        if n_clusters is None:
            opt_results = self.find_optimal_clusters(X, algorithm='agglomerative')
            n_clusters = opt_results['optimal_k']
            print(f"Автоматически определено оптимальное количество кластеров: {n_clusters}")
        
        self.n_clusters = n_clusters
        
        # Обучение модели
        self.fitted_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            **kwargs
        )
        
        self.labels_ = self.fitted_model.fit_predict(X)
        self.X_fitted = X.copy()
        
        # Вычисление центров кластеров
        self.cluster_centers_ = np.array([
            X[self.labels_ == i].mean(axis=0) 
            for i in range(n_clusters)
        ])
        
        return self
    
    def fit_dbscan(self, X, eps=None, min_samples=None, **kwargs):
        """
        DBSCAN кластеризация с автоматическим подбором параметров.
        
        Parameters:
        -----------
        X : np.ndarray
            Данные для кластеризации
        eps : float, optional
            Радиус окрестности (автоопределение если None)
        min_samples : int, optional
            Минимальное количество точек в кластере
        **kwargs : dict
            Дополнительные параметры для DBSCAN
            
        Returns:
        --------
        self
        """
        self.algorithm = 'dbscan'
        
        # Автоматическое определение параметров
        if eps is None:
            # Простая эвристика для eps на основе k-ближайших соседей
            from sklearn.neighbors import NearestNeighbors
            k = min_samples if min_samples else 4
            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors_fit = neighbors.fit(X)
            distances, indices = neighbors_fit.kneighbors(X)
            distances = np.sort(distances, axis=0)
            distances = distances[:, k-1]  # k-ая ближайшая точка
            eps = np.percentile(distances, 90)  # 90-й перцентиль как eps
            
        if min_samples is None:
            min_samples = max(2, int(np.log(len(X))))  # Эвристика на основе размера данных
        
        print(f"Параметры DBSCAN: eps={eps:.4f}, min_samples={min_samples}")
        
        # Обучение модели
        self.fitted_model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        self.labels_ = self.fitted_model.fit_predict(X)
        self.X_fitted = X.copy()
        
        # Количество кластеров (исключая шум)
        unique_labels = np.unique(self.labels_)
        self.n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Вычисление центров кластеров (исключая выбросы)
        valid_clusters = [i for i in unique_labels if i != -1]
        if valid_clusters:
            self.cluster_centers_ = np.array([
                X[self.labels_ == i].mean(axis=0) 
                for i in valid_clusters
            ])
        else:
            self.cluster_centers_ = np.array([])
        
        return self
    
    def fit_faiss_kmeans(self, X, n_clusters=None, **kwargs):
        """
        FAISS K-means для больших данных.
        
        Parameters:
        -----------
        X : np.ndarray
            Данные для кластеризации  
        n_clusters : int, optional
            Количество кластеров
        **kwargs : dict
            Дополнительные параметры для FAISS
            
        Returns:
        --------
        self
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS не установлен. Используйте: pip install faiss-cpu")
        
        self.algorithm = 'faiss_kmeans'
        
        # Автоматическое определение количества кластеров
        if n_clusters is None:
            opt_results = self.find_optimal_clusters(X, algorithm='kmeans')
            n_clusters = opt_results['optimal_k']
            print(f"Автоматически определено оптимальное количество кластеров: {n_clusters}")
        
        self.n_clusters = n_clusters
        
        # Кластеризация
        self.labels_ = self._faiss_kmeans(X, n_clusters, **kwargs)
        self.X_fitted = X.copy()
        
        # Вычисление центров кластеров
        self.cluster_centers_ = np.array([
            X[self.labels_ == i].mean(axis=0) 
            for i in range(n_clusters)
        ])
        
        return self
    
    def calculate_metrics(self, exclude_noise=True):
        """
        Вычисление метрик качества кластеризации.
        
        Parameters:
        -----------
        exclude_noise : bool, default=True
            Исключать ли выбросы из расчета (для DBSCAN)
            
        Returns:
        --------
        dict : словарь с метриками качества
        """
        if self.labels_ is None or self.X_fitted is None:
            return {"error": "Модель не обучена"}
        
        # Подготовка данных (исключение выбросов для DBSCAN)
        if exclude_noise and self.algorithm == 'dbscan':
            mask = self.labels_ != -1
            X_clean = self.X_fitted[mask]
            labels_clean = self.labels_[mask]
        else:
            X_clean = self.X_fitted
            labels_clean = self.labels_
        
        # Проверка наличия кластеров
        unique_labels = np.unique(labels_clean)
        if len(unique_labels) < 2:
            return {"error": "Недостаточно кластеров для расчета метрик"}
        
        metrics = {}
        
        try:
            metrics['silhouette_score'] = silhouette_score(X_clean, labels_clean)
        except Exception as e:
            metrics['silhouette_score'] = f"Ошибка: {str(e)}"
        
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_clean, labels_clean)
        except Exception as e:
            metrics['davies_bouldin_score'] = f"Ошибка: {str(e)}"
        
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_clean, labels_clean)
        except Exception as e:
            metrics['calinski_harabasz_score'] = f"Ошибка: {str(e)}"
        
        # Дополнительная информация
        metrics['n_clusters'] = self.n_clusters
        metrics['n_samples'] = len(X_clean)
        metrics['algorithm'] = self.algorithm
        
        if self.algorithm == 'dbscan':
            n_noise = np.sum(self.labels_ == -1)
            metrics['n_noise_points'] = n_noise
            metrics['noise_percentage'] = n_noise / len(self.labels_) * 100
        
        return metrics
    
    def plot_dendrogram(self, X, **kwargs):
        """
        Построение дендрограммы для иерархической кластеризации.
        
        Parameters:
        -----------
        X : np.ndarray
            Данные для построения дендрограммы
        **kwargs : dict
            Параметры для dendrogram
            
        Returns:
        --------
        matplotlib figure
        """
        if X.shape[0] > 1000:
            print(f"⚠️  Внимание: большое количество точек ({X.shape[0]}). Дендрограмма может быть нечитаемой.")
        
        # Вычисление матрицы связей
        linkage_matrix = linkage(X, method='ward')
        
        # Построение дендрограммы
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix, **kwargs)
        plt.title('Дендрограмма иерархической кластеризации')
        plt.xlabel('Индекс образца')
        plt.ylabel('Расстояние')
        plt.show()
        
        return plt.gcf()
    
    def plot_clusters_pca(self, X=None, labels=None, title=None):
        """
        Визуализация кластеров с помощью PCA проекции.
        
        Parameters:
        -----------
        X : np.ndarray, optional
            Данные (использует self.X_fitted если None)
        labels : np.ndarray, optional
            Метки кластеров (использует self.labels_ если None)
        title : str, optional
            Заголовок графика
            
        Returns:
        --------
        matplotlib figure
        """
        if X is None:
            X = self.X_fitted
        if labels is None:
            labels = self.labels_
        
        if X is None or labels is None:
            raise ValueError("Данные или метки не определены")
        
        # PCA для снижения размерности
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Построение scatter plot
        plt.figure(figsize=(12, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, (label, color) in enumerate(zip(unique_labels, colors)):
            if label == -1:
                # Выбросы (для DBSCAN)
                plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], 
                           c='black', marker='x', s=50, label='Выбросы', alpha=0.6)
            else:
                plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], 
                           c=[color], marker='o', s=50, label=f'Кластер {label}', alpha=0.7)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсии)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсии)')
        
        if title is None:
            title = f'Визуализация кластеров ({self.algorithm.upper()}) - PCA проекция'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return plt.gcf()


def compare_clustering_methods(X, methods=['agglomerative', 'dbscan'], n_clusters=None):
    """
    Сравнение различных методов кластеризации.
    
    Parameters:
    -----------
    X : np.ndarray
        Данные для кластеризации
    methods : list, default=['agglomerative', 'dbscan']
        Список методов для сравнения
    n_clusters : int, optional
        Количество кластеров (где применимо)
        
    Returns:
    --------
    dict : результаты сравнения методов
    """
    results = {}
    
    for method in methods:
        print(f"\n🔄 Выполнение кластеризации: {method.upper()}")
        
        clustering = MetallurgyClustering()
        
        try:
            if method == 'agglomerative':
                clustering.fit_agglomerative(X, n_clusters=n_clusters)
            elif method == 'dbscan':
                clustering.fit_dbscan(X)
            elif method == 'faiss_kmeans':
                if FAISS_AVAILABLE:
                    clustering.fit_faiss_kmeans(X, n_clusters=n_clusters)
                else:
                    print("⚠️  FAISS недоступен, пропускаем FAISS K-means")
                    continue
            else:
                print(f"⚠️  Неизвестный метод: {method}")
                continue
            
            # Вычисление метрик
            metrics = clustering.calculate_metrics()
            
            results[method] = {
                'clustering_object': clustering,
                'metrics': metrics,
                'n_clusters': clustering.n_clusters,
                'labels': clustering.labels_
            }
            
            print(f"✅ {method.upper()}: {clustering.n_clusters} кластеров")
            if 'silhouette_score' in metrics:
                print(f"   Silhouette Score: {metrics['silhouette_score']:.3f}")
            
        except Exception as e:
            print(f"❌ Ошибка в {method}: {str(e)}")
            results[method] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Пример использования
    print("🧠 Модуль кластеризации металлургических данных")
    print("Поддерживаемые алгоритмы:")
    print("- Agglomerative Clustering с дендрограммами")
    print("- DBSCAN с обработкой выбросов")
    if FAISS_AVAILABLE:
        print("- FAISS K-means для больших данных")
    else:
        print("- FAISS K-means (не установлен)")
    print("\nМетрики качества:")
    print("- Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index")