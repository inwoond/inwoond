# Базовый класс для алгоритмов кластеризации

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Dict, Any

class BaseClusteringAlgorithm(ABC):
    """
    Базовый абстрактный класс для алгоритмов кластеризации.
    
    Этот класс определяет общий интерфейс для всех алгоритмов
    кластеризации в проекте.
    """
    
    def __init__(self, n_clusters: int = 2, **kwargs):
        """
        Инициализация алгоритма кластеризации.
        
        Args:
            n_clusters: Количество кластеров
            **kwargs: Дополнительные параметры алгоритма
        """
        self.n_clusters = n_clusters
        self.labels_ = None
        self.cluster_centers_ = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseClusteringAlgorithm':
        """
        Обучение алгоритма кластеризации.
        
        Args:
            X: Матрица признаков размера (n_samples, n_features)
            
        Returns:
            self: Обученный алгоритм
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание кластеров для новых данных.
        
        Args:
            X: Матрица признаков размера (n_samples, n_features)
            
        Returns:
            labels: Массив меток кластеров
        """
        pass
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Обучение и предсказание за один вызов.
        
        Args:
            X: Матрица признаков размера (n_samples, n_features)
            
        Returns:
            labels: Массив меток кластеров
        """
        return self.fit(X).predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Получение параметров алгоритма.
        
        Returns:
            params: Словарь с параметрами
        """
        params = {}
        for key, value in self.__dict__.items():
            if not key.endswith('_'):
                params[key] = value
        return params
    
    def set_params(self, **params) -> 'BaseClusteringAlgorithm':
        """
        Установка параметров алгоритма.
        
        Args:
            **params: Параметры для установки
            
        Returns:
            self: Алгоритм с обновленными параметрами
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self