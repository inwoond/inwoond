"""
Модуль предобработки данных для металлургических данных.

Автоматическое определение типов признаков и выбор стратегии кодирования:
- Числовые признаки: StandardScaler или MinMaxScaler
- Категориальные признаки:
  * ≤20 уникальных значений → OneHotEncoder
  * >20 уникальных значений → LabelEncoder
  * Редкие категории (>5-10%) → объединение в "Other"
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')


class MetallurgyPreprocessor:
    """
    Класс для автоматической предобработки металлургических данных.
    """
    
    def __init__(self, numerical_scaler='standard', rare_threshold=0.05):
        """
        Инициализация препроцессора.
        
        Parameters:
        -----------
        numerical_scaler : str, default='standard'
            Тип скейлера: 'standard' или 'minmax'
        rare_threshold : float, default=0.05
            Порог для объединения редких категорий (5% по умолчанию)
        """
        self.numerical_scaler = numerical_scaler
        self.rare_threshold = rare_threshold
        self.numeric_features = []
        self.categorical_features = []
        self.onehot_features = []
        self.label_features = []
        self.preprocessor = None
        self.is_fitted = False
        
    def detect_feature_types(self, df):
        """
        Автоматическое определение типов признаков.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходный датафрейм
            
        Returns:
        --------
        dict : словарь с типами признаков
        """
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Дополнительная проверка числовых признаков, которые могут быть категориальными
        potentially_categorical = []
        for col in numeric_features:
            unique_values = df[col].nunique()
            if unique_values <= 10 and df[col].dtype in ['int64', 'int32']:
                potentially_categorical.append(col)
        
        self.numeric_features = [col for col in numeric_features if col not in potentially_categorical]
        self.categorical_features = categorical_features + potentially_categorical
        
        return {
            'numeric': self.numeric_features,
            'categorical': self.categorical_features,
            'potentially_categorical': potentially_categorical
        }
    
    def _process_rare_categories(self, df, column, threshold=None):
        """
        Объединение редких категорий в группу "Other".
        
        Parameters:
        -----------
        df : pd.DataFrame
            Датафрейм
        column : str
            Название столбца
        threshold : float, optional
            Порог для редких категорий
            
        Returns:
        --------
        pd.Series : обработанный столбец
        """
        if threshold is None:
            threshold = self.rare_threshold
            
        value_counts = df[column].value_counts(normalize=True)
        rare_categories = value_counts[value_counts < threshold].index
        
        processed_column = df[column].copy()
        processed_column = processed_column.replace(rare_categories, 'Other')
        
        return processed_column
    
    def analyze_categorical_features(self, df):
        """
        Анализ категориальных признаков для выбора стратегии кодирования.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходный датафрейм
            
        Returns:
        --------
        dict : результаты анализа
        """
        analysis = {}
        
        for col in self.categorical_features:
            unique_count = df[col].nunique()
            
            # Обработка редких категорий
            processed_col = self._process_rare_categories(df, col)
            unique_after_processing = processed_col.nunique()
            
            analysis[col] = {
                'unique_before': unique_count,
                'unique_after': unique_after_processing,
                'encoding_strategy': 'onehot' if unique_after_processing <= 20 else 'label'
            }
            
            if unique_after_processing <= 20:
                self.onehot_features.append(col)
            else:
                self.label_features.append(col)
                
        return analysis
    
    def fit(self, df):
        """
        Обучение препроцессора на данных.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Тренировочные данные
            
        Returns:
        --------
        self
        """
        # Определение типов признаков
        self.detect_feature_types(df)
        
        # Анализ категориальных признаков
        self.analyze_categorical_features(df)
        
        # Обработка редких категорий для всех категориальных признаков
        df_processed = df.copy()
        for col in self.categorical_features:
            df_processed[col] = self._process_rare_categories(df, col)
        
        # Создание трансформеров
        transformers = []
        
        # Числовые признаки
        if self.numeric_features:
            if self.numerical_scaler == 'standard':
                numeric_transformer = StandardScaler()
            else:
                numeric_transformer = MinMaxScaler()
            transformers.append(('num', numeric_transformer, self.numeric_features))
        
        # OneHot кодирование
        if self.onehot_features:
            onehot_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            transformers.append(('onehot', onehot_transformer, self.onehot_features))
        
        # Label кодирование
        if self.label_features:
            label_transformer = LabelEncoder()
            # Для множественных столбцов создаем отдельные трансформеры
            for col in self.label_features:
                transformers.append((f'label_{col}', label_transformer, [col]))
        
        # Создание ColumnTransformer
        if transformers:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='drop'  # Отбрасываем необработанные столбцы
            )
            
            self.preprocessor.fit(df_processed)
        
        self.is_fitted = True
        return self
    
    def transform(self, df):
        """
        Трансформация данных.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Данные для трансформации
            
        Returns:
        --------
        np.ndarray : трансформированные данные
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor должен быть обучен перед трансформацией")
        
        # Обработка редких категорий
        df_processed = df.copy()
        for col in self.categorical_features:
            df_processed[col] = self._process_rare_categories(df, col)
        
        # Трансформация
        if self.preprocessor is not None:
            X_transformed = self.preprocessor.transform(df_processed)
        else:
            X_transformed = np.array([]).reshape(len(df), 0)
        
        return X_transformed
    
    def fit_transform(self, df):
        """
        Обучение и трансформация за один вызов.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходные данные
            
        Returns:
        --------
        np.ndarray : трансформированные данные
        """
        return self.fit(df).transform(df)
    
    def get_feature_names(self):
        """
        Получение названий признаков после трансформации.
        
        Returns:
        --------
        list : список названий признаков
        """
        if not self.is_fitted or self.preprocessor is None:
            return []
        
        try:
            return self.preprocessor.get_feature_names_out()
        except AttributeError:
            # Для старых версий sklearn
            feature_names = []
            feature_names.extend(self.numeric_features)
            
            # OneHot features
            if hasattr(self.preprocessor, 'named_transformers_'):
                if 'onehot' in self.preprocessor.named_transformers_:
                    onehot_names = self.preprocessor.named_transformers_['onehot'].get_feature_names_out(self.onehot_features)
                    feature_names.extend(onehot_names)
            
            # Label features
            feature_names.extend(self.label_features)
            
            return feature_names
    
    def get_preprocessing_report(self):
        """
        Генерация отчета о предобработке.
        
        Returns:
        --------
        dict : отчет о предобработке
        """
        if not self.is_fitted:
            return {"error": "Preprocessor не обучен"}
        
        report = {
            "numerical_scaler": self.numerical_scaler,
            "rare_threshold": self.rare_threshold,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "onehot_features": self.onehot_features,
            "label_features": self.label_features,
            "total_features_after_processing": len(self.get_feature_names())
        }
        
        return report


def load_and_preprocess_data(file_path, target_column=None, **preprocessing_params):
    """
    Утилитарная функция для загрузки и предобработки данных.
    
    Parameters:
    -----------
    file_path : str
        Путь к файлу данных (CSV или XLSX)
    target_column : str, optional
        Название целевой переменной (исключается из предобработки)
    **preprocessing_params : dict
        Параметры для MetallurgyPreprocessor
        
    Returns:
    --------
    tuple : (X_processed, preprocessor, original_df)
    """
    # Загрузка данных
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Поддерживаются только CSV и Excel файлы")
    
    # Исключение целевой переменной
    if target_column and target_column in df.columns:
        features_df = df.drop(columns=[target_column])
    else:
        features_df = df.copy()
    
    # Предобработка
    preprocessor = MetallurgyPreprocessor(**preprocessing_params)
    X_processed = preprocessor.fit_transform(features_df)
    
    return X_processed, preprocessor, df


if __name__ == "__main__":
    # Пример использования
    print(" Модуль предобработки металлургических данных")
    print("Поддерживаемые возможности:")
    print("- Автоматическое определение типов признаков")
    print("- OneHot кодирование для категорий ≤20 значений")
    print("- Label кодирование для категорий >20 значений") 
    print("- Объединение редких категорий в 'Other'")
    print("- StandardScaler/MinMaxScaler для числовых признаков")