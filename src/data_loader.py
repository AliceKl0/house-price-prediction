"""
Модуль загрузки данных о недвижимости
"""

import pandas as pd
import numpy as np

def load_data(data_path=None):
    """
    Загружает данные о недвижимости из CSV файла или создает синтетический датасет.
    
    Сначала пытается загрузить данные по указанному пути, затем проверяет
    стандартную директорию data/raw/. Если файлы не найдены, генерирует
    синтетические данные с реалистичными зависимостями.
    
    Args:
        data_path: Путь к CSV файлу. Если None, используется стандартный путь
                  или генерируются синтетические данные.
    
    Returns:
        DataFrame с колонками: area, bedrooms, bathrooms, stories, mainroad,
        guestroom, basement, hotwaterheating, airconditioning, parking,
        prefarea, furnishingstatus, price (целевая переменная)
    """
    if data_path:
        try:
            df = pd.read_csv(data_path)
            print(f"✓ Загружен реальный датасет из {data_path}")
            return df
        except FileNotFoundError:
            print(f"⚠ Файл {data_path} не найден, используем синтетические данные...")
    
    try:
        df = pd.read_csv('data/raw/housing_data.csv')
        print("✓ Загружен реальный датасет из data/raw/housing_data.csv")
        return df
    except FileNotFoundError:
        pass
    
    # Если реальных данных нет, создаем синтетические
    print("Генерация синтетических данных о недвижимости...")
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'area': np.random.normal(2000, 800, n_samples).clip(500, 5000).astype(int),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.15, 0.3, 0.3, 0.15, 0.05]),
        'bathrooms': np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.4, 0.3, 0.1]),
        'stories': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'mainroad': np.random.choice(['yes', 'no'], n_samples, p=[0.7, 0.3]),
        'guestroom': np.random.choice(['yes', 'no'], n_samples, p=[0.4, 0.6]),
        'basement': np.random.choice(['yes', 'no'], n_samples, p=[0.5, 0.5]),
        'hotwaterheating': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7]),
        'airconditioning': np.random.choice(['yes', 'no'], n_samples, p=[0.6, 0.4]),
        'parking': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
        'prefarea': np.random.choice(['yes', 'no'], n_samples, p=[0.4, 0.6]),
        'furnishingstatus': np.random.choice(['furnished', 'semi-furnished', 'unfurnished'], 
                                            n_samples, p=[0.3, 0.4, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Формируем целевую переменную (цена) на основе признаков
    base_price = 500000
    
    price = (
        base_price +
        df['area'] * 200 +
        df['bedrooms'] * 150000 +
        df['bathrooms'] * 100000 +
        df['stories'] * 50000 +
        (df['mainroad'] == 'yes') * 200000 +
        (df['guestroom'] == 'yes') * 80000 +
        (df['basement'] == 'yes') * 100000 +
        (df['hotwaterheating'] == 'yes') * 50000 +
        (df['airconditioning'] == 'yes') * 120000 +
        df['parking'] * 60000 +
        (df['prefarea'] == 'yes') * 150000 +
        (df['furnishingstatus'] == 'furnished') * 80000 +
        (df['furnishingstatus'] == 'semi-furnished') * 40000 +
        np.random.normal(0, 100000, n_samples)
    )
    
    # Добавляем нелинейные эффекты для более реалистичной модели
    price += (df['area'] * df['bedrooms']) * 50
    price += (df['airconditioning'] == 'yes') * (df['area'] > 2500) * 50000
    
    df['price'] = price.clip(200000, 10000000).astype(int)
    
    print(f"✓ Создан синтетический датасет: {df.shape[0]} записей, {df.shape[1]} признаков")
    
    return df

