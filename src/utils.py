"""
Утилиты и вспомогательные функции для работы с данными о недвижимости
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def feature_engineering(df):
    """
    Создает дополнительные признаки на основе существующих.
    
    Добавляет комбинированные признаки, которые помогают моделям
    лучше улавливать зависимости между характеристиками недвижимости.
    
    Args:
        df: Исходный датасет с базовыми признаками
    
    Returns:
        Датасет с добавленными признаками
    """
    df_fe = df.copy()
    
    df_fe['area_per_bedroom'] = df_fe['area'] / (df_fe['bedrooms'] + 1)
    df_fe['total_rooms'] = df_fe['bedrooms'] + df_fe['bathrooms']
    df_fe['area_per_story'] = df_fe['area'] / (df_fe['stories'] + 1)
    
    df_fe['comfort_score'] = (
        (df_fe['mainroad'] == 'yes').astype(int) +
        (df_fe['guestroom'] == 'yes').astype(int) +
        (df_fe['basement'] == 'yes').astype(int) +
        (df_fe['hotwaterheating'] == 'yes').astype(int) +
        (df_fe['airconditioning'] == 'yes').astype(int) +
        df_fe['parking'] +
        (df_fe['prefarea'] == 'yes').astype(int)
    )
    
    df_fe['premium_house'] = (
        ((df_fe['area'] > df_fe['area'].quantile(0.75)) & 
         (df_fe['comfort_score'] > df_fe['comfort_score'].quantile(0.75)))
    ).astype(int)
    
    df_fe['area_comfort_interaction'] = df_fe['area'] * df_fe['comfort_score'] / 100
    df_fe['area_parking_interaction'] = df_fe['area'] * df_fe['parking'] / 100
    
    furnishing_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
    df_fe['furnishing_encoded'] = df_fe['furnishingstatus'].map(furnishing_map)
    
    print(f"✓ Feature engineering: добавлено {len(df_fe.columns) - len(df.columns)} новых признаков")
    
    return df_fe

