"""
Модуль исследовательского анализа данных (EDA) для недвижимости
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def exploratory_data_analysis(df, output_dir='.'):
    """
    Проводит разведочный анализ данных и создает визуализации.
    
    Анализирует распределения признаков, корреляции между переменными,
    зависимость цены от различных факторов. Сохраняет два файла:
    - eda_visualizations.png: основные графики распределений и зависимостей
    - correlation_matrix.png: матрица корреляций между числовыми признаками
    
    Args:
        df: Датасет с данными о недвижимости (должен содержать колонку 'price')
        output_dir: Папка для сохранения графиков (по умолчанию текущая)
    """
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    print("\n1. Информация о датасете:")
    print(f"   Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"\n2. Первые 5 строк:")
    print(df.head())
    
    print("\n3. Статистика:")
    print(df.describe())
    
    print("\n4. Пропущенные значения:")
    print(df.isnull().sum())
    
    print("\n5. Статистика целевой переменной (цена):")
    print(f"   Средняя цена: ${df['price'].mean():,.0f}")
    print(f"   Медианная цена: ${df['price'].median():,.0f}")
    print(f"   Мин. цена: ${df['price'].min():,.0f}")
    print(f"   Макс. цена: ${df['price'].max():,.0f}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Exploratory Data Analysis - House Prices', fontsize=16, fontweight='bold')
    
    axes[0, 0].hist(df['price'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Распределение цен на недвижимость')
    axes[0, 0].set_xlabel('Цена ($)')
    axes[0, 0].set_ylabel('Количество')
    axes[0, 0].ticklabel_format(style='plain', axis='x')
    
    axes[0, 1].scatter(df['area'], df['price'], alpha=0.5, color='#2ecc71', s=20)
    axes[0, 1].set_title('Зависимость цены от площади')
    axes[0, 1].set_xlabel('Площадь (кв.м)')
    axes[0, 1].set_ylabel('Цена ($)')
    axes[0, 1].ticklabel_format(style='plain', axis='y')
    
    sns.boxplot(data=df, x='bedrooms', y='price', ax=axes[0, 2], palette='viridis')
    axes[0, 2].set_title('Цена в зависимости от количества спален')
    axes[0, 2].set_xlabel('Количество спален')
    axes[0, 2].set_ylabel('Цена ($)')
    axes[0, 2].ticklabel_format(style='plain', axis='y')
    
    sns.boxplot(data=df, x='airconditioning', y='price', ax=axes[1, 0], palette=['#e74c3c', '#2ecc71'])
    axes[1, 0].set_title('Влияние кондиционера на цену')
    axes[1, 0].set_xlabel('Кондиционер')
    axes[1, 0].set_ylabel('Цена ($)')
    axes[1, 0].ticklabel_format(style='plain', axis='y')
    
    sns.boxplot(data=df, x='furnishingstatus', y='price', ax=axes[1, 1], palette='Set2')
    axes[1, 1].set_title('Цена в зависимости от меблировки')
    axes[1, 1].set_xlabel('Меблировка')
    axes[1, 1].set_ylabel('Цена ($)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].ticklabel_format(style='plain', axis='y')
    
    sns.boxplot(data=df, x='parking', y='price', ax=axes[1, 2], palette='coolwarm')
    axes[1, 2].set_title('Влияние количества парковочных мест на цену')
    axes[1, 2].set_xlabel('Парковочные места')
    axes[1, 2].set_ylabel('Цена ($)')
    axes[1, 2].ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'eda_visualizations.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Визуализации сохранены в '{output_path}'")
    plt.close()
    
    plt.figure(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Корреляционная матрица признаков', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Корреляционная матрица сохранена в '{output_path}'")
    plt.close()

