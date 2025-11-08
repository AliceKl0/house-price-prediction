"""
Модуль обучения и оценки регрессионных моделей для предсказания цен на недвижимость
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle

from src.utils import feature_engineering

def _check_xgboost_available():
    """Проверяет, доступен ли XGBoost"""
    try:
        from xgboost import XGBRegressor
        return True, XGBRegressor
    except (ImportError, Exception) as e:
        return False, None

def prepare_data(df):
    """
    Подготавливает данные для обучения моделей.
    
    Выполняет полный пайплайн предобработки: создание новых признаков,
    кодирование категориальных переменных, разделение на train/test,
    стандартизацию признаков.
    
    Args:
        df: Исходный датасет с данными о недвижимости
    
    Returns:
        Кортеж из (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names)
        где scaler - обученный StandardScaler для масштабирования новых данных
    """
    df_processed = feature_engineering(df)
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    X = df_processed.drop('price', axis=1)
    y = df_processed['price']
    
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

def train_models(X_train, X_test, y_train, y_test, output_dir='.'):
    """
    Обучает несколько регрессионных моделей и выбирает лучшую с подбором гиперпараметров.
    
    Тестирует Linear Regression, Ridge, Random Forest и XGBoost (если доступен).
    Для лучшей модели выполняет GridSearchCV для оптимизации гиперпараметров.
    Сохраняет лучшую модель в models/best_model.pkl и создает визуализации.
    
    Args:
        X_train: Обучающая выборка признаков (уже масштабированная)
        X_test: Тестовая выборка признаков (уже масштабированная)
        y_train: Обучающая выборка целевой переменной (цены)
        y_test: Тестовая выборка целевой переменной (цены)
        output_dir: Папка для сохранения графиков и модели
    
    Returns:
        Кортеж (results_dict, best_model_name), где results_dict содержит
        результаты всех моделей с метриками и предсказаниями
    """
    print("\n" + "=" * 60)
    print("MACHINE LEARNING MODELS - REGRESSION")
    print("=" * 60)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    xgboost_available, XGBRegressor = _check_xgboost_available()
    if xgboost_available:
        models['XGBoost'] = XGBRegressor(random_state=42, n_jobs=-1)
    else:
        print("\n⚠ XGBoost недоступен (требуется OpenMP runtime). Продолжаем без XGBoost.")
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  RMSE: ${rmse:,.0f}")
        print(f"  MAE: ${mae:,.0f}")
        print(f"  R² Score: {r2:.4f}")
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        print(f"  Cross-Validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_scores': cv_scores
        }
    
    print("\n" + "-" * 60)
    print("Подбор гиперпараметров для лучшей модели...")
    best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
    print(f"Выбрана модель: {best_model_name}")
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    elif best_model_name == 'XGBoost' and xgboost_available:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1]
        }
        base_model = XGBRegressor(random_state=42, n_jobs=-1)
    elif best_model_name == 'Ridge Regression':
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }
        base_model = Ridge(random_state=42)
    else:
        print("  ⚠ Пропускаем GridSearch для этой модели")
        param_grid = None
        base_model = None
    
    if param_grid is not None and base_model is not None:
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"\nЛучшие параметры: {grid_search.best_params_}")
        print(f"Лучший CV R² score: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        y_pred_tuned = best_model.predict(X_test)
        rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
        mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
        r2_tuned = r2_score(y_test, y_pred_tuned)
        
        print(f"Test RMSE (tuned): ${rmse_tuned:,.0f}")
        print(f"Test MAE (tuned): ${mae_tuned:,.0f}")
        print(f"Test R² (tuned): {r2_tuned:.4f}")
        
        results[f'{best_model_name} (Tuned)'] = {
            'model': best_model,
            'y_pred': y_pred_tuned,
            'rmse': rmse_tuned,
            'mae': mae_tuned,
            'r2': r2_tuned,
            'cv_scores': None
        }
        
        model_path = os.path.join('models', 'best_model.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"\n✓ Лучшая модель сохранена в '{model_path}'")
    
    visualize_results(results, y_test, output_dir)
    
    final_best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
    print(f"\n{'='*60}")
    print(f"Лучшая модель: {final_best_model_name}")
    print(f"R² Score: {results[final_best_model_name]['r2']:.4f}")
    print(f"RMSE: ${results[final_best_model_name]['rmse']:,.0f}")
    print(f"{'='*60}")
    
    return results, final_best_model_name

def visualize_results(results, y_test, output_dir='.'):
    """
    Создает визуализации для сравнения моделей.
    
    Генерирует два графика:
    - model_performance.png: сравнение предсказаний с реальными значениями
    - residuals_analysis.png: анализ остатков для проверки качества модели
    
    Args:
        results: Словарь с результатами моделей
        y_test: Реальные значения цен для тестовой выборки
        output_dir: Папка для сохранения графиков
    """
    n_models = len(results)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Model Performance: Predictions vs Actual Prices', fontsize=16, fontweight='bold')
    
    for idx, (name, result) in enumerate(results.items()):
        axes[idx].scatter(y_test, result['y_pred'], alpha=0.5, s=20)
        min_val = min(y_test.min(), result['y_pred'].min())
        max_val = max(y_test.max(), result['y_pred'].max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[idx].set_xlabel('Реальная цена ($)', fontsize=10)
        axes[idx].set_ylabel('Предсказанная цена ($)', fontsize=10)
        axes[idx].set_title(f'{name}\nR² = {result["r2"]:.4f}, RMSE = ${result["rmse"]:,.0f}')
        axes[idx].ticklabel_format(style='plain', axis='both')
        axes[idx].grid(alpha=0.3)
    
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Результаты моделей сохранены в '{output_path}'")
    plt.close()
    
    best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
    best_result = results[best_model_name]
    residuals = y_test - best_result['y_pred']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Residuals Analysis - {best_model_name}', fontsize=14, fontweight='bold')
    
    axes[0].hist(residuals, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_title('Распределение остатков')
    axes[0].set_xlabel('Остаток (Реальная цена - Предсказанная цена)')
    axes[0].set_ylabel('Частота')
    axes[0].ticklabel_format(style='plain', axis='x')
    axes[0].grid(alpha=0.3)
    
    axes[1].scatter(best_result['y_pred'], residuals, alpha=0.5, s=20, color='#2ecc71')
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_title('Остатки vs Предсказанные значения')
    axes[1].set_xlabel('Предсказанная цена ($)')
    axes[1].set_ylabel('Остаток')
    axes[1].ticklabel_format(style='plain', axis='x')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'residuals_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Анализ остатков сохранен в '{output_path}'")
    plt.close()

def feature_importance_analysis(model, feature_names, output_dir='.'):
    """
    Анализирует важность признаков для tree-based моделей.
    
    Использует встроенный метод feature_importances_ для Random Forest или XGBoost.
    Создает горизонтальный бар-график с топ-15 признаками и выводит топ-10 в консоль.
    
    Args:
        model: Обученная модель с методом feature_importances_ (Random Forest, XGBoost)
        feature_names: Список названий признаков в том же порядке, что и в данных
        output_dir: Папка для сохранения графика feature_importance.png
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        top_n = min(15, len(indices))
        
        plt.figure(figsize=(10, 8))
        model_type = type(model).__name__
        plt.title(f'Топ-{top_n} важных признаков ({model_type})', fontsize=14, fontweight='bold')
        plt.barh(range(top_n), importances[indices[:top_n]], color='#3498db')
        plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
        plt.xlabel('Важность', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Анализ важности признаков сохранен в '{output_path}'")
        plt.close()
        
        print(f"\nТоп-10 важных признаков:")
        for i in range(min(10, len(indices))):
            print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

