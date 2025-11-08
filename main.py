"""
House Price Prediction - ML Project
Главный файл для запуска анализа и предсказания цен на недвижимость
"""

from src.data_loader import load_data
from src.eda import exploratory_data_analysis
from src.model_training import prepare_data, train_models, feature_importance_analysis

def main():
    """Основная функция запуска проекта"""
    print("\n" + "="*60)
    print("HOUSE PRICE PREDICTION - ML PROJECT")
    print("="*60 + "\n")
    
    print("Загрузка данных...")
    df = load_data()
    
    exploratory_data_analysis(df, output_dir='results')
    
    print("\nПодготовка данных для моделирования...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)
    
    results, best_model_name = train_models(X_train, X_test, y_train, y_test, output_dir='results')
    
    # Для tree-based моделей можно посмотреть важность признаков
    tree_models = ['Random Forest', 'XGBoost', 'Random Forest (Tuned)', 'XGBoost (Tuned)']
    for model_name in tree_models:
        if model_name in results:
            feature_importance_analysis(results[model_name]['model'], feature_names, output_dir='results')
            break
    
    print("\n" + "="*60)
    print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
    print("="*60)
    print("\nСозданные файлы:")
    print("  - results/eda_visualizations.png")
    print("  - results/correlation_matrix.png")
    print("  - results/model_performance.png")
    print("  - results/residuals_analysis.png")
    print("  - results/feature_importance.png")
    print("  - models/best_model.pkl")

if __name__ == "__main__":
    main()

