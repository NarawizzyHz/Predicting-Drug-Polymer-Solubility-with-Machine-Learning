
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_ml_data():
    print("Machine Learning Model Training")
    print("Loading data...")
    
    try:
        X = pd.read_csv('X_ml_ready.csv')
        y = pd.read_csv('y_ml_ready.csv')
        feature_names = pd.read_csv('selected_features.csv')['selected_features'].values
        
        print("Data loaded")
        print(f"Features: {X.shape[1]}")
        print(f"Samples: {X.shape[0]}")
        print(f"Target range: {y['logS'].min():.2f} to {y['logS'].max():.2f}")
        
        return X.values, y['logS'].values, feature_names
    except FileNotFoundError as e:
        print(f"Error: Run feature_engineering.py first")
        return None, None, None

def split_data_stratified(X, y, test_size=0.2, random_state=42):
    print("\nSplitting data...")
    
    y_series = pd.Series(y)
    bins = [-12, -6, -4, -2, 2]
    labels = ['Very Poor', 'Poor', 'Moderate', 'Good']
    y_classes = pd.cut(y_series, bins=bins, labels=labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y_classes
    )
    
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test

def initialize_models():
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=0.01, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        'Support Vector Regression': SVR(kernel='rbf', C=100, gamma=0.1),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
        'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=42),
        'MLP Regressor': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    print(f"Initialized {len(models)} ML models")
    return models

def evaluate_model_cv(model, X_train, y_train, model_name, cv_folds=5):
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
    rmse_scores = -cross_val_score(model, X_train, y_train, cv=cv, 
                                   scoring='neg_root_mean_squared_error', n_jobs=-1)
    mae_scores = -cross_val_score(model, X_train, y_train, cv=cv, 
                                  scoring='neg_mean_absolute_error', n_jobs=-1)
    
    return {
        'Model': model_name,
        'CV_R2_mean': np.mean(r2_scores),
        'CV_R2_std': np.std(r2_scores),
        'CV_RMSE_mean': np.mean(rmse_scores),
        'CV_RMSE_std': np.std(rmse_scores),
        'CV_MAE_mean': np.mean(mae_scores),
        'CV_MAE_std': np.std(mae_scores)
    }

def train_and_evaluate_all_models(models, X_train, X_test, y_train, y_test):
    print("\nTraining and evaluating models")
    
    results = []
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}")
        
        try:
            cv_results = evaluate_model_cv(model, X_train, y_train, model_name)
            model.fit(X_train, y_train)
            trained_models[model_name] = model
            
            y_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            
            model_results = {
                **cv_results,
                'Test_R2': test_r2,
                'Test_RMSE': test_rmse,
                'Test_MAE': test_mae,
                'Test_R2_adj': 1 - (1 - test_r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
            }
            
            results.append(model_results)
            print(f"  CV R²: {cv_results['CV_R2_mean']:.4f} ± {cv_results['CV_R2_std']:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_R2', ascending=False)
    
    print("\nModel Performance Summary")
    
    display_cols = ['Model', 'CV_R2_mean', 'CV_R2_std', 'Test_R2', 'Test_RMSE', 'Test_MAE']
    formatted_df = results_df[display_cols].copy()
    formatted_df.columns = ['Model', 'CV R² (Mean)', 'CV R² (Std)', 'Test R²', 'Test RMSE', 'Test MAE']
    print(formatted_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    return results_df, trained_models

def create_model_comparison_plot(results_df):
    print("\nCreating comparison plot")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    plot_df = results_df.sort_values('Test_R2', ascending=True)
    
    axes[0, 0].barh(range(len(plot_df)), plot_df['Test_R2'], color='steelblue')
    axes[0, 0].set_yticks(range(len(plot_df)))
    axes[0, 0].set_yticklabels(plot_df['Model'])
    axes[0, 0].set_xlabel('Test R² Score')
    axes[0, 0].set_title('Test Set Performance (R²)')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    for i, (bar, r2) in enumerate(zip(axes[0, 0].patches, plot_df['Test_R2'])):
        axes[0, 0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{r2:.3f}', va='center')
    
    axes[0, 1].barh(range(len(plot_df)), plot_df['Test_RMSE'], color='lightcoral')
    axes[0, 1].set_yticks(range(len(plot_df)))
    axes[0, 1].set_yticklabels([])
    axes[0, 1].set_xlabel('Test RMSE (logS units)')
    axes[0, 1].set_title('Test Set Error (RMSE)')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    for i, (bar, rmse) in enumerate(zip(axes[0, 1].patches, plot_df['Test_RMSE'])):
        axes[0, 1].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{rmse:.3f}', va='center')
    
    x_pos = np.arange(len(plot_df))
    axes[1, 0].errorbar(plot_df['CV_R2_mean'], x_pos, 
                       xerr=plot_df['CV_R2_std'], 
                       fmt='o', color='darkgreen', 
                       capsize=5, markersize=8)
    axes[1, 0].set_yticks(x_pos)
    axes[1, 0].set_yticklabels(plot_df['Model'])
    axes[1, 0].set_xlabel('CV R² Score (Mean ± Std)')
    axes[1, 0].set_title('Cross-Validation Performance')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3)
    
    model_types = {
        'Linear': ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet'],
        'Tree-based': ['Decision Tree', 'Random Forest', 'Gradient Boosting'],
        'Ensemble': ['XGBoost', 'LightGBM', 'AdaBoost'],
        'Other': ['Support Vector Regression', 'K-Nearest Neighbors', 'MLP Regressor']
    }
    
    colors = {'Linear': 'blue', 'Tree-based': 'green', 'Ensemble': 'orange', 'Other': 'purple'}
    
    for model_type, model_list in model_types.items():
        type_results = plot_df[plot_df['Model'].isin(model_list)]
        if len(type_results) > 0:
            axes[1, 1].scatter(type_results['Test_RMSE'], type_results['Test_R2'],
                             s=100, alpha=0.7, label=model_type, color=colors[model_type])
    
    axes[1, 1].set_xlabel('Test RMSE')
    axes[1, 1].set_ylabel('Test R²')
    axes[1, 1].set_title('Error vs Accuracy by Model Type')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Model comparison plot saved")

def analyze_best_model_predictions(best_model, X_test, y_test, feature_names):
    print("\nAnalyzing best model performance")
    
    y_pred = best_model.predict(X_test)
    residuals = y_test - y_pred
    abs_residuals = np.abs(residuals)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Best Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Predicted vs Experimental (TOP-LEFT)
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=20)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                   'r--', lw=2, label='Perfect')
    axes[0, 0].set_xlabel('Experimental logS')
    axes[0, 0].set_ylabel('Predicted logS')
    axes[0, 0].set_title('Predicted vs Experimental')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add R² text to plot
    r2 = r2_score(y_test, y_pred)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Residual Analysis (TOP-MIDDLE)
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted logS')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Analysis')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add MAE text to plot
    mae = mean_absolute_error(y_test, y_pred)
    axes[0, 1].text(0.05, 0.95, f'MAE = {mae:.3f}', transform=axes[0, 1].transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Residual Distribution (TOP-RIGHT)
    axes[0, 2].hist(residuals, bins=30, edgecolor='black', alpha=0.7, density=True)
    axes[0, 2].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0, 2].set_xlabel('Residuals (logS units)')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Residual Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add normal distribution overlay
    from scipy import stats
    x = np.linspace(residuals.min(), residuals.max(), 100)
    pdf = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    axes[0, 2].plot(x, pdf, 'r-', lw=2, alpha=0.7, label='Normal fit')
    axes[0, 2].legend()
    
    # === IMPROVED BOTTOM ROW ===
    
    # 4. Error vs True Value (BOTTOM-LEFT) - REPLACES Error by Class
    scatter = axes[1, 0].scatter(y_test, abs_residuals, c=y_test, 
                                 cmap='viridis', alpha=0.6, s=30)
    axes[1, 0].axhline(y=mae, color='r', linestyle='--', lw=2, alpha=0.7, label=f'MAE = {mae:.3f}')
    axes[1, 0].set_xlabel('True logS Value')
    axes[1, 0].set_ylabel('Absolute Error (logS units)')
    axes[1, 0].set_title('Error vs True Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='True logS')
    
    # 5. Enhanced Cumulative Error (BOTTOM-MIDDLE)
    sorted_errors = np.sort(abs_residuals)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1, 1].plot(sorted_errors, cumulative, 'b-', lw=2)
    
    # Add multiple threshold lines
    for percentile in [50, 80, 90, 95]:
        error_threshold = np.percentile(abs_residuals, percentile)
        axes[1, 1].axvline(x=error_threshold, color='r', linestyle='--', lw=1, alpha=0.5)
        axes[1, 1].axhline(y=percentile/100, color='r', linestyle='--', lw=1, alpha=0.5)
        axes[1, 1].text(error_threshold + 0.02, percentile/100 - 0.02, 
                       f'{percentile}%: {error_threshold:.3f}',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    axes[1, 1].set_xlabel('Absolute Error Threshold (logS units)')
    axes[1, 1].set_ylabel('Cumulative Proportion')
    axes[1, 1].set_title('Cumulative Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Bland-Altman Plot (BOTTOM-RIGHT) - REPLACES Confusion Matrix
    means = (y_test + y_pred) / 2
    differences = y_pred - y_test
    
    axes[1, 2].scatter(means, differences, alpha=0.5, s=20)
    axes[1, 2].axhline(y=np.mean(differences), color='r', linestyle='-', lw=2, 
                      label=f'Mean bias: {np.mean(differences):.3f}')
    axes[1, 2].axhline(y=np.mean(differences) + 1.96 * np.std(differences), 
                      color='gray', linestyle='--', lw=1.5, 
                      label='95% limits of agreement')
    axes[1, 2].axhline(y=np.mean(differences) - 1.96 * np.std(differences), 
                      color='gray', linestyle='--', lw=1.5)
    axes[1, 2].axhline(y=0, color='k', linestyle=':', lw=1, alpha=0.5)
    
    axes[1, 2].set_xlabel('Mean of Experimental and Predicted logS')
    axes[1, 2].set_ylabel('Predicted - Experimental logS')
    axes[1, 2].set_title('Bland-Altman Plot (Agreement Analysis)')
    axes[1, 2].legend(loc='best', fontsize=9)
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add agreement statistics
    loa_upper = np.mean(differences) + 1.96 * np.std(differences)
    loa_lower = np.mean(differences) - 1.96 * np.std(differences)
    axes[1, 2].text(0.05, 0.95, 
                   f'95% LoA: [{loa_lower:.3f}, {loa_upper:.3f}]',
                   transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('best_model_analysis_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print comprehensive statistics
    print("\n" + "="*60)
    print("PERFORMANCE STATISTICS")
    print("="*60)
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f} logS")
    print(f"MAE: {mae:.3f} logS")
    print(f"Max Absolute Error: {np.max(abs_residuals):.3f} logS")
    print(f"Mean Bias (Predicted - Actual): {np.mean(differences):.3f} logS")
    
    print(f"\nError Distribution Percentiles:")
    for p in [25, 50, 75, 90, 95]:
        print(f"  {p}th percentile: {np.percentile(abs_residuals, p):.3f} logS")
    
    print(f"\nPractical Interpretation:")
    print(f"Average error: {mae:.3f} logS units")
    print(f"Concentration error factor: {10**mae:.1f}x")
    print(f"95% of predictions within: {np.percentile(abs_residuals, 95):.3f} logS")
    
    # Calculate percentage within practical error thresholds
    thresholds = [0.5, 1.0, 1.5, 2.0]  # logS units
    print(f"\nPredictions within error thresholds:")
    for thresh in thresholds:
        pct_within = np.sum(abs_residuals <= thresh) / len(abs_residuals) * 100
        conc_factor = 10**thresh
        print(f"  ≤ {thresh:.1f} logS ({conc_factor:.1f}x concentration): {pct_within:.1f}%")
    
    # Agreement analysis
    within_loa = np.sum((differences >= loa_lower) & (differences <= loa_upper)) / len(differences) * 100
    print(f"\nAgreement Analysis:")
    print(f"  95% Limits of Agreement: [{loa_lower:.3f}, {loa_upper:.3f}]")
    print(f"  Percentage within LoA: {within_loa:.1f}% (expected: 95%)")
    
    # Check for proportional bias
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(means, differences)
    print(f"  Correlation between means and differences: {corr:.3f}")
    if p_value < 0.05:
        print(f"  Significant proportional bias detected (p = {p_value:.4f})")
    else:
        print(f"  No significant proportional bias (p = {p_value:.4f})")

def analyze_percent_discrepancy(y_true, y_pred, model_name):
    print("\nPercent Discrepancy Analysis")
    
    actual_conc = 10**y_true
    predicted_conc = 10**y_pred
    
    percent_errors = 100 * np.abs((predicted_conc - actual_conc) / actual_conc)
    percent_errors = np.nan_to_num(percent_errors, nan=1000, posinf=1000, neginf=1000)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Percent Discrepancy: {model_name}', fontsize=16, fontweight='bold')
    
    axes[0, 0].hist(percent_errors[percent_errors < 500], bins=30, 
                   edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(x=100, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Percent Error (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Percent Error Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    sorted_errors = np.sort(percent_errors[percent_errors < 1000])
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    
    axes[0, 1].plot(sorted_errors, cumulative, 'b-', linewidth=2)
    axes[0, 1].axhline(y=0.9, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 1].axvline(x=np.percentile(percent_errors[percent_errors < 1000], 90), 
                      color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 1].set_xlabel('Percent Error (%)')
    axes[0, 1].set_ylabel('Cumulative Proportion')
    axes[0, 1].set_title('Cumulative Error Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    scatter = axes[0, 2].scatter(y_true, percent_errors, 
                                c=actual_conc, cmap='viridis', 
                                alpha=0.6, s=30, norm='log')
    axes[0, 2].set_xlabel('Actual logS')
    axes[0, 2].set_ylabel('Percent Error (%)')
    axes[0, 2].set_title('Error vs Solubility Level')
    axes[0, 2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 2], label='Actual Concentration (mol/L)')
    
    error_categories = ['< 10%', '10-50%', '50-100%', '100-500%', '> 500%']
    error_bins = [0, 10, 50, 100, 500, 1000]
    category_counts = []
    
    for i in range(len(error_bins)-1):
        count = np.sum((percent_errors >= error_bins[i]) & (percent_errors < error_bins[i+1]))
        category_counts.append(count)
    
    colors = ['lightgreen', 'yellowgreen', 'gold', 'orange', 'lightcoral']
    bars = axes[1, 0].bar(range(len(error_categories)), category_counts, color=colors)
    axes[1, 0].set_xticks(range(len(error_categories)))
    axes[1, 0].set_xticklabels(error_categories, rotation=45)
    axes[1, 0].set_ylabel('Number of Compounds')
    axes[1, 0].set_title('Error Category Distribution')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    total = len(y_true)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = height / total * 100
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{percentage:.1f}%', ha='center', va='bottom')
    
    bins = [-12, -6, -4, -2, 2]
    labels = ['Very Poor', 'Poor', 'Moderate', 'Good']
    solubility_classes = pd.cut(y_true, bins=bins, labels=labels)
    
    median_errors = []
    mean_errors = []
    
    for cls in labels:
        mask = solubility_classes == cls
        if mask.any():
            median_errors.append(np.median(percent_errors[mask]))
            mean_errors.append(np.mean(percent_errors[mask]))
        else:
            median_errors.append(0)
            mean_errors.append(0)
    
    x_pos = np.arange(len(labels))
    width = 0.35
    
    axes[1, 1].bar(x_pos - width/2, median_errors, width, 
                  label='Median Error', color='lightblue', alpha=0.8)
    axes[1, 1].bar(x_pos + width/2, mean_errors, width, 
                  label='Mean Error', color='lightcoral', alpha=0.8)
    
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(labels, rotation=45)
    axes[1, 1].set_ylabel('Percent Error (%)')
    axes[1, 1].set_title('Error by Solubility Class')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    worst_indices = np.argsort(percent_errors)[-10:][::-1]
    worst_actual = y_true[worst_indices]
    worst_pred = y_pred[worst_indices]
    worst_errors = percent_errors[worst_indices]
    
    axes[1, 2].barh(range(10), worst_errors[:10], color='lightcoral')
    axes[1, 2].set_yticks(range(10))
    axes[1, 2].set_yticklabels([f'logS={worst_actual[i]:.1f}' for i in range(10)])
    axes[1, 2].set_xlabel('Percent Error (%)')
    axes[1, 2].set_title('Top 10 Worst Predictions')
    axes[1, 2].invert_yaxis()
    axes[1, 2].grid(True, alpha=0.3, axis='x')
    
    for i, (error, actual, pred) in enumerate(zip(worst_errors[:10], worst_actual[:10], worst_pred[:10])):
        axes[1, 2].text(error + 5, i, f'Pred: {pred:.1f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('percent_discrepancy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    median_error = np.median(percent_errors[percent_errors < 1000])
    mean_error = np.mean(percent_errors[percent_errors < 1000])
    
    print("\nPercent Discrepancy Summary")
    print(f"Median Percent Error: {median_error:.1f}%")
    print(f"Mean Percent Error: {mean_error:.1f}%")
    
    print(f"\nPercentage of predictions within:")
    print(f"10% error: {np.sum(percent_errors <= 10)/len(y_true)*100:.1f}%")
    print(f"50% error: {np.sum(percent_errors <= 50)/len(y_true)*100:.1f}%")
    print(f"100% error (2x concentration): {np.sum(percent_errors <= 100)/len(y_true)*100:.1f}%")
    print(f"500% error: {np.sum(percent_errors <= 500)/len(y_true)*100:.1f}%")
    
    print(f"\nPractical Interpretation")
    if median_error <= 50:
        print(f"Median error of {median_error:.1f}% is acceptable for early-stage screening")
        print(f"{np.sum(percent_errors <= 100)/len(y_true)*100:.1f}% of predictions are within 2x concentration error")
    else:
        print(f"Median error of {median_error:.1f}% suggests model needs improvement")
        print(f"Only {np.sum(percent_errors <= 100)/len(y_true)*100:.1f}% within 2x error")
    
    print(f"\nExperimental variability comparison:")
    print(f"Typical reproducibility: 20-50% error")
    print(f"Different methods/labs: 100-200% error")
    print(f"Model performance: {median_error:.1f}% median error")
    
    error_analysis = pd.DataFrame({
        'Actual_logS': y_true,
        'Predicted_logS': y_pred,
        'Actual_Conc_mol_L': actual_conc,
        'Predicted_Conc_mol_L': predicted_conc,
        'Percent_Error': percent_errors,
        'Solubility_Class': solubility_classes
    })
    
    error_analysis.to_csv('percent_error_analysis.csv', index=False)
    print(f"\nDetailed error analysis saved")
    
    return percent_errors

def analyze_feature_importance(best_model, feature_names, X_train):
    print("\nFeature Importance Analysis")
    
    model_type = type(best_model).__name__
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_)
    else:
        print("No feature importance available")
        return None
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    
    bars = plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance Score')
    plt.title(f'Top 15 Features ({model_type})')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, imp) in enumerate(zip(bars, top_features['Importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{imp:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig('feature_importance_best_model.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTop 10 Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"{i+1:2}. {row['Feature']:30}: {row['Importance']:.4f}")
    
    importance_df.to_csv('feature_importance_best_model.csv', index=False)
    print(f"\nFeature importance saved")
    
    return importance_df

def save_model_and_artifacts(best_model, results_df, importance_df):
    print("\nSaving model and artifacts")
    
    import joblib
    import json
    
    best_model_name = results_df.iloc[0]['Model']
    model_filename = f'best_model_{best_model_name.replace(" ", "_")}.pkl'
    joblib.dump(best_model, model_filename)
    print(f"Model saved: {model_filename}")
    
    results_df.to_csv('model_performance_results.csv', index=False)
    print(f"Results saved")
    
    config = {
        'best_model': best_model_name,
        'test_r2': float(results_df.iloc[0]['Test_R2']),
        'test_rmse': float(results_df.iloc[0]['Test_RMSE']),
        'test_mae': float(results_df.iloc[0]['Test_MAE']),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': type(best_model).__name__
    }
    
    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config saved")

def generate_ml_report(results_df, best_model, importance_df):
    print("\nGenerating ML report")
    
    report = []
    report.append("Machine Learning for Drug Solubility Prediction")
    
    best_model_name = results_df.iloc[0]['Model']
    report.append(f"\nExecutive Summary")
    report.append(f"Best Model: {best_model_name}")
    report.append(f"Test R²: {results_df.iloc[0]['Test_R2']:.4f}")
    report.append(f"Test RMSE: {results_df.iloc[0]['Test_RMSE']:.3f} logS")
    report.append(f"Test MAE: {results_df.iloc[0]['Test_MAE']:.3f} logS")
    
    report.append(f"\nModel Performance")
    for i, row in results_df.iterrows():
        report.append(f"{i+1:2}. {row['Model']:25}: R²={row['Test_R2']:.4f}")
    
    if importance_df is not None:
        report.append(f"\nTop 10 Features")
        for i, row in importance_df.head(10).iterrows():
            report.append(f"{i+1:2}. {row['Feature']:30} ({row['Importance']:.4f})")
    
    mae = results_df.iloc[0]['Test_MAE']
    report.append(f"\nPractical Implications")
    report.append(f"Average error: {mae:.3f} logS units")
    report.append(f"Concentration error: {10**mae:.1f}x")
    report.append(f"Suitable for: Early-stage drug screening")
    
    with open('ml_analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print("Report saved")
    
    print(f"\nML Training Summary")
    print(f"Best model: {best_model_name}")
    print(f"Test R²: {results_df.iloc[0]['Test_R2']:.4f}")
    print(f"Error: {10**results_df.iloc[0]['Test_MAE']:.1f}x concentration")

def main():
    X, y, feature_names = load_ml_data()
    if X is None:
        return
    
    X_train, X_test, y_train, y_test = split_data_stratified(X, y)
    models = initialize_models()
    
    results_df, trained_models = train_and_evaluate_all_models(
        models, X_train, X_test, y_train, y_test
    )
    
    create_model_comparison_plot(results_df)
    
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    
    print(f"\nBest model selected: {best_model_name}")
    
    analyze_best_model_predictions(best_model, X_test, y_test, feature_names)
    
    y_pred = best_model.predict(X_test)
    percent_errors = analyze_percent_discrepancy(y_test, y_pred, best_model_name)
    
    importance_df = analyze_feature_importance(best_model, feature_names, X_train)
    save_model_and_artifacts(best_model, results_df, importance_df)
    generate_ml_report(results_df, best_model, importance_df)
    
    print("\nML Pipeline Complete")

if __name__ == "__main__":
    main()