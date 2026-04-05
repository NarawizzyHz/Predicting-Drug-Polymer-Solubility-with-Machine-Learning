
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_inspect_data():
    """Load and inspect the actual dataset structure"""
    
    print("Feature Engineering")
    
    try:
        df = pd.read_csv('drug_solubility_dataset.csv')
        print("SUCCESS: Dataset loaded successfully.")
        print(f"   Compounds: {len(df):,}")
        print(f"   Features: {len(df.columns)}")
        
        print("\nACTUAL COLUMNS IN DATASET:")
        print("-" * 40)
        for i, col in enumerate(df.columns, 1):
            col_type = str(df[col].dtype)
            unique_count = df[col].nunique()
            print(f"   {i:2}. {col:25} | Type: {col_type:10} | Unique: {unique_count:4}")
        
        print(f"\nTarget variable 'logS' statistics:")
        print(f"   Mean: {df['logS'].mean():.3f}")
        print(f"   Range: {df['logS'].min():.3f} to {df['logS'].max():.3f}")
        print(f"   Std Dev: {df['logS'].std():.3f}")
        
        return df
    except FileNotFoundError:
        print("ERROR: File not found. Please run data_acquisition.py first.")
        return None

def create_advanced_features(df):
    """Create advanced molecular descriptors from existing features"""
    
    print("\nCreating advanced molecular features...")
    
    # Original features
    original_features = df.columns.tolist()
    
    # 1. Size and complexity features
    df['MolWt_log'] = np.log1p(df['MolWt'])  # Log transform
    df['MolWt_squared'] = df['MolWt'] ** 2
    df['MolWt_cubed'] = df['MolWt'] ** 3
    
    # 2. Lipophilicity transformations
    df['MolLogP_squared'] = df['MolLogP'] ** 2
    df['MolLogP_abs'] = np.abs(df['MolLogP'])
    df['MolLogP_signed_sqrt'] = np.sign(df['MolLogP']) * np.sqrt(np.abs(df['MolLogP']))
    
    # 3. Rotatable bond features
    df['Flexibility_Index'] = df['NumRotatableBonds'] / (df['MolWt']/100 + 1)
    df['Rigidity_Score'] = 1 / (df['NumRotatableBonds'] + 1)
    
    # 4. Aromaticity features
    df['Aromatic_Index'] = df['AromaticProportion'] * 100
    df['Aromatic_Score'] = df['AromaticProportion'] * df['MolLogP']
    
    # 5. Composite descriptors (based on chemical principles)
    # Lipinski's Rule of 5 inspired features
    df['Lipinski_MW'] = (df['MolWt'] <= 500).astype(int)
    df['Lipinski_LogP'] = (df['MolLogP'] <= 5).astype(int)
    df['Lipinski_Score'] = df['Lipinski_MW'] + df['Lipinski_LogP']
    
    # Solubility heuristic scores
    df['Sol_Heuristic_1'] = -0.5 * (df['MolWt'] / 500) - 0.8 * (df['MolLogP'] / 5)
    df['Sol_Heuristic_2'] = 0.3 * (1 - df['AromaticProportion']) - 0.2 * (df['NumRotatableBonds'] / 10)
    
    # 6. Interaction terms
    df['MW_LogP_Interaction'] = df['MolWt'] * df['MolLogP'] / 1000
    df['MW_Aromatic_Interaction'] = df['MolWt'] * df['AromaticProportion']
    df['LogP_Rotatable_Interaction'] = df['MolLogP'] * df['NumRotatableBonds']
    
    # 7. Polynomial combinations
    df['Complexity_Score'] = (df['MolWt'] * df['NumRotatableBonds']) / 1000
    df['Polarity_Estimate'] = 1 / (np.abs(df['MolLogP']) + 0.1)
    
    # 8. Binned features
    df['MW_Bin'] = pd.cut(df['MolWt'], bins=5, labels=['VL', 'L', 'M', 'H', 'VH'])
    df['LogP_Bin'] = pd.cut(df['MolLogP'], bins=5, labels=['VL', 'L', 'M', 'H', 'VH'])
    df['Rotatable_Bin'] = pd.cut(df['NumRotatableBonds'], bins=5, labels=['VL', 'L', 'M', 'H', 'VH'])
    
    # Convert categorical to dummy variables
    df = pd.get_dummies(df, columns=['MW_Bin', 'LogP_Bin', 'Rotatable_Bin'], drop_first=True)
    
    # 9. Statistical features
    df['Z_Score_MolWt'] = (df['MolWt'] - df['MolWt'].mean()) / df['MolWt'].std()
    df['Z_Score_LogP'] = (df['MolLogP'] - df['MolLogP'].mean()) / df['MolLogP'].std()
    
    # 10. Ratio features
    df['MW_per_RotatableBond'] = df['MolWt'] / (df['NumRotatableBonds'] + 1)
    df['LogP_per_MW'] = df['MolLogP'] / (df['MolWt']/100 + 0.001)
    
    created_features = len(df.columns) - len(original_features)
    print(f"   Created {created_features} new features")
    print(f"   Total features now: {len(df.columns)}")
    
    # Save feature descriptions
    feature_descriptions = {
        'MolWt_log': 'Log-transformed molecular weight',
        'MolWt_squared': 'Squared molecular weight',
        'MolWt_cubed': 'Cubed molecular weight',
        'MolLogP_squared': 'Squared lipophilicity',
        'Flexibility_Index': 'Rotatable bonds normalized by molecular weight',
        'Aromatic_Index': 'Aromatic proportion as percentage',
        'Lipinski_Score': 'Compliance with Lipinski Rule of 5',
        'MW_LogP_Interaction': 'Interaction between molecular weight and lipophilicity',
        'Complexity_Score': 'Composite score of molecular complexity',
        'Polarity_Estimate': 'Estimated polarity based on lipophilicity'
    }
    
    pd.DataFrame(list(feature_descriptions.items()), 
                columns=['Feature', 'Description']).to_csv('feature_descriptions.csv', index=False)
    
    return df

def analyze_feature_distributions(df):
    """Analyze distributions of all features"""
    
    print("\nAnalyzing feature distributions...")
    
    # Get numeric columns (exclude target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'logS']
    
    # Create distribution plots for top 12 features
    n_features = min(12, len(numeric_cols))
    top_features = numeric_cols[:n_features]
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        if i < len(axes):
            axes[i].hist(df[feature], bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{feature} Distribution')
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   Feature distribution plot saved as 'feature_distributions.png'")

def handle_missing_values(df):
    """Handle missing values"""
    
    print("\nHandling missing values...")
    
    missing_before = df.isnull().sum().sum()
    
    if missing_before > 0:
        print(f"   Missing values found: {missing_before}")
        
        # Fill with median for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"     Filled {col} with median: {median_val:.4f}")
        
        missing_after = df.isnull().sum().sum()
        print(f"   Missing values after treatment: {missing_after}")
    else:
        print("   No missing values found.")
    
    return df

def remove_outliers(df, target_col='logS', threshold=1.5):
    """Remove outliers using IQR method"""
    
    print("\nRemoving outliers...")
    
    original_count = len(df)
    
    # Calculate IQR
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    # Identify outliers
    outliers_mask = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
    outliers = df[outliers_mask]
    
    if len(outliers) > 0:
        df_clean = df[~outliers_mask]
        removed_count = original_count - len(df_clean)
        
        print(f"   IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"   Outliers detected: {removed_count} compounds")
        print(f"   Remaining compounds: {len(df_clean)}")
        
        # Save outlier information
        outliers.to_csv('outliers_removed.csv', index=False)
        print(f"   Outliers saved to 'outliers_removed.csv'")
        
        return df_clean
    else:
        print(f"   No outliers detected (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")
        return df

def prepare_ml_data(df, target_col='logS'):
    """Prepare data for machine learning"""
    
    print("\nPreparing data for machine learning...")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create DataFrame with scaled features
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Save scaler parameters
    scaler_params = pd.DataFrame({
        'feature': feature_cols,
        'mean': scaler.mean_,
        'scale': scaler.scale_
    })
    scaler_params.to_csv('scaler_parameters.csv', index=False)
    
    print(f"   Scaled {len(feature_cols)} features")
    print(f"   Target variable: {target_col} (range: {y.min():.2f} to {y.max():.2f})")
    
    return X_scaled_df, y, np.array(feature_cols), scaler

def perform_feature_selection(X, y, feature_names, k=15):
    """Select most important features"""
    
    print(f"\nPerforming feature selection (top {k} features)...")
    
    # Method 1: F-regression
    selector_f = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    X_f = selector_f.fit_transform(X, y)
    f_scores = selector_f.scores_
    f_pvalues = selector_f.pvalues_
    
    # Method 2: Mutual Information
    selector_mi = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
    X_mi = selector_mi.fit_transform(X, y)
    mi_scores = selector_mi.scores_
    
    # Combine scores
    f_scores_norm = f_scores / f_scores.max() if f_scores.max() > 0 else f_scores
    mi_scores_norm = mi_scores / mi_scores.max() if mi_scores.max() > 0 else mi_scores
    combined_scores = f_scores_norm + mi_scores_norm
    
    # Select top k features
    top_indices = np.argsort(combined_scores)[-k:][::-1]
    selected_features = feature_names[top_indices]
    X_selected = X[:, top_indices]
    
    # Create results DataFrame
    feature_results = pd.DataFrame({
        'Feature': feature_names,
        'F_Score': f_scores,
        'F_pValue': f_pvalues,
        'MI_Score': mi_scores,
        'Combined_Score': combined_scores
    })
    
    # Sort by combined score
    feature_results = feature_results.sort_values('Combined_Score', ascending=False)
    
    print("\nTop 10 features by importance:")
    print("-" * 80)
    print(feature_results.head(10).to_string(index=False))
    
    # Save results
    feature_results.to_csv('feature_selection_results.csv', index=False)
    print(f"\n   Feature selection results saved to 'feature_selection_results.csv'")
    
    return X_selected, selected_features, feature_results

def visualize_feature_importance(feature_results):
    """Visualize feature importance"""
    
    print("\nCreating feature importance visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get top 15 features
    top_features = feature_results.head(15)
    
    # Plot 1: Combined scores
    axes[0].barh(range(len(top_features)), top_features['Combined_Score'], color='steelblue')
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['Feature'])
    axes[0].set_xlabel('Combined Importance Score')
    axes[0].set_title('Top 15 Features (Combined Score)')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, score in enumerate(top_features['Combined_Score']):
        axes[0].text(score + 0.01, i, f'{score:.2f}', va='center')
    
    # Plot 2: F-score vs MI-score
    x_pos = np.arange(len(top_features))
    width = 0.35
    
    axes[1].barh(x_pos - width/2, top_features['F_Score']/top_features['F_Score'].max(), 
                width, label='F-Score (Linear)', color='lightcoral', alpha=0.7)
    axes[1].barh(x_pos + width/2, top_features['MI_Score']/top_features['MI_Score'].max(), 
                width, label='MI-Score (Non-linear)', color='lightgreen', alpha=0.7)
    
    axes[1].set_yticks(x_pos)
    axes[1].set_yticklabels(top_features['Feature'])
    axes[1].set_xlabel('Normalized Score')
    axes[1].set_title('Linear vs Non-linear Feature Importance')
    axes[1].legend()
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   Feature importance plot saved as 'feature_importance.png'")

def analyze_correlations(df, target_col='logS'):
    """Analyze correlations between features and target"""
    
    print("\nAnalyzing correlations...")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlations with target
    correlations = []
    for col in numeric_cols:
        if col != target_col:
            corr = df[col].corr(df[target_col])
            correlations.append({
                'Feature': col,
                'Correlation': corr,
                'Abs_Correlation': abs(corr)
            })
    
    corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)
    
    # Plot top correlations
    plt.figure(figsize=(10, 8))
    top_corr = corr_df.head(15)
    
    colors = ['red' if x < 0 else 'green' for x in top_corr['Correlation']]
    bars = plt.barh(range(len(top_corr)), top_corr['Correlation'], color=colors)
    plt.yticks(range(len(top_corr)), top_corr['Feature'])
    plt.xlabel('Correlation Coefficient with logS')
    plt.title('Top 15 Features Correlated with Solubility')
    plt.axvline(x=0, color='black', linewidth=0.5)
    plt.gca().invert_yaxis()
    
    # Add correlation values
    for i, (bar, corr) in enumerate(zip(bars, top_corr['Correlation'])):
        plt.text(bar.get_width() + (0.01 if corr >= 0 else -0.05), 
                bar.get_y() + bar.get_height()/2, 
                f'{corr:.3f}', va='center', 
                color='black', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('target_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save correlations
    corr_df.to_csv('feature_correlations.csv', index=False)
    print(f"   Correlation analysis saved to 'feature_correlations.csv'")
    print(f"   Correlation plot saved as 'target_correlations.png'")
    
    return corr_df

def save_processed_data(df, X_selected, selected_features, y):
    """Save processed data for ML"""
    
    print("\nSaving processed data...")
    
    # Save full dataset
    df.to_csv('processed_drug_solubility.csv', index=False)
    
    # Save ML-ready data
    pd.DataFrame(X_selected, columns=selected_features).to_csv('X_ml_ready.csv', index=False)
    pd.DataFrame(y, columns=['logS']).to_csv('y_ml_ready.csv', index=False)
    
    # Save selected feature names
    pd.DataFrame({'selected_features': selected_features}).to_csv('selected_features.csv', index=False)
    
    print("   Processed data saved:")
    print("     - processed_drug_solubility.csv: Full processed dataset")
    print("     - X_ml_ready.csv: Feature matrix for ML")
    print("     - y_ml_ready.csv: Target variable for ML")
    print("     - selected_features.csv: Selected feature names")

def generate_summary_report(df, selected_features, corr_df):
    """Generate summary report"""
    
    print("\nGenerating summary report...")
    
    report = []
    report.append("Feature Engineering sum")
    
    # Basic statistics
    report.append(f"\n1. DATASET STATISTICS:")
    report.append(f"   Total compounds: {len(df)}")
    report.append(f"   Total features created: {len(df.columns)}")
    report.append(f"   Features selected for ML: {len(selected_features)}")
    report.append(f"   Target variable range: {df['logS'].min():.2f} to {df['logS'].max():.2f}")
    
    # Feature categories
    original_features = ['MolLogP', 'MolWt', 'NumRotatableBonds', 'AromaticProportion', 'logS']
    derived_features = [col for col in df.columns if col not in original_features]
    
    report.append(f"\n2. FEATURE CATEGORIES:")
    report.append(f"   Original features: {len(original_features)}")
    report.append(f"   Derived features: {len(derived_features)}")
    
    # Top correlations
    report.append(f"\n3. TOP CORRELATIONS WITH SOLUBILITY:")
    top_pos = corr_df[corr_df['Correlation'] > 0].head(3)
    top_neg = corr_df[corr_df['Correlation'] < 0].head(3)
    
    report.append(f"   Most positively correlated:")
    for _, row in top_pos.iterrows():
        report.append(f"     - {row['Feature']}: {row['Correlation']:.3f}")
    
    report.append(f"\n   Most negatively correlated:")
    for _, row in top_neg.iterrows():
        report.append(f"     - {row['Feature']}: {row['Correlation']:.3f}")
    
    # Selected features
    report.append(f"\n4. SELECTED FEATURES FOR ML:")
    for i, feat in enumerate(selected_features[:10], 1):
        report.append(f"   {i:2}. {feat}")
    if len(selected_features) > 10:
        report.append(f"   ... and {len(selected_features) - 10} more")
    
    # Chemical insights
    report.append(f"\n5. CHEMICAL INSIGHTS:")
    report.append(f"   • Molecular weight range: {df['MolWt'].min():.1f} to {df['MolWt'].max():.1f} g/mol")
    report.append(f"   • Lipophilicity range: {df['MolLogP'].min():.2f} to {df['MolLogP'].max():.2f}")
    report.append(f"   • Rotatable bonds range: {df['NumRotatableBonds'].min()} to {df['NumRotatableBonds'].max()}")
    report.append(f"   • Aromatic proportion range: {df['AromaticProportion'].min():.2f} to {df['AromaticProportion'].max():.2f}")
    
    # Save report
    with open('feature_engineering_summary.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print("   Summary report saved to 'feature_engineering_summary.txt'")
