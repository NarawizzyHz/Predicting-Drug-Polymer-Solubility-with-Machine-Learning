
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load the saved dataset"""
    print("=" * 70)
    print("DATA EXPLORATION AND VISUALIZATION")
    print("=" * 70)
    
    try:
        df = pd.read_csv('drug_solubility_dataset.csv')
        print("SUCCESS: Dataset loaded successfully.")
        print(f"   Compounds: {len(df):,}")
        print(f"   Features: {len(df.columns)}")
        return df
    except FileNotFoundError:
        print("ERROR: File not found. Please run data_acquisition.py first.")
        return None

def create_distribution_plots(df):
    """Create distribution plots for key features"""
    
    print("\nCreating distribution plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Distribution of Drug Properties', fontsize=16, fontweight='bold')
    
    # 1. Target variable: logS
    axes[0, 0].hist(df['logS'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(df['logS'].mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {df["logS"].mean():.2f}')
    axes[0, 0].axvline(df['logS'].median(), color='green', linestyle=':', 
                      linewidth=2, label=f'Median: {df["logS"].median():.2f}')
    axes[0, 0].set_xlabel('logS (mol/L)', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Drug Solubility Distribution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Molecular Weight
    axes[0, 1].hist(df['Molecular Weight'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[0, 1].set_xlabel('Molecular Weight (g/mol)', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].set_title('Molecular Weight Distribution', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. LogP
    axes[0, 2].hist(df['LogP (octanol-water)'], bins=30, edgecolor='black', alpha=0.7, color='salmon')
    axes[0, 2].set_xlabel('LogP', fontweight='bold')
    axes[0, 2].set_ylabel('Frequency', fontweight='bold')
    axes[0, 2].set_title('Lipophilicity Distribution', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Polar Surface Area
    axes[1, 0].hist(df['Polar Surface Area'], bins=30, edgecolor='black', alpha=0.7, color='gold')
    axes[1, 0].set_xlabel('Polar Surface Area (Å²)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Polarity Distribution', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. H-Bond Donors
    donor_counts = df['Number of H-Bond Donors'].value_counts().sort_index()
    axes[1, 1].bar(donor_counts.index, donor_counts.values, color='violet', alpha=0.7)
    axes[1, 1].set_xlabel('Number of H-Bond Donors', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontweight='bold')
    axes[1, 1].set_title('H-Bond Donor Distribution', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. H-Bond Acceptors
    acceptor_counts = df['Number of H-Bond Acceptors'].value_counts().sort_index()
    axes[1, 2].bar(acceptor_counts.index[:15], acceptor_counts.values[:15], color='orange', alpha=0.7)
    axes[1, 2].set_xlabel('Number of H-Bond Acceptors', fontweight='bold')
    axes[1, 2].set_ylabel('Frequency', fontweight='bold')
    axes[1, 2].set_title('H-Bond Acceptor Distribution (Top 15)', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # 7. Boxplot of logS by solubility class
    df['Solubility Class'] = pd.cut(df['logS'], 
                                    bins=[-10, -6, -4, -2, 2],
                                    labels=['Very Poor', 'Poor', 'Moderate', 'Good'])
    solubility_order = ['Very Poor', 'Poor', 'Moderate', 'Good']
    box_data = [df[df['Solubility Class'] == cls]['logS'] for cls in solubility_order]
    axes[2, 0].boxplot(box_data, labels=solubility_order)
    axes[2, 0].set_xlabel('Solubility Class', fontweight='bold')
    axes[2, 0].set_ylabel('logS', fontweight='bold')
    axes[2, 0].set_title('Solubility Distribution by Class', fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3, axis='y')
    
    # 8. Kernel Density Plot
    for cls in solubility_order:
        subset = df[df['Solubility Class'] == cls]['logS']
        sns.kdeplot(subset, label=cls, ax=axes[2, 1])
    axes[2, 1].set_xlabel('logS', fontweight='bold')
    axes[2, 1].set_ylabel('Density', fontweight='bold')
    axes[2, 1].set_title('Solubility Density by Class', fontweight='bold')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. QQ Plot for normality check
    stats.probplot(df['logS'], dist="norm", plot=axes[2, 2])
    axes[2, 2].set_title('Normality Check (Q-Q Plot)', fontweight='bold')
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('1_distribution_plots.png', dpi=300, bbox_inches='tight')
    print("   Saved: '1_distribution_plots.png'")
    plt.show()

def create_correlation_analysis(df):
    """Create correlation plots and analysis"""
    
    print("\nCreating correlation analysis...")
    
    # Prepare data for correlation (numeric columns only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_df = df[numeric_cols].copy()
    
    # Create correlation matrix
    corr_matrix = corr_df.corr()
    
    # Create figure for correlation analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Heatmap of all correlations
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=axes[0, 0], cbar_kws={'label': 'Correlation'})
    axes[0, 0].set_title('Feature Correlation Matrix', fontweight='bold')
    
    # 2. Correlation with target (logS)
    target_corr = corr_matrix['logS'].drop('logS').sort_values()
    colors = ['red' if x < 0 else 'green' for x in target_corr.values]
    axes[0, 1].barh(range(len(target_corr)), target_corr.values, color=colors)
    axes[0, 1].set_yticks(range(len(target_corr)))
    axes[0, 1].set_yticklabels(target_corr.index)
    axes[0, 1].set_xlabel('Correlation with logS', fontweight='bold')
    axes[0, 1].set_title('Feature Correlation with Solubility', fontweight='bold')
    axes[0, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # 3. Scatter plot: Molecular Weight vs logS
    axes[1, 0].scatter(df['Molecular Weight'], df['logS'], alpha=0.5, color='blue')
    z = np.polyfit(df['Molecular Weight'], df['logS'], 1)
    p = np.poly1d(z)
    axes[1, 0].plot(df['Molecular Weight'], p(df['Molecular Weight']), "r--", 
                    linewidth=2, label=f'Fit: y = {z[0]:.4f}x + {z[1]:.2f}')
    axes[1, 0].set_xlabel('Molecular Weight (g/mol)', fontweight='bold')
    axes[1, 0].set_ylabel('logS', fontweight='bold')
    axes[1, 0].set_title('Molecular Weight vs Solubility', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Scatter plot: LogP vs logS
    axes[1, 1].scatter(df['LogP (octanol-water)'], df['logS'], alpha=0.5, color='green')
    z = np.polyfit(df['LogP (octanol-water)'], df['logS'], 1)
    p = np.poly1d(z)
    axes[1, 1].plot(df['LogP (octanol-water)'], p(df['LogP (octanol-water)']), "r--", 
                    linewidth=2, label=f'Fit: y = {z[0]:.4f}x + {z[1]:.2f}')
    axes[1, 1].set_xlabel('LogP', fontweight='bold')
    axes[1, 1].set_ylabel('logS', fontweight='bold')
    axes[1, 1].set_title('Lipophilicity vs Solubility', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('2_correlation_analysis.png', dpi=300, bbox_inches='tight')
    print("   Saved: '2_correlation_analysis.png'")
    plt.show()
    
    # Print correlation insights
    print("\nCorrelation Insights:")
    print("=" * 50)
    
    # Top positive correlations with logS
    target_corr_sorted = corr_matrix['logS'].sort_values(ascending=False)
    print("\nTop POSITIVE correlations with solubility:")
    for i, (feature, corr) in enumerate(target_corr_sorted.items()):
        if feature != 'logS' and corr > 0:
            print(f"   {i+1:2}. {feature:30} : {corr:.3f}")
            if i >= 4:
                break
    
    # Top negative correlations with logS
    print("\nTop NEGATIVE correlations with solubility:")
    for i, (feature, corr) in enumerate(target_corr_sorted.items()):
        if feature != 'logS' and corr < 0:
            print(f"   {i+1:2}. {feature:30} : {corr:.3f}")
            if i >= 4:
                break

def create_relationship_plots(df):
    """Create detailed relationship plots"""
    
    print("\nCreating relationship plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Advanced Relationship Analysis', fontsize=16, fontweight='bold')
    
    # 1. 3D effect: MW vs LogP vs logS (color coded)
    scatter = axes[0, 0].scatter(df['Molecular Weight'], 
                                df['LogP (octanol-water)'], 
                                c=df['logS'], 
                                cmap='viridis', 
                                alpha=0.6, 
                                s=50)
    axes[0, 0].set_xlabel('Molecular Weight', fontweight='bold')
    axes[0, 0].set_ylabel('LogP', fontweight='bold')
    axes[0, 0].set_title('MW vs LogP (Color = Solubility)', fontweight='bold')
    plt.colorbar(scatter, ax=axes[0, 0], label='logS')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Polar Surface Area vs logS with H-bond donors
    scatter = axes[0, 1].scatter(df['Polar Surface Area'], 
                                df['logS'], 
                                c=df['Number of H-Bond Donors'], 
                                cmap='plasma', 
                                alpha=0.6, 
                                s=50)
    axes[0, 1].set_xlabel('Polar Surface Area (Å²)', fontweight='bold')
    axes[0, 1].set_ylabel('logS', fontweight='bold')
    axes[0, 1].set_title('Polarity vs Solubility (Color = H-Bond Donors)', fontweight='bold')
    plt.colorbar(scatter, ax=axes[0, 1], label='H-Bond Donors')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Pairwise correlation matrix (simplified)
    features_subset = ['logS', 'Molecular Weight', 'LogP (octanol-water)', 'Polar Surface Area']
    corr_subset = df[features_subset].corr()
    mask = np.triu(np.ones_like(corr_subset, dtype=bool))
    sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f', 
                cmap='RdYlBu', center=0, square=True, ax=axes[1, 0])
    axes[1, 0].set_title('Key Features Correlation', fontweight='bold')
    
    # 4. Violin plot by solubility class
    df['Solubility Class'] = pd.cut(df['logS'], 
                                    bins=[-10, -6, -4, -2, 2],
                                    labels=['Very Poor', 'Poor', 'Moderate', 'Good'])
    sns.violinplot(x='Solubility Class', y='Molecular Weight', data=df, 
                  ax=axes[1, 1], palette='Set2')
    axes[1, 1].set_xlabel('Solubility Class', fontweight='bold')
    axes[1, 1].set_ylabel('Molecular Weight', fontweight='bold')
    axes[1, 1].set_title('Molecular Weight Distribution by Solubility', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('3_relationship_plots.png', dpi=300, bbox_inches='tight')
    print("   Saved: '3_relationship_plots.png'")
    plt.show()

def create_solubility_rules(df):
    """Analyze and visualize solubility rules"""
    
    print("\nAnalyzing solubility rules...")
    
    # Rule 1: Lipinski's Rule of 5 (simplified for solubility)
    df['Lipinski_MW'] = df['Molecular Weight'] <= 500
    df['Lipinski_LogP'] = df['LogP (octanol-water)'] <= 5
    df['Lipinski_HBD'] = df['Number of H-Bond Donors'] <= 5
    df['Lipinski_HBA'] = df['Number of H-Bond Acceptors'] <= 10
    
    df['Lipinski_Pass'] = (df['Lipinski_MW'] & df['Lipinski_LogP'] & 
                          df['Lipinski_HBD'] & df['Lipinski_HBA'])
    
    # Calculate statistics
    total_drugs = len(df)
    pass_count = df['Lipinski_Pass'].sum()
    fail_count = total_drugs - pass_count
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Pass/Fail pie chart
    labels = ['Pass Lipinski', 'Fail Lipinski']
    sizes = [pass_count, fail_count]
    colors = ['lightgreen', 'lightcoral']
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, shadow=True)
    axes[0].set_title('Lipinski Rule of 5 Compliance', fontweight='bold')
    
    # 2. Solubility comparison
    pass_sol = df[df['Lipinski_Pass']]['logS'].mean()
    fail_sol = df[~df['Lipinski_Pass']]['logS'].mean()
    
    bars = axes[1].bar(['Pass Lipinski', 'Fail Lipinski'], 
                      [pass_sol, fail_sol], 
                      color=['lightgreen', 'lightcoral'])
    axes[1].set_ylabel('Average logS', fontweight='bold')
    axes[1].set_title('Average Solubility by Lipinski Compliance', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('4_solubility_rules.png', dpi=300, bbox_inches='tight')
    print("   Saved: '4_solubility_rules.png'")
    plt.show()
    
    # Print rule analysis
    print("\nLipinski Rule of 5 Analysis:")
    print("=" * 50)
    print(f"   Total drugs analyzed: {total_drugs}")
    print(f"   Pass all rules: {pass_count} ({pass_count/total_drugs*100:.1f}%)")
    print(f"   Fail at least one rule: {fail_count} ({fail_count/total_drugs*100:.1f}%)")
    print(f"\n   Average solubility (pass): {pass_sol:.2f} logS")
    print(f"   Average solubility (fail): {fail_sol:.2f} logS")
    print(f"   Difference: {abs(pass_sol - fail_sol):.2f} logS units")

def save_insights_report(df):
    """Save comprehensive insights report"""
    
    print("\nGenerating insights report...")
    
    insights = []
    
    # 1. Basic statistics
    insights.append("Drug Solubility Dataset Report")
    insights.append(f"\nDataset Statistics:")
    insights.append(f"  Total compounds: {len(df)}")
    insights.append(f"  Solubility range: {df['logS'].min():.2f} to {df['logS'].max():.2f} logS")
    insights.append(f"  Mean solubility: {df['logS'].mean():.2f} logS")
    insights.append(f"  Standard deviation: {df['logS'].std():.2f}")
    
    # 2. Solubility classes
    bins = [-10, -6, -4, -2, 2]
    labels = ['Very Poor', 'Poor', 'Moderate', 'Good']
    df['Class'] = pd.cut(df['logS'], bins=bins, labels=labels)
    class_counts = df['Class'].value_counts().sort_index()
    
    insights.append(f"\nSolubility Classification:")
    for cls, count in class_counts.items():
        percentage = count / len(df) * 100
        insights.append(f"  {cls}: {count} compounds ({percentage:.1f}%)")
    
    # 3. Feature correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    target_corr = corr_matrix['logS'].drop('logS').sort_values(ascending=False)
    
    insights.append(f"\nTop Correlations with Solubility:")
    insights.append("  (Positive = more soluble, Negative = less soluble)")
    for feature, corr in target_corr.items():
        insights.append(f"  {feature:30}: {corr:7.3f}")
    
    # 4. Chemical insights
    insights.append(f"\nCHEMICAL INSIGHTS:")
    insights.append(f"  1. Molecular Weight Impact:")
    insights.append(f"     Average MW for soluble drugs (logS > -2): {df[df['logS'] > -2]['Molecular Weight'].mean():.1f} g/mol")
    insights.append(f"     Average MW for insoluble drugs (logS < -6): {df[df['logS'] < -6]['Molecular Weight'].mean():.1f} g/mol")
    
    insights.append(f"\n  2. Lipophilicity (LogP) Impact:")
    insights.append(f"     Average LogP for soluble drugs: {df[df['logS'] > -2]['LogP (octanol-water)'].mean():.2f}")
    insights.append(f"     Average LogP for insoluble drugs: {df[df['logS'] < -6]['LogP (octanol-water)'].mean():.2f}")
    
    insights.append(f"\n  3. Polar Surface Area:")
    insights.append(f"     Soluble drugs average PSA: {df[df['logS'] > -2]['Polar Surface Area'].mean():.1f} Å²")
    insights.append(f"     Insoluble drugs average PSA: {df[df['logS'] < -6]['Polar Surface Area'].mean():.1f} Å²")
    
    # 5. Recommendations for ML
    insights.append(f"\nMACHINE LEARNING RECOMMENDATIONS:")
    insights.append(f"  1. Key features to include: Molecular Weight, LogP, Polar Surface Area")
    insights.append(f"  2. Target transformation: Consider log transformation if needed")
    insights.append(f"  3. Outliers: Check compounds with logS < -8 (very insoluble)")
    insights.append(f"  4. Validation: Use stratified sampling based on solubility classes")
    
    # Save to file
    with open('data_insights_report.txt', 'w') as f:
        f.write('\n'.join(insights))
    
    print("   Saved: 'data_insights_report.txt'")
    
    # Also print to console
    print("Summary:")
    for line in insights[-20:]:
        print(line)

def main():
    """Main function to run data exploration"""
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Create all visualizations
    create_distribution_plots(df)
    create_correlation_analysis(df)
    create_relationship_plots(df)
    create_solubility_rules(df)
    save_insights_report(df)
    
    print("\nFiles created:")
    print("   1. 1_distribution_plots.png - Distribution of all features")
    print("   2. 2_correlation_analysis.png - Correlation heatmaps and scatter plots")
    print("   3. 3_relationship_plots.png - Advanced relationship plots")
    print("   4. 4_solubility_rules.png - Lipinski rule analysis")
    print("   5. data_insights_report.txt - Detailed insights and recommendations")
    
    print("\nNext step: Run 'python feature_engineering.py'")

if __name__ == "__main__":
    main()