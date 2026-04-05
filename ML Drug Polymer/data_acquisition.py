
import pandas as pd
import numpy as np

def download_solubility_data():
    """Download the Delaney (ESOL) solubility dataset"""
    
    print("Drug Solubility Predicting")
    print("\nDownloading dataset...")
    
    # PRIMARY DATASET: Delaney (ESOL)
    url = "https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv"
    
    try:
        df = pd.read_csv(url)
        print("SUCCESS: Dataset downloaded successfully.")
    except Exception as e:
        print(f"ERROR: Download failed - {e}")
        print("\nTrying alternative source...")
        
        # Alternative source
        alt_url = "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv"
        df = pd.read_csv(alt_url)
        print("SUCCESS: Dataset downloaded from alternative source.")
    
    # Display basic info
    print(f"\nDataset Information:")
    print(f"   Number of compounds: {len(df):,}")
    print(f"   Number of features: {len(df.columns)}")
    print(f"   Memory usage: {df.memory_usage().sum() / 1024:.1f} KB")
    
    return df

def explore_dataset(df):
    """Explore the dataset structure and statistics"""
    
    print("Dataset Exploring")
    
    # 1. Display column information
    print("\nColumns Available:")
    for i, col in enumerate(df.columns, 1):
        col_type = str(df[col].dtype)
        unique_count = df[col].nunique()
        print(f"   {i:2}. {col:25} | Type: {col_type:10} | Unique: {unique_count:4}")
    
    # 2. Display first few compounds
    print("\nFirst 3 Compounds:")
    print(df.head(3).to_string())
    
    # 3. Target variable statistics
    print("\nTarget Variable (logS) Statistics:")
    print(df['logS'].describe().to_string())
    
    # 4. Check for missing values
    print("\nMissing Values Check:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   No missing values found.")
    else:
        print("   Missing values found:")
        for col, count in missing[missing > 0].items():
            print(f"     {col}: {count} missing")
    
    return df

def save_dataset(df, filename="drug_solubility_dataset.csv"):
    """Save dataset to CSV file"""
    
    df.to_csv(filename, index=False)
    print(f"\nDataset Saved:")
    print(f"   Filename: {filename}")
    print(f"   Location: Current directory")
    print(f"   Size: {len(df)} compounds × {len(df.columns)} features")
    
    return filename

def create_summary_report(df):
    """Create a summary report of the dataset"""
    
    print("Sum Report")
    
    # Create summary statistics
    summary_data = {
        'Dataset Name': ['Delaney (ESOL) Solubility Dataset'],
        'Source': ['GitHub (Data Professor)'],
        'Compounds': [len(df)],
        'Features': [len(df.columns)],
        'Target Variable': ['logS (aqueous solubility in mol/L)'],
        'logS Range': [f"{df['logS'].min():.2f} to {df['logS'].max():.2f}"],
        'Mean logS': [f"{df['logS'].mean():.2f}"],
        'Highly Soluble (logS > -2)': [len(df[df['logS'] > -2])],
        'Poorly Soluble (logS < -6)': [len(df[df['logS'] < -6])],
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv('dataset_summary.csv', index=False)
    print(f"\nSummary saved as 'dataset_summary.csv'")
