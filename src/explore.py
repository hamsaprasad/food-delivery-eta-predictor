import os
import pandas as pd

def main():
    # Resolve path to delivery_data.csv relative to this script or current working directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "delivery_data.csv")
    
    if not os.path.exists(data_path):
        # Fallback to direct relative path
        data_path = "data/delivery_data.csv"
        
    print(f"Loading dataset from: {data_path}\n")
    df = pd.read_csv(data_path)
    
    print("=" * 60)
    print("1. DATASET SHAPE")
    print("=" * 60)
    print(f"Total Rows: {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]}")
    print(f"Shape tuple: {df.shape}\n")
    
    print("=" * 60)
    print("2. DATA TYPES")
    print("=" * 60)
    print(df.dtypes)
    print()
    
    print("=" * 60)
    print("3. MISSING VALUE COUNTS")
    print("=" * 60)
    missing = df.isnull().sum()
    print(missing)
    total_missing = missing.sum()
    print(f"\nTotal Missing Values across all columns: {total_missing}")
    print()
    
    print("=" * 60)
    print("4. SUMMARY STATISTICS: Time_taken_min & Road_Distance_km")
    print("=" * 60)
    cols = ["Time_taken_min", "Road_Distance_km"]
    # Check if cols exist
    available_cols = [c for c in cols if c in df.columns]
    if available_cols:
        stats = df[available_cols].describe()
        print(stats)
    else:
        print(f"Columns {cols} not found in dataset!")
    print("=" * 60)

if __name__ == "__main__":
    main()
