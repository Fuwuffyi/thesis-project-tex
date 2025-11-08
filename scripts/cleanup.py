import os
import re
import pandas as pd
from pathlib import Path
from shutil import copy2
import numpy as np

# --- Configuration ---
pattern = re.compile(r"Scene_(\d+)_(opengl|vulkan)_.+_(frames|system)\.csv")
input_dir = Path("data")
output_dir = Path("data_clean")
first_n_rows_to_drop = 20
outlier_std_threshold = 2

# --- Helpers ---
def remove_outliers(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Removes rows where specified numeric columns have outliers beyond threshold.
    Uses IQR method in addition to std dev for more robust outlier detection.
    """
    if df.empty:
        return df
    
    # Select columns to check for outliers
    if columns is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    else:
        numeric_cols = [col for col in columns if col in df.columns]
    
    if not numeric_cols:
        return df
    
    mask = pd.Series([True] * len(df), index=df.index)
    
    for col in numeric_cols:
        # Z-score method
        col_mean = df[col].mean()
        col_std = df[col].std()
        if col_std > 0:
            z_scores = np.abs((df[col] - col_mean) / col_std)
            mask &= (z_scores <= outlier_std_threshold)
        
        # IQR method for additional robustness
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            mask &= (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
    
    return df[mask]

def clean_frames_file(filepath: Path, output_path: Path):
    """Clean frames CSV by removing initial rows and outliers."""
    try:
        df = pd.read_csv(filepath)
        
        if df.empty:
            print(f"  Warning: Empty dataframe in {filepath.name}")
            return
        
        # Drop initial warmup frames
        if len(df) > first_n_rows_to_drop:
            df = df.iloc[first_n_rows_to_drop:]
        
        # Focus outlier removal on key performance metrics
        key_metrics = ['FrameTime(ms)', 'FPS']
        available_metrics = [col for col in key_metrics if col in df.columns]
        
        if available_metrics:
            original_len = len(df)
            df = remove_outliers(df, columns=available_metrics)
            removed = original_len - len(df)
            if removed > 0:
                print(f"  Removed {removed} outlier rows from {filepath.name}")
        
        # Save cleaned data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  ✓ Cleaned {filepath.name}: {len(df)} frames")
        
    except Exception as e:
        print(f"  Error processing {filepath.name}: {e}")

def process_system_file(input_path: Path, output_path: Path):
    """Copy system info files with validation."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        copy2(input_path, output_path)
        print(f"  ✓ Copied {input_path.name}")
    except Exception as e:
        print(f"  Error copying {input_path.name}: {e}")

# --- Main ---
print("=== Starting Data Cleanup ===\n")

file_count = 0
for subdir, _, files in os.walk(input_dir):
    subdir_path = Path(subdir)
    relative_subdir = subdir_path.relative_to(input_dir) if subdir_path != input_dir else Path(".")
    
    if files:
        print(f"Processing: {relative_subdir}")
    
    for filename in files:
        if pattern.match(filename):
            input_path = subdir_path / filename
            rel_path = input_path.relative_to(input_dir)
            output_path = output_dir / rel_path

            if filename.endswith("frames.csv"):
                clean_frames_file(input_path, output_path)
                file_count += 1
            elif filename.endswith("system.csv"):
                process_system_file(input_path, output_path)
                file_count += 1

print(f"\n=== Cleanup Complete ===")
print(f"Total files processed: {file_count}")
print(f"Output directory: {output_dir}")
