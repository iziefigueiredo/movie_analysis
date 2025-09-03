# main.py
import os
import pandas as pd
from pathlib import Path

# Importing all necessary modules from the 'src' package
from src import (
    # Data processing and merging functions
    load_imdb,
    clean_imdb,
    load_oscar,
    clean_oscar,
    transform_genres,
    merge_imdb_data,
    transform_oscar_data,
    merge_oscar_data,
    
    # Scripts for analysis and modeling
    run_profile,
    run_eda,
    run_modeling,
    run_prediction,
    run_text_analysis,
)


def main_pipeline():
    """
    Orchestrates the full data science pipeline: processing, EDA, modeling, and prediction.
    """
    print("Starting the full data science pipeline...")

    # Defining project directory paths
    base_dir = Path(__file__).parent
    data_raw_dir = base_dir / "data" / "raw"
    data_processed_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    reports_viz_dir = reports_dir / "viz"
    reports_stats_dir = reports_dir / "stats"

    # Ensure all necessary directories exist
    os.makedirs(data_raw_dir, exist_ok=True)
    os.makedirs(data_processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_viz_dir, exist_ok=True)
    os.makedirs(reports_stats_dir, exist_ok=True)

    # --- Step 1: Cleaning raw data ---
    print("\n--- 1. Cleaning raw data ---")
    df_imdb_raw_path = data_raw_dir / "imdb.csv"
    df_oscar_raw_path = data_raw_dir / "oscar.csv"

    try:
        df_imdb = clean_imdb(load_imdb(df_imdb_raw_path))
        df_oscar = clean_oscar(load_oscar(df_oscar_raw_path))
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}. "
              "Please ensure the raw CSV files are in the 'data/raw/' directory.")
        return

    df_imdb.to_csv(data_processed_dir / "imdb_clean.csv", index=False)
    df_oscar.to_csv(data_processed_dir / "oscar_clean.csv", index=False)
    print("Cleaning complete. Files saved to 'data/processed/'.")

    # --- Step 2: Merging data ---
    print("\n--- 2. Data merging and transformation ---")
    transform_genres()
    merge_imdb_data()
    df_oscar_agg = transform_oscar_data()
    if df_oscar_agg is not None:
        merge_oscar_data(df_oscar_aggregated=df_oscar_agg)
    print("Data merging complete. 'imdb_oscar.csv' file created.")

    # --- Step 3: EDA and Profiling ---
    print("\n--- 3. Exploratory Data Analysis (EDA) ---")
    run_profile(out_dir=str(reports_viz_dir))
    run_eda(out_viz=reports_viz_dir, out_stats=reports_stats_dir)
    print("Exploratory analysis complete. Reports and visualizations generated in 'reports/'.")

    # --- Step 4: Modeling ---
    print("\n--- 4. Model training and evaluation ---")
    run_modeling()
    print("Modeling complete. Models saved to 'models/'.")

    # --- Step 5: Prediction ---
    print("\n--- 5. Prediction on new data ---")
    run_prediction()
    print("Prediction complete.")

    # --- Step 6: Text Analysis ---
    print("\n--- 6. Text analysis by genre ---")
    run_text_analysis()
    print("Text analysis complete.")

    print("\n Full pipeline executed successfully!")

if __name__ == "__main__":
    main_pipeline()