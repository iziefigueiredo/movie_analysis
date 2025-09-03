# main.py
import os
import pandas as pd
from pathlib import Path

from src import (
    # IMDB
    load_imdb,    
    clean_imdb,    
   
    # OSCAR
    load_oscar,    
    clean_oscar,   

    # Merge / features
    transform_genres,
    merge_imdb_data,
    transform_oscar_data,
    merge_oscar_data,

    # Modeling
    run_modeling,

    # Prediction
    run_prediction,
)


def main_pipeline():
    """
    Orquestra a pipeline completa de processamento, modelagem e predição.
    """
    print("Iniciando a pipeline de processamento de dados...")

    base_dir = Path(__file__).parent
    data_raw_dir = base_dir / "data" / "raw"
    data_processed_dir = base_dir / "data" / "processed"

    # Garante que os diretórios existam
    os.makedirs(data_raw_dir, exist_ok=True)
    os.makedirs(data_processed_dir, exist_ok=True)

    # 1. Limpeza dos dados brutos
    # Lembre-se que você precisa ter os arquivos .csv na pasta data/raw
    df_imdb_raw_path = data_raw_dir / "imdb.csv"
    df_oscar_raw_path = data_raw_dir / "oscar.csv"

    try:
        df_imdb = clean_imdb(pd.read_csv(df_imdb_raw_path))   
        df_oscar = clean_oscar(load_oscar(df_oscar_raw_path)) 
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado: {e}. "
              "Por favor, verifique se todos os arquivos .csv brutos estão na pasta raw.")
        return

    # Salva os arquivos limpos na pasta processed
    df_imdb.to_csv(data_processed_dir / "imdb_clean.csv", index=False)
    df_oscar.to_csv(data_processed_dir / "oscar_clean.csv", index=False)

    # 2. Transformações e uniões
    transform_genres()
    merge_imdb_data()
    

    df_oscar_agg = transform_oscar_data()
    if df_oscar_agg is not None:
        merge_oscar_data(df_oscar_aggregated=df_oscar_agg)

    print("Pipeline de união de dados concluída.")
    
    # --- Etapa 3: Modelagem ---
    print("\nIniciando o treinamento dos modelos...")
    run_modeling()
    
    # --- Etapa 4: Predição ---
    print("\nIniciando a predição com os modelos treinados...")
    run_prediction()


if __name__ == "__main__":
    main_pipeline()