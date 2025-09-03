import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# =========================== CONFIG =========================== #
MODEL_DIR = Path("models")
FEATURES_PATH = MODEL_DIR / "features.pkl"
STAR_COLS = ["Star1", "Star2", "Star3", "Star4"]


# =========================== FUNÇÕES =========================== #
def load_models():
    """Carrega os modelos treinados e a lista de features."""
    models = {}
    try:
        models["RandomForest"] = joblib.load(MODEL_DIR / "rf_model.pkl")
        models["GradientBoosting"] = joblib.load(MODEL_DIR / "gb_model.pkl")
        models["LinearRegression"] = joblib.load(MODEL_DIR / "lr_model.pkl")
    except FileNotFoundError as e:
        print(f"Erro ao carregar um dos modelos: {e}. Certifique-se de executar o 'modeling.py' primeiro.")
        return None
    return models


def load_features():
    """Carrega a lista de features usadas no treino."""
    try:
        return joblib.load(FEATURES_PATH)
    except FileNotFoundError as e:
        print(f"Erro: Arquivo de features não encontrado em {FEATURES_PATH}. Certifique-se de que ele foi salvo pelo 'modeling.py'.")
        return None


def prepare_data(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Prepara os dados de entrada para corresponderem às features do modelo."""
    prepared_data = {}
    for feature in features:
        # 1. Trata colunas numéricas
        if feature in ["No_of_Votes", "Meta_score", "Runtime", "Gross"]:
            prepared_data[feature] = pd.to_numeric(df.get(feature, pd.Series([np.nan] * len(df))), errors='coerce').fillna(0).tolist()
        
        # 2. Trata colunas de atores
        elif feature.startswith("actor_"):
            actor_name = feature.replace("actor_", "")
            is_present = df[STAR_COLS].apply(lambda row: actor_name in str(row.astype(str).values), axis=1)
            prepared_data[feature] = is_present.astype(int).tolist()

        # 3. Trata colunas de diretores
        elif feature.startswith("director_"):
            director_name = feature.replace("director_", "")
            # Verifica se a coluna 'Director' existe antes de tentar acessá-la
            if "Director" in df.columns:
                is_present = (df["Director"] == director_name).fillna(False)
            else:
                is_present = pd.Series([False] * len(df))
            prepared_data[feature] = is_present.astype(int).tolist()
        
        # 4. Trata colunas de gêneros (dummys)
        else: 
            is_present = df.get(feature, pd.Series([0] * len(df)))
            prepared_data[feature] = is_present.fillna(0).astype(int).tolist()
    
    return pd.DataFrame(prepared_data, columns=features)


def make_prediction(model, prepared_df: pd.DataFrame) -> np.ndarray:
    """Faz a predição usando um modelo específico."""
    return model.predict(prepared_df)


def main():
    models = load_models()
    features = load_features()

    if models is None or features is None:
        return

    # Dados para predição, incluindo "The Shawshank Redemption"
    new_movies = pd.DataFrame([
        {
            "film": "The Shawshank Redemption",
            "Director": "Frank Darabont",
            "Runtime": 142,
            "No_of_Votes": 2343110,
            "Gross": 28341469,
            "Meta_score": 80,
            "Drama": 1,
            "Star1": "Tim Robbins",
            "Star2": "Morgan Freeman",
            "Star3": "Bob Gunton",
            "Star4": "William Sadler"
        },
        {
            "film": "Inception",
            "Director": "Christopher Nolan",
            "Runtime": 148,
            "No_of_Votes": 2400000,
            "Gross": 292576195,
            "Meta_score": 74,
            "Action": 1, "Adventure": 1, "Sci-Fi": 1, "Thriller": 1,
            "Star1": "Leonardo DiCaprio", "Star2": "Joseph Gordon-Levitt"
        }
    ])
    
    # Prepara os dados uma única vez
    prepared_df = prepare_data(new_movies, features)

    print("=== Previsões para filmes de exemplo ===")
    
    # Faz a predição para cada filme usando cada modelo
    for name, model in models.items():
        predictions = make_prediction(model, prepared_df)
        print(f"\nModelo: {name}")
        for i, film in new_movies.iterrows():
            print(f"  - Filme: {film['film']:<30} | Nota Prevista: {predictions[i]:.3f}")


if __name__ == "__main__":
    main()