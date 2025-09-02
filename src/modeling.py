import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

# =========================== CONFIG =========================== #
INP_DATA_PATH   = "data/processed/imdb_oscar.csv"
MODEL_DIR       = "models"
MODEL_PATH      = os.path.join(MODEL_DIR, "imdb_predictor.pkl")
FEATURES_PATH   = os.path.join(MODEL_DIR, "model_features.pkl")
TARGET          = "IMDB_Rating"
STAR_COLS       = ["Star1", "Star2", "Star3", "Star4"]

# =========================== FUNÇÕES =========================== #
def add_director_features(df: pd.DataFrame, n_top=50) -> tuple[pd.DataFrame, list[str]]:
    """
    Cria colunas binárias para os N diretores mais frequentes.
    """
    df = df.copy()
    top_directors = df["Director"].value_counts().head(n_top).index.tolist()
    top_directors = [d for d in top_directors if pd.notna(d)]
    
    for director in top_directors:
        df[f"director_{director}"] = (df["Director"] == director).astype(int)
    
    return df, [f"director_{d}" for d in top_directors]


def add_actor_features(df: pd.DataFrame, n_top=100) -> tuple[pd.DataFrame, list[str]]:
    """
    Cria colunas binárias (dummies) para os N atores mais frequentes.
    Usa um prefixo para evitar conflitos de nomes.
    """
    df = df.copy()
    all_stars = pd.melt(df, value_vars=STAR_COLS, value_name="actor")
    top_actors = all_stars["actor"].value_counts().head(n_top).index.tolist()
    top_actors = [a for a in top_actors if pd.notna(a)]

    for actor in top_actors:
        df[f"actor_{actor}"] = df[STAR_COLS].apply(lambda row: actor in row.values, axis=1).astype(int)
    
    return df, [f"actor_{a}" for a in top_actors]


def train_model(
    df: pd.DataFrame,
    target: str = TARGET
) -> tuple[RandomForestRegressor, list[str]]:
    """
    Prepara os dados (incluindo atores e diretores), treina um modelo de regressão e o retorna junto
    com as colunas (features) utilizadas.
    """
    df = df.copy()

    # Adiciona features de diretores e atores
    df, director_features = add_director_features(df)
    df, actor_features = add_actor_features(df)

    # Define features e target
    numerical_features = ["Runtime", "No_of_Votes", "Gross", "Meta_score"]
    genre_features = [c for c in df.columns if df[c].isin([0, 1]).all() and c not in numerical_features and c not in actor_features and c not in director_features]
    
    imputer = SimpleImputer(strategy="mean")
    df[numerical_features] = imputer.fit_transform(df[numerical_features])
    
    features = numerical_features + genre_features + actor_features + director_features
    X = df[features]
    y = df[target]

    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    print(f"Dimensões do treino: {X_train.shape}")
    print(f"Dimensões do teste: {X_test.shape}")

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Avaliação no conjunto de teste:")
    print(f"  - MAE (Mean Absolute Error): {mae:.2f}")
    print(f"  - R2 (R-squared): {r2:.2f}")

    return model, features

def save_model(model, features: list, model_path: str = MODEL_PATH, features_path: str = FEATURES_PATH):
    """Salva o modelo e a lista de features."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"[OK] Modelo salvo em: {model_path}")
    print(f"[OK] Features salvas em: {features_path}")

# =========================== RUNNER =========================== #
if __name__ == "__main__":
    if not os.path.exists(INP_DATA_PATH):
        print(f"Erro: Arquivo {INP_DATA_PATH} não encontrado. Execute a pipeline de processamento primeiro.")
    else:
        print("[Início] Treinando modelo preditivo com features de atores e diretores...")
        df = pd.read_csv(INP_DATA_PATH)
        df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce')
        model, features = train_model(df)
        save_model(model, features)
        print("[Fim] Treinamento e salvamento concluídos.")