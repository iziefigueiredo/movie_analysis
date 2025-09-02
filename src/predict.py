import os
import pickle
import numpy as np
import pandas as pd

# =========================== CONFIG =========================== #
MODEL_DIR       = "models"
MODEL_PATH      = os.path.join(MODEL_DIR, "imdb_predictor.pkl")
FEATURES_PATH   = os.path.join(MODEL_DIR, "model_features.pkl")
TARGET          = "IMDB_Rating"
STAR_COLS       = ["Star1", "Star2", "Star3", "Star4"]

# =========================== FUNÇÕES =========================== #
def load_model_and_features():
    """Carrega o modelo treinado e a lista de features."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        print(f"Erro: Arquivos do modelo não encontrados. Execute 'modeling.py' primeiro.")
        return None, None
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, 'rb') as f:
        features = pickle.load(f)
        
    return model, features

def predict_imdb_rating(
    df: pd.DataFrame,
    model,
    features: list
) -> pd.DataFrame:
    """
    Recebe um DataFrame de filmes e utiliza o modelo para prever a
    nota do IMDB. Abordagem otimizada para performance.
    """
    # 1. Prepara a entrada criando um dicionário de dados
    data_for_prediction = {}
    
    # 2. Popula o dicionário com os dados necessários
    for feature in features:
        # Lida com colunas numéricas (popula com dados do input)
        if feature in ["Runtime", "No_of_Votes", "Gross", "Meta_score"]:
            data_for_prediction[feature] = pd.to_numeric(df.get(feature, pd.Series([np.nan] * len(df))), errors='coerce').fillna(0).tolist()
        # Lida com colunas de gênero e ator (dummies)
        else:
            is_present = pd.Series([0] * len(df))
            if feature.startswith("actor_"):
                actor_name = feature.replace("actor_", "")
                is_present = df.apply(
                    lambda row: (actor_name in str(row.get("Star1", "")) or 
                                 actor_name in str(row.get("Star2", "")) or 
                                 actor_name in str(row.get("Star3", "")) or 
                                 actor_name in str(row.get("Star4", ""))), axis=1
                )
            elif feature.startswith("director_"):
                director_name = feature.replace("director_", "")
                is_present = (df["Director"] == director_name)
            else: # Colunas de gênero
                is_present = df.get(feature, pd.Series([0] * len(df)))
                
            data_for_prediction[feature] = is_present.fillna(0).astype(int).tolist()

    # 3. Cria o DataFrame de uma vez a partir do dicionário
    df_pred = pd.DataFrame(data_for_prediction, columns=features)
    
    # 4. Faz a predição
    df_pred["predicted_imdb_rating"] = model.predict(df_pred[features])
    
    # 5. Retorna o resultado com as colunas originais
    return pd.concat([df.reset_index(drop=True), df_pred["predicted_imdb_rating"]], axis=1)

# =========================== RUNNER =========================== #
if __name__ == "__main__":
    model, features = load_model_and_features()
    
    if model and features:
        print("[Início] Fazendo previsões com o modelo carregado...")

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
                "Star2": "Tim Robbins",
                "Star3": "Bob Gunton",
                "Star4": "William Sadler"# Adicionando ator para o exemplo

            },
            {
                "film": "The Matrix",
                "Director": "Lana Wachowski",
                "Runtime": 136,
                "No_of_Votes": 1900000,
                "Gross": 171479930,
                "Meta_score": 88,
                "Action": 1, "Sci-Fi": 1,
                "Star1": "Keanu Reeves"
            },
            {
                "film": "A Quiet Place",
                "Director": "John Krasinski",
                "Runtime": 90,
                "No_of_Votes": 490000,
                "Gross": 188024361,
                "Meta_score": 82,
                "Drama": 1, "Horror": 1
            }
        ])
        
        predictions = predict_imdb_rating(new_movies, model, features)
        
        print("\nPrevisões geradas:")
        print(predictions[["film", "predicted_imdb_rating"]])
        
        print("\n[Fim] Predição concluída.")