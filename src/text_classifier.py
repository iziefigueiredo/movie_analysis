import pandas as pd
import re
from collections import Counter
from pathlib import Path
import os
import nltk
from nltk.corpus import stopwords

# Certifique-se de que as stopwords estão baixadas
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# =========================== CONFIG =========================== #
DATA_PATH = Path("data/processed/imdb_oscar.csv")
TEXT_COL = "Overview"

# =========================== FUNÇÕES =========================== #
def load_and_prepare_data(path: Path) -> pd.DataFrame:
    """Carrega os dados e filtra as colunas necessárias."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(path)
    if TEXT_COL not in df.columns:
        raise ValueError(f"A coluna '{TEXT_COL}' não foi encontrada.")
    return df.dropna(subset=[TEXT_COL]).reset_index(drop=True)

def preprocess_text(text: str) -> list[str]:
    """Faz a limpeza básica do texto e tokeniza."""
    text = re.sub(r'[^a-z\s]', '', text.lower())
    tokens = [word for word in text.split() if word not in stopwords.words('english')]
    return tokens

def detect_genre_cols(df: pd.DataFrame) -> list[str]:
    """Detecta as colunas de gênero (dummys) de forma mais segura."""
    # Lista de colunas não-gênero para exclusão
    non_genre_cols = [
        "film", "Overview", "Director", "Star1", "Star2", "Star3", "Star4", 
        "oscar_wins", "oscar_nominations", "title_norm", "Unnamed: 0", 
        "IMDB_Rating", "Released_Year", "Runtime", "Gross", "Meta_score", 
        "No_of_Votes", "Certificate", "winner", "canonicalcategory",
        "best_picture_nom", "best_picture_win"
    ]
    genre_cols = [c for c in df.columns if c not in non_genre_cols and set(df[c].dropna().unique()).issubset({0, 1})]
    return genre_cols

def analyze_genre_tokens():
    """Analisa a frequência de tokens por gênero para inferir o perfil do filme."""
    print("Iniciando a análise de tokens por gênero...")
    try:
        df = load_and_prepare_data(DATA_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"Erro: {e}")
        return
    
    genre_cols = detect_genre_cols(df)
    if not genre_cols:
        print("Aviso: Nenhuma coluna de gênero encontrada. A análise não pode continuar.")
        return
        
    df['tokens'] = df[TEXT_COL].apply(preprocess_text)
    
    genre_profiles = {}
    for genre in genre_cols:
        genre_df = df[df[genre] == 1]
        all_tokens = [token for sublist in genre_df['tokens'] for token in sublist]
        token_counts = Counter(all_tokens).most_common(20)
        genre_profiles[genre] = token_counts
    
    print("\nAnálise de Tokens por Gênero:")
    for genre, tokens in genre_profiles.items():
        print(f"\n--- Gênero: {genre} ---")
        if not tokens:
            print("Nenhum token encontrado.")
        else:
            for token, count in tokens:
                print(f"  - '{token}': {count} ocorrências")

if __name__ == "__main__":
    analyze_genre_tokens()