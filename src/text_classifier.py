import pandas as pd
import re
from collections import Counter
from pathlib import Path
import os
import nltk
from nltk.corpus import stopwords


try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# =========================== CONFIG =========================== #
DATA_PATH = Path("data/processed/imdb_oscar.csv")
TEXT_COL = "Overview"

# =========================== FUNCTIONS ========================== #
def load_data(data_path: Path) -> pd.DataFrame:
    """Loads the data and filters for necessary columns."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    df = pd.read_csv(data_path)
    if TEXT_COL not in df.columns:
        raise ValueError(f"The column '{TEXT_COL}' was not found.")
    return df.dropna(subset=[TEXT_COL]).reset_index(drop=True)

def preprocess_text(text: str) -> list[str]:
    """Performs basic text cleaning and tokenization."""
    text = re.sub(r'[^a-z\s]', '', text.lower())
    tokens = [word for word in text.split() if word not in stopwords.words('english')]
    return tokens

def detect_genres(df: pd.DataFrame) -> list[str]:
    """Detects dummy genre columns in a safer way."""
    non_genre_cols = [
        "film", "Overview", "Director", "Star1", "Star2", "Star3", "Star4", 
        "oscar_wins", "oscar_nominations", 
        "IMDB_Rating", "Released_Year", "Runtime", "Gross", "Meta_score", 
        "No_of_Votes", "Certificate", "winner", "canonicalcategory",
        "best_picture_nom", "best_picture_win"
    ]
    genre_cols = [c for c in df.columns if c not in non_genre_cols and set(df[c].dropna().unique()).issubset({0, 1})]
    return genre_cols

def analyze_tokens():
    """Analyzes the frequency of tokens by genre to infer a film's profile."""
    print("Starting genre token analysis...")
    try:
        data_frame = load_data(DATA_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return
    
    genre_cols = detect_genres(data_frame)
    if not genre_cols:
        print("Warning: No genre columns found. Analysis cannot continue.")
        return
        
    data_frame['tokens'] = data_frame[TEXT_COL].apply(preprocess_text)
    
    genre_profiles = {}
    for genre in genre_cols:
        genre_data = data_frame[data_frame[genre] == 1]
        all_tokens = [token for sublist in genre_data['tokens'] for token in sublist]
        token_counts = Counter(all_tokens).most_common(20)
        genre_profiles[genre] = token_counts
    
    print("\nGenre Token Analysis:")
    for genre, tokens in genre_profiles.items():
        print(f"\n--- Genre: {genre} ---")
        if not tokens:
            print("No tokens found.")
        else:
            for token, count in tokens:
                print(f"  - '{token}': {count} occurrences")

if __name__ == "__main__":
    analyze_tokens()