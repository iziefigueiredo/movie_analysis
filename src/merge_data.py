import os
import re
import pandas as pd
import unidecode

# -------------------- util: canonical key --------------------
def normalize_title(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).casefold().strip()
    s = unidecode.unidecode(s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# ----------------- 1) one-hot for genres (IMDB) ----------------
def transform_genres(
    input_path="data/processed/imdb_clean.csv",
    output_path="data/processed/genres_encoded.csv"
):
    """
    Reads imdb_clean, transforms the 'Genre' column into boolean columns per film
    and saves the result. Also generates a 'title_norm' column for future merges.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    df = df.rename(columns={'Series_Title': 'film'})
    df['title_norm'] = df['film'].map(normalize_title)

    df_genres = df[['title_norm', 'Genre']].copy()
    df_genres['Genre'] = df_genres['Genre'].str.split(', ')
    df_genres = df_genres.explode('Genre')
    df_genres['Genre'] = df_genres['Genre'].str.strip()

    one_hot = (
        pd.get_dummies(df_genres, columns=['Genre'], prefix='', prefix_sep='')
          .groupby('title_norm')
          .max()
          .reset_index()
    )

    one_hot.to_csv(output_path, index=False)
    print(f"[OK] Transformed data saved to: {output_path}")

# --------------- 2) Merge IMDB + Genres ------------------
def merge_imdb_data(
    genres_path="data/processed/genres_encoded.csv",
    imdb_path="data/processed/imdb_clean.csv",
    output_path="data/processed/imdb_merged.csv"
):
    """
    Merges the one-hot encoded genres with the cleaned IMDB data.
    Key: 'title_norm' (more stable than 'film').
    """
    try:
        df_genres = pd.read_csv(genres_path)
    except FileNotFoundError:
        print(f"Error: File not found at {genres_path}")
        return

    try:
        df_imdb = pd.read_csv(imdb_path)
    except FileNotFoundError:
        print(f"Error: File not found at {imdb_path}")
        return

    df_imdb = df_imdb.rename(columns={'Series_Title': 'film'})
    df_imdb['title_norm'] = df_imdb['film'].map(normalize_title)

    cols_to_keep = ['title_norm', 'film', 'Released_Year', 'Runtime', 'IMDB_Rating', 'No_of_Votes', 'Gross', 'Overview', "Director", "Star1", "Star2", "Star3", "Star4", "Meta_score"]
    cols_to_keep = [c for c in cols_to_keep if c in df_imdb.columns]
    base = df_imdb[cols_to_keep].drop_duplicates(subset=['title_norm'])

    df_merged = pd.merge(base, df_genres, on='title_norm', how='left')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    print(f"[OK] IMDB data merged and saved to: {output_path}")

# -------- 3) Aggregate Oscar data (by title_norm)---------
def transform_oscar_data(oscar_path="data/processed/oscar_clean.csv"):
    """
    Aggregates Oscar data per film, creating counts for nominations and wins
    + flags for 'Best Picture' (nominated/won). Uses 'title_norm'.
    Returns an aggregated DataFrame.
    """
    try:
        df_oscar = pd.read_csv(oscar_path)
    except FileNotFoundError:
        print(f"Error: File not found at {oscar_path}")
        return None

    if 'film' not in df_oscar.columns:
        print("Error: oscar_clean.csv must have a 'film' column.")
        return None

    df_oscar['title_norm'] = df_oscar['film'].map(normalize_title)
    df_oscar['winner'] = df_oscar['winner'].astype(bool)

    is_bp = df_oscar['canonicalcategory'].str.contains('BEST PICTURE', case=False, na=False)

    agg = (
        df_oscar.groupby('title_norm')
                .agg(
                    oscar_nominations=('winner', 'size'),
                    oscar_wins=('winner', 'sum'),
                    best_picture_nom=('canonicalcategory', lambda s: int(is_bp.reindex(s.index, fill_value=False).any())),
                    best_picture_win=('winner', lambda w: int((w & is_bp.reindex(w.index, fill_value=False)).any()))
                )
                .reset_index()
    )
    return agg

def merge_oscar_data(
    main_path="data/processed/imdb_merged.csv",
    df_oscar_aggregated=None,
    output_path="data/processed/imdb_oscar.csv"
):
    """
    Merges 'imdb_merged' with the aggregated Oscar data using 'title_norm' (LEFT JOIN).
    """
    try:
        df_main = pd.read_csv(main_path)
    except FileNotFoundError:
        print(f"Error: File not found at {main_path}")
        return

    if df_oscar_aggregated is None:
        print("Error: Aggregated Oscar DataFrame not provided.")
        return

    if 'title_norm' not in df_main.columns:
        df_main['title_norm'] = df_main['film'].map(normalize_title)

    df_merged = pd.merge(df_main, df_oscar_aggregated, on='title_norm', how='left')

    df_merged = df_merged.drop(columns=['title_norm'])
    
    for c in ['oscar_nominations', 'oscar_wins', 'best_picture_nom', 'best_picture_win']:
        if c in df_merged.columns:
            df_merged[c] = df_merged[c].fillna(0).astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    print(f"[OK] Oscar data merged and saved to: {output_path}")

# -------------------------- runner ----------------------
if __name__ == "__main__":
    transform_genres()
    merge_imdb_data()

    df_oscar_agg = transform_oscar_data()
    if df_oscar_agg is not None:
        merge_oscar_data(df_oscar_aggregated=df_oscar_agg)

    print("Pipeline (IMDB + optional Oscar data) completed.")