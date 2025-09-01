import os
import re
import pandas as pd
import unidecode

# -------------------- util: chave canônica --------------------
def normalize_title(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).casefold().strip()
    s = unidecode.unidecode(s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# ----------------- 1) one-hot de gêneros (IMDB) ----------------
def transform_genres(
    input_path="data/processed/imdb_clean.csv",
    output_path="data/processed/genres_encodded.csv"  # mantém seu nome
):
    """
    Lê o imdb_clean, transforma Genre em colunas booleanas por filme
    e salva o resultado. Gera também title_norm para merges futuros.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {input_path}")
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
    print(f"[OK] Dados transformados salvos em: {output_path}")

# --------------- 2) IMDB + gêneros ------------------
def merge_imdb_data(
    genres_path="data/processed/genres_encodded.csv",
    imdb_path="data/processed/imdb_clean.csv",
    output_path="data/processed/imdb_merged.csv"
):
    """
    Une o one-hot de gêneros ao IMDB limpo.
    Chave: title_norm (mais estável que 'film').
    """
    try:
        df_genres = pd.read_csv(genres_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {genres_path}")
        return

    try:
        df_imdb = pd.read_csv(imdb_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {imdb_path}")
        return

    df_imdb = df_imdb.rename(columns={'Series_Title': 'film'})
    df_imdb['title_norm'] = df_imdb['film'].map(normalize_title)

    cols_to_keep = ['title_norm', 'film', 'Released_Year', 'Runtime', 'IMDB_Rating', 'No_of_Votes', 'Gross', 'Overview', "Director", "Star1", "Star2", "Star3", "Star4", "Meta_score"]
    cols_to_keep = [c for c in cols_to_keep if c in df_imdb.columns]
    base = df_imdb[cols_to_keep].drop_duplicates(subset=['title_norm'])

    df_merged = pd.merge(base, df_genres, on='title_norm', how='left')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    print(f"[OK] Dados do IMDB unidos e salvos em: {output_path}")

# -------- 3) Oscar agregado (por title_norm)---------
def transform_oscar_data(oscar_path="data/processed/oscar_clean.csv"):
    """
    Agrega Oscar por filme, criando contagens de indicações e vitórias
    + flags de 'Best Picture' (indicado/ganhou). Usa title_norm.
    Retorna DataFrame agregado.
    """
    try:
        df_oscar = pd.read_csv(oscar_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {oscar_path}")
        return None

    if 'film' not in df_oscar.columns:
        print("Erro: oscar_clean.csv precisa ter a coluna 'film'.")
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
    main_path="data/processed/imdb_merged.csv",            # agora junta direto no IMDB
    df_oscar_aggregated=None,
    output_path="data/processed/imdb_oscar.csv"            # saída focada: IMDB + Oscar
):
    """
    Une IMDB_merged com o agregado do Oscar por title_norm (LEFT JOIN).
    """
    try:
        df_main = pd.read_csv(main_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {main_path}")
        return

    if df_oscar_aggregated is None:
        print("Erro: DataFrame agregado do Oscar não fornecido.")
        return

    # garante que title_norm exista em df_main (deve existir da etapa anterior)
    if 'title_norm' not in df_main.columns:
        df_main['title_norm'] = df_main['film'].map(normalize_title)

    df_merged = pd.merge(df_main, df_oscar_aggregated, on='title_norm', how='left')

    #drop coluna auxiliar
    df_merged = df_merged.drop(columns=['title_norm'])
    
    # preenchimento das métricas do Oscar
    for c in ['oscar_nominations', 'oscar_wins', 'best_picture_nom', 'best_picture_win']:
        if c in df_merged.columns:
            df_merged[c] = df_merged[c].fillna(0).astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    print(f"[OK] Dados do Oscar unidos e salvos em: {output_path}")

# -------------------------- runner ----------------------
if __name__ == "__main__":
    transform_genres()
    merge_imdb_data()

    df_oscar_agg = transform_oscar_data()
    if df_oscar_agg is not None:
        merge_oscar_data(df_oscar_aggregated=df_oscar_agg)

    print("Pipeline (IMDB + Oscar opcional) concluído.")
