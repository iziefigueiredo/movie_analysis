import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Carrega o CSV de dados brutos."""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Faz limpeza básica das colunas principais."""
    

    # Drop duplicados e nulos
    df = df.drop_duplicates()
    df = df.dropna()

    #Drop colunas irrelevantes
    df = df.drop(columns=["Unnamed: 0"])

    df["Runtime"] = df["Runtime"].str.extract(r"(\d+)").astype(int)

    # Released_Year -> numérico 
    df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce").astype("Int64")

    return df

if __name__ == "__main__":
    raw_path = "data/raw/imdb.csv"
    processed_path = "data/processed/imdb_clean.csv"
    df_raw = load_data(raw_path)
    df_clean = clean_data(df_raw)

    print("Antes:", df_raw.head(2))
    print("Depois:", df_clean.head(2))

    # salva versão limpa
    df_clean.to_csv(processed_path, index=False)
    print(f"Dados processados salvos em {processed_path}")
