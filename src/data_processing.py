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

    #========Conversão de tipos========
    
    #Retirar " min" e converter Runtime para numérico
    df["Runtime"] = df["Runtime"].str.extract(r"(\d+)").astype(int)

    # Released_Year -> numérico 
    df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce").astype("Int64")

    # Gross -> numérico
    df["Gross"] = df["Gross"].str.replace(",", "", regex=False).astype("Int64")

    #======== Padronização de categorias ========
    
    # Certificate -> padronizar categorias
    if "Certificate" in df.columns:
        map_cert = {
            "U/A": "UA",
            "Passed": "Approved",
            "GP": "PG",
        }
        df["Certificate"] = df["Certificate"].replace(map_cert)

        main_certs = {"U", "A", "UA", "R", "PG-13", "PG", "G", "Approved", "GP"}
        df["Certificate"] = df["Certificate"].where(df["Certificate"].isin(main_certs), "Other")
    
    
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
