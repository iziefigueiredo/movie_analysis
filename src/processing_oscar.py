import re
import unidecode
import os
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)  

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", "_", regex=True)
          .str.replace(r"[^\w]", "", regex=True)
    )

    df = df.drop(
        columns=["ceremony", "year", "note", "nomid", "nomineeids", "filmid", "citation", "multifilmnomination", "nominees", "detail", "class", "category"],
        errors="ignore"
    )


    # remove linhas onde 'film' é nulo ou vazio
    df = df.dropna(subset=["film"])
    df = df[df["film"].str.strip() != ""]

    # Tira espaços nas pontas das strings
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    # Remove linhas totalmente vazias (todas colunas NaN/vazias)
    df = df.replace({"": pd.NA}).dropna(how="all")

    # Mantém apenas linhas com 'film' 
    if "film" in df.columns:
        df = df.dropna(subset=["film"])
        df["film"] = df["film"].str.replace(r"\s+", " ", regex=True)

    # Converte 'winner' para boolean 
    df["winner"] = df["winner"].fillna("False").astype(bool)
    
    if "winner" in df.columns:
        df["winner"] = (
            df["winner"].astype(str).str.lower().map({
                "true": True, "false": False, "yes": True, "no": False, "1": True, "0": False
            }).fillna(False)
        )

   

    df = df.drop_duplicates()

    return df

if __name__ == "__main__":
    raw_path = "data/raw/oscar.csv"
    processed_path = "data/processed/oscar_clean.csv"
    os.makedirs("data/processed", exist_ok=True)

    df_raw = load_data(raw_path)
    df_clean = clean_data(df_raw)

    print("Antes:", df_raw.shape, df_raw.columns.tolist()[:8])
    print("Depois:", df_clean.shape, df_clean.columns.tolist())

    df_clean.to_csv(processed_path, index=False, encoding="utf-8")
    print(f"[OK] Dados processados salvos em {processed_path}")
