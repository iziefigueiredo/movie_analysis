import os
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)  # como os anteriores

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # 0) Padroniza nomes de colunas primeiro (snake_case)
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", "_", regex=True)
          .str.replace(r"[^\w]", "", regex=True)
    )

    # 1) Remove colunas irrelevantes (já em snake_case)
    df = df.drop(
        columns=["year", "ceremony", "film_year", "note", "nomid", "nomineeids", "filmid", "citation"],
        errors="ignore"
    )

    # 2) Tira espaços nas pontas das strings
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    # 3) Remove linhas totalmente vazias (todas colunas NaN/vazias)
    df = df.replace({"": pd.NA}).dropna(how="all")

    # 4) Mantém apenas linhas com 'film' (chave mínima para análise)
    if "film" in df.columns:
        df = df.dropna(subset=["film"])
        # normaliza 'film' minimamente para evitar diferenças bobas
        df["film"] = df["film"].str.replace(r"\s+", " ", regex=True)

    # 5) Converte 'winner' para boolean se existir
    if "winner" in df.columns:
        df["winner"] = (
            df["winner"].astype(str).str.lower().map({
                "true": True, "false": False, "yes": True, "no": False, "1": True, "0": False
            }).fillna(False)
        )

    # 6) Remove colunas constantes (zero variância)
    nunique = df.nunique(dropna=False)
    drop_const = nunique[nunique <= 1].index.tolist()
    if drop_const:
        df = df.drop(columns=drop_const)

    # 7) Remove duplicatas por subconjunto (use o que existir)
    subset_keys = [c for c in ["film", "category", "name"] if c in df.columns]
    if subset_keys:
        df = df.drop_duplicates(subset=subset_keys)
    else:
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
