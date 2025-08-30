import pandas as pd
import ast

def load_tmdb(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_tmdb(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
   
    # Drop duplicados e nulos
    df = df.drop_duplicates()
    df = df.dropna()

    # Drop colunas irrelevantes
    df = df.drop(columns=["runtime", "genres", "homepage", "id", "original_language", "overview", "spoken_languages"])
    
    
    #======== Padronização ======== 
   
   #Coluna keywords 
    df["keywords"] = df["keywords"].fillna("[]")

    # transformar de string para lista de dicts
    df["keywords"] = df["keywords"].apply(ast.literal_eval)

    # extrair só os "name"
    df["keywords_clean"] = df["keywords"].apply(
        lambda x: [d["name"] for d in x if "name" in d]
    )

    #juntar em uma única string separada por vírgulas
    df["keywords"] = df["keywords_clean"].apply(lambda x: ", ".join(x)) 
    
    #descartar a coluna auxiliar
    df = df.drop(columns=["keywords_clean"])

    # garantir que não seja nulo
    df["production_companies"] = df["production_companies"].fillna("[]")

    # transformar string em lista de dicts
    df["production_companies"] = df["production_companies"].apply(ast.literal_eval)

    # extrair só os nomes
    df["production_companies_clean"] = df["production_companies"].apply(
        lambda x: [d["name"] for d in x if "name" in d]
    )

    # juntar em uma única string
    df["production_companies"] = df["production_companies_clean"].apply(lambda x: ", ".join(x))

    # descartar a coluna auxiliar
    df = df.drop(columns=["production_companies_clean"])

    # Coluna production_countries
    # garantir que não seja nulo
    df["production_countries"] = df["production_countries"].fillna("[]")

    # transformar string JSON → lista de dicts
    df["production_countries"] = df["production_countries"].apply(ast.literal_eval)

    # extrair só os nomesd
    df["production_countries_clean"] = df["production_countries"].apply(
        lambda x: [d["name"] for d in x if "name" in d]
    )

    # juntar em string separada por vírgulas
    df["production_countries"] = df["production_countries_clean"].apply(lambda x: ", ".join(x))

    # descartar auxiliar
    df = df.drop(columns=["production_countries_clean"])   
    
    return df

if __name__ == "__main__":
    raw_path = "data/raw/tmdb.csv"
    processed_path = "data/processed/tmdb_clean.csv"

    df_raw = load_tmdb(raw_path)
    df_clean = clean_tmdb(df_raw)

    df_clean.to_csv(processed_path, index=False)
    print("Antes:", df_raw.shape, "Depois:", df_clean.shape)
    print(f"TMDB limpo salvo em: {processed_path}")
