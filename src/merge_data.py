import pandas as pd
import os

def transform_genres(input_path="data/processed/imdb_clean.csv", output_path="data/processed/genres_encodded.csv"):
    """
    Lê o arquivo imdb_clean, transforma a coluna Genre em colunas booleanas
    separadas para cada gênero e salva o resultado em um novo CSV.
    """
    # Garante que o diretório de destino exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Carrega os dados
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {input_path}")
        return

    # 2. Renomeia a coluna 'Series_Title' para 'film'
    df = df.rename(columns={'Series_Title': 'film'})
    
    # 3. Prepara a coluna 'Genre' para one-hot encoding
    # A coluna Genre vem como uma string com múltiplos valores (ex: "Action, Adventure")
    # Para transformá-la, a dividimos em linhas separadas
    df_genres = df[['film', 'Genre']].copy()
    df_genres['Genre'] = df_genres['Genre'].str.split(', ')
    df_genres = df_genres.explode('Genre')
    df_genres['Genre'] = df_genres['Genre'].str.strip()

    # 4. Cria as colunas booleanas para cada gênero usando get_dummies
    df_one_hot = pd.get_dummies(df_genres, columns=['Genre'], prefix='', prefix_sep='').groupby('film').max()

    # 5. Salva o novo DataFrame no arquivo CSV
    df_one_hot.to_csv(output_path, index=True)
    
    print(f"[OK] Dados transformados salvos em: {output_path}")

if __name__ == "__main__":
    transform_genres()