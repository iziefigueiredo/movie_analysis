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
    df_genres = df[['film', 'Genre']].copy()
    df_genres['Genre'] = df_genres['Genre'].str.split(', ')
    df_genres = df_genres.explode('Genre')
    df_genres['Genre'] = df_genres['Genre'].str.strip()

    # 4. Cria as colunas booleanas para cada gênero usando get_dummies
    df_one_hot = pd.get_dummies(df_genres, columns=['Genre'], prefix='', prefix_sep='').groupby('film').max()

    # 5. Salva o novo DataFrame no arquivo CSV
    df_one_hot.to_csv(output_path, index=True)
    
    print(f"[OK] Dados transformados salvos em: {output_path}")


def merge_imdb_data(genres_path="data/processed/genres_encodded.csv", imdb_path="data/processed/imdb_clean.csv", output_path="data/processed/imdb_merged.csv"):
    """
    Lê os arquivos de gêneros codificados e dados limpos do IMDB,
    e os une com base no nome do filme, salvando o resultado em um novo CSV.
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

    # Renomear a coluna para coincidir com o DataFrame de gêneros
    df_imdb = df_imdb.rename(columns={'Series_Title': 'film'})

    # Selecionar as colunas desejadas do DataFrame do IMDB
    cols_to_merge = ['film', 'Released_Year', 'Runtime', 'IMDB_Rating', 'No_of_Votes', 'Gross']
    df_imdb_selected = df_imdb[cols_to_merge]

    # Unir os DataFrames
    df_merged = pd.merge(df_genres, df_imdb_selected, on='film', how='inner')

    # Salvar o DataFrame unido em um novo CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_csv(output_path, index=False)

    print(f"[OK] Dados do IMDB unidos e salvos em: {output_path}")


def merge_tmdb_imdb(imdb_path="data/processed/imdb_merged.csv", tmdb_path="data/processed/tmdb_clean.csv", output_path="data/processed/merge_imdb_tmdb.csv"):
    """
    Lê os arquivos unidos do IMDB e limpos do TMDB, e os une com base no nome do filme.
    """
    try:
        df_imdb = pd.read_csv(imdb_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {imdb_path}")
        return

    try:
        df_tmdb = pd.read_csv(tmdb_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {tmdb_path}")
        return

    # Renomear a coluna para coincidir com o DataFrame do IMDB
    df_tmdb = df_tmdb.rename(columns={'title': 'film'})

    # Selecionar as colunas desejadas do DataFrame do TMDB
    cols_to_merge = ['film', 'budget', 'revenue', 'popularity' ]
    df_tmdb_selected = df_tmdb[cols_to_merge]

    # Unir os DataFrames
    df_merged = pd.merge(df_imdb, df_tmdb_selected, on='film', how='inner')

    # Salvar o DataFrame unido em um novo CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_csv(output_path, index=False)

    print(f"[OK] Dados finais unidos e salvos em: {output_path}")





if __name__ == "__main__":
    transform_genres()
    merge_imdb_data()
    merge_tmdb_imdb()
    
    print("Pipeline de união de dados concluída.")