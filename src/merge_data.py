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


def transform_oscar_data(oscar_path="data/processed/oscar_clean.csv"):
    """
    Lê o arquivo de dados brutos do Oscar e agrega o número de indicações e vitórias por filme.
    Retorna o DataFrame agregado.
    """
    try:
        df_oscar = pd.read_csv(oscar_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {oscar_path}")
        return None

    # Certifica que a coluna 'winner' é booleana
    df_oscar['winner'] = df_oscar['winner'].astype(bool)

    # Agrega os dados por filme
    df_aggregated = (
        df_oscar.groupby('film')
        .agg(
            oscar_nominations=('film', 'count'),
            oscar_wins=('winner', 'sum')
        )
        .reset_index()
    )
    return df_aggregated


def merge_oscar_data(main_path="data/processed/merge_imdb_tmdb.csv", df_oscar_aggregated=None, output_path="data/processed/final_dataset.csv"):
    """
    Lê o arquivo de dados unidos (IMDB + TMDB) e os dados agregados do Oscar,
    e os une com base no nome do filme.
    """
    try:
        df_main = pd.read_csv(main_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {main_path}")
        return

    # Certifica que os dados agregados do Oscar foram fornecidos
    if df_oscar_aggregated is None:
        print("Erro: DataFrame agregado do Oscar não fornecido.")
        return
        
    # Unir os DataFrames
    df_merged = pd.merge(df_main, df_oscar_aggregated, on='film', how='inner')

    # Salvar o DataFrame unido em um novo CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_csv(output_path, index=False)

    print(f"[OK] Dados do Oscar unidos e salvos em: {output_path}")


if __name__ == "__main__":
    transform_genres()
    merge_imdb_data()
    merge_tmdb_imdb()
    
    # Nova etapa de agregação e união do Oscar
    df_oscar_agg = transform_oscar_data()
    if df_oscar_agg is not None:
        merge_oscar_data(df_oscar_aggregated=df_oscar_agg)
    
    print("Pipeline de união de dados concluída.")