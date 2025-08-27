# src/exploratory_analysis.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


#======== Configurações entrada/saída ========
INP = "data/processed/imdb_clean.csv"
FIGDIR = "reports/figures"


#======== Padronização de estilo ========
PALETTE = {
    "dark":  "#0A2342",   # fundo
    "bar":   "#4FC3F7",   # barras
    "edge":  "#0288D1",   # borda barra
    "text":  "#ECEFF1",   # texto
    "grid":  "#607D8B"    # grid
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["dark"],
    "axes.facecolor":   PALETTE["dark"],
    "axes.edgecolor":   PALETTE["text"],
    "axes.labelcolor":  PALETTE["text"],
    "xtick.color":      PALETTE["text"],
    "ytick.color":      PALETTE["text"],
    "text.color":       PALETTE["text"],
    "axes.titlecolor":  PALETTE["text"],
    "grid.color":       PALETTE["grid"],
    "grid.alpha":       0.3
})

#======== Salvar plot em reports/figures ========
def savefig(name: str):
    os.makedirs(FIGDIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close()

#======== Análise exploratória ========

# Histograma das notas IMDB
def imdb_hist(df):
    df["IMDB_Rating"].hist(
        bins=15,
        color=PALETTE["bar"],
        edgecolor=PALETTE["edge"],
        alpha=0.9
    )
    plt.title("Distribuição das notas IMDB")
    plt.xlabel("IMDB Rating")
    plt.ylabel("Frequência")
    plt.grid(True, linestyle="--", alpha=0.4)
    savefig("imdb_hist")

# Boxplot das notas IMDB
def imdb_boxplot(df):
    df.boxplot(
        column="IMDB_Rating",
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor=PALETTE["bar"], color=PALETTE["edge"]),
        medianprops=dict(color=PALETTE["edge"]),
        whiskerprops=dict(color=PALETTE["edge"]),
        capprops=dict(color=PALETTE["edge"]),
        flierprops=dict(markerfacecolor=PALETTE["bar"], markeredgecolor=PALETTE["edge"])
    )
    plt.title("Boxplot das notas IMDB")
    plt.ylabel("IMDB Rating")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    savefig("imdb_boxplot")


# Scatter plot: Gross vs No_of_Votes (log-log)
def gross_votes(df):
    
    plt.figure(figsize=(8,6))
    plt.scatter(
        np.log1p(df["No_of_Votes"]),
        np.log1p(df["Gross"]),
        alpha=0.6,
        s=30,
        color=PALETTE["bar"],
        edgecolor=PALETTE["edge"],
        linewidth=0.5
    )
    plt.title("Relação entre Votos e Bilheteria (log-log)")
    plt.xlabel("log(1 + Número de Votos)")
    plt.ylabel("log(1 + Bilheteria)")
    savefig("gross_votes")


# Scatter plot: Gross vs IMDB_Rating (log-linear)
def imdb_gross(df):

    plt.figure(figsize=(8,6))
    plt.scatter(
        np.log1p(df["Gross"]),   
        df["IMDB_Rating"],           
        alpha=0.6,
        s=30,
        c=PALETTE["bar"],
        edgecolors=PALETTE["edge"],
        linewidths=0.5
    )
    plt.title("Relação entre Bilheteria e Nota IMDB")
    plt.xlabel("Bilheteria (escala log)")
    plt.ylabel("Nota IMDB")
    savefig("imdb_gross")


# Matriz de correlação
def corr_matrix(df: pd.DataFrame):
   
    # colormap divergente: azul → branco → ciano
    cmap = LinearSegmentedColormap.from_list(
        "cmap",
        [PALETTE["edge"], "white", PALETTE["bar"]]
    )

    plt.figure(figsize=(8,6))
    num_df = df.select_dtypes(include=["int64", "float64", "Int64"])
    corr = num_df.corr(numeric_only=True)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,                
        cbar=True,
        linewidths=0.3,
        linecolor=PALETTE["dark"],
        annot_kws={"color": "black", "size": 10}  
    )

    plt.title("Matriz de Correlação (variáveis numéricas)")
    savefig("corr_matrix")




if __name__ == "__main__":
    df = pd.read_csv(INP)
    imdb_hist(df)
    imdb_boxplot(df)
    gross_votes(df)
    imdb_gross(df)
    corr_matrix(df)

    print(f"Figuras salvas em {FIGDIR}/")
