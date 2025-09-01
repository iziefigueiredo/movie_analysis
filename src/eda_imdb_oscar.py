# src/eda_imdb_oscar.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------- Configuração ------------------------- #
INP_DEFAULT   = Path("data/processed/imdb_oscar.csv")
OUT_VIZ_DIR   = Path("reports/viz")
OUT_STATS_DIR = Path("reports/stats")

COL_TITLE   = "film"
COL_RATING  = "IMDB_Rating"
COL_META    = "Meta_score"
COL_GROSS   = "Gross"
COL_VOTES   = "No_of_Votes"
COL_WINS    = "oscar_wins"
STAR_COLS   = ["Star1", "Star2", "Star3", "Star4"]

# ------------------------- Utils ------------------------- #
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def log1p_safe(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    s = np.where(s < 0, 0, s)
    return pd.Series(np.log1p(s))

# ------------------------- Gráficos simples ------------------------- #
def hist(ax, s, title: str, bins=30, log=False):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if log: s = log1p_safe(s)
    ax.hist(s, bins=bins, edgecolor="white")
    ax.set_title(title); ax.grid(alpha=0.3)

def boxplot(ax, group, values, title: str, order=None):
    values = pd.to_numeric(values, errors="coerce")
    m = group.notna() & values.notna()
    group, values = group[m], values[m]
    if group.nunique() < 2:
        ax.text(0.5,0.5,"dados insuf.",ha="center"); ax.axis("off"); return
    if order is None: order = group.value_counts().index.tolist()
    data = [values[group==c] for c in order]
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_title(title); ax.tick_params(axis="x", rotation=45); ax.grid(alpha=0.3)

# ------------------------- Pipeline ------------------------- #
def run(inp: Path=INP_DEFAULT, out_viz: Path=OUT_VIZ_DIR, out_stats: Path=OUT_STATS_DIR, topn=10):
    ensure_dir(out_viz); ensure_dir(out_stats)
    df = pd.read_csv(inp)
    print(f"[OK] Base lida: {df.shape[0]} linhas, {df.shape[1]} colunas")

    # ------------------ 1) Distribuições ------------------ #
    fig, axes = plt.subplots(2,2,figsize=(11,8))
    hist(axes[0,0], df[COL_RATING], "Distribuição: IMDB_Rating")
    hist(axes[0,1], df[COL_META], "Distribuição: Meta_score")
    hist(axes[1,0], df[COL_VOTES], "Distribuição: No_of_Votes (log)", log=True)
    hist(axes[1,1], df[COL_GROSS], "Distribuição: Gross (log)", log=True)
    fig.tight_layout(); fig.savefig(out_viz/"distributions.png",dpi=150); plt.close(fig)

    # ------------------ 2) Gêneros ------------------ #
    # detecta colunas dummy de gênero (0/1)
    genre_cols = [c for c in df.columns if df[c].dropna().isin([0,1]).all() and c not in [COL_WINS]]
    genre_stats = []
    for g in genre_cols:
        sub = df[df[g]==1]
        genre_stats.append({
            "genre": g,
            "n_films": len(sub),
            "mean_rating": sub[COL_RATING].mean(),
            "mean_gross": sub[COL_GROSS].mean(),
            "mean_votes": sub[COL_VOTES].mean(),
            "oscar_win_rate": sub[COL_WINS].sum()/len(sub) if len(sub)>0 else 0
        })
    genre_stats = pd.DataFrame(genre_stats).sort_values("n_films",ascending=False)
    genre_stats.to_csv(out_stats/"genres_summary.csv",index=False)

    # gráfico: rating médio por gênero (topn)
    fig, ax = plt.subplots(figsize=(10,6))
    top = genre_stats.head(topn)
    ax.barh(top["genre"], top["mean_rating"], color="skyblue")
    ax.set_xlabel("Nota média IMDB"); ax.set_title("Nota média por gênero")
    for i,v in enumerate(top["mean_rating"]): ax.text(v,i,f"{v:.2f}",va="center")
    fig.tight_layout(); fig.savefig(out_viz/"rating_by_genre.png",dpi=150); plt.close(fig)

    # ------------------ 3) Atores ------------------ #
    # junta Star1-4 numa coluna
    actors = pd.melt(df, id_vars=[COL_TITLE,COL_RATING,COL_GROSS,COL_VOTES,COL_WINS],
                        value_vars=STAR_COLS, value_name="actor").dropna(subset=["actor"])
    actor_stats = (actors.groupby("actor")
                        .agg(total_films=(COL_TITLE,"count"),
                             mean_rating=(COL_RATING,"mean"),
                             mean_gross=(COL_GROSS,"mean"),
                             wins=(COL_WINS,"sum"))
                        .sort_values("total_films",ascending=False))
    actor_stats.to_csv(out_stats/"actors_summary.csv")

    # gráfico: top atores por nº de filmes
    topa = actor_stats.head(topn)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(topa.index, topa["total_films"], color="orange")
    ax.set_title(f"Top {topn} atores mais frequentes"); ax.set_xlabel("Qtd. de filmes")
    for i,v in enumerate(topa["total_films"]): ax.text(v,i,str(v),va="center")
    fig.tight_layout(); fig.savefig(out_viz/"actors_top.png",dpi=150); plt.close(fig)

    print(f"[ok] Arquivos salvos em {out_viz} e {out_stats}")

# ------------------------- CLI ------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="EDA do imdb_oscar com gêneros e atores")
    ap.add_argument("--inp",default=str(INP_DEFAULT))
    ap.add_argument("--out-viz",default=str(OUT_VIZ_DIR))
    ap.add_argument("--out-stats",default=str(OUT_STATS_DIR))
    ap.add_argument("--topn",type=int,default=10)
    return ap.parse_args()

if __name__=="__main__":
    args = parse_args()
    run(inp=Path(args.inp), out_viz=Path(args.out_viz), out_stats=Path(args.out_stats), topn=args.topn)
