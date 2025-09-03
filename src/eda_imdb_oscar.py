# src/eda_imdb_oscar.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================== CONFIG ============================== #
INP_DEFAULT   = Path("data/processed/imdb_oscar.csv")
OUT_VIZ_DIR   = Path("reports/viz")
OUT_STATS_DIR = Path("reports/stats")

COL_TITLE  = "film"
COL_RATING = "IMDB_Rating"
COL_META   = "Meta_score"
COL_VOTES  = "No_of_Votes"
COL_GROSS  = "Gross"
COL_WINS   = "oscar_wins"
STAR_COLS  = ["Star1", "Star2", "Star3", "Star4"]


# ====================== TEMA E HELPERS VISUAIS ====================== #
def set_theme():
    import matplotlib as mpl
    plt.style.use("default")
    mpl.rcParams.update({
        "figure.figsize": (12, 7),
        "figure.dpi": 160,
        "axes.facecolor": "#0f172a",   # slate-900
        "figure.facecolor": "#0f172a",
        "axes.edgecolor": "#1e293b",   # slate-800
        "axes.labelcolor": "#e2e8f0",  # slate-200
        "xtick.color": "#cbd5e1",      # slate-300
        "ytick.color": "#cbd5e1",
        "text.color":  "#e2e8f0",
        "grid.color":  "#334155",      # slate-700
        "grid.alpha":  0.35,
        "axes.grid":   True,
        "font.size":   12,
        "axes.titleweight": "semibold",
    })

PALETTE = {
    "bar":   "#60a5fa",  # blue-400
    "bar2":  "#34d399",  # emerald-400
    "line":  "#f59e0b",  # amber-500
    "accent":"#f472b6",  # pink-400
}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def savefig(ax, path: Path, title: str | None = None):
    if title: ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.tight_layout()
    ax.figure.savefig(path, bbox_inches="tight")
    plt.close(ax.figure)

def add_value_labels_h(ax, values, fmt="{:.2f}", dx=0.02):
    if not len(values): return
    off = max(values) * dx if max(values) != 0 else 0.02
    for i, v in enumerate(values):
        ax.text(v + off, i, fmt.format(v), va="center", ha="left", fontsize=11, color="#e2e8f0")


# ============================ UTILIDADES ============================ #
def log1p_safe(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    s = np.where(s < 0, 0, s)
    return pd.Series(np.log1p(s))

def barh_simple(index, values, xlabel, title, color=PALETTE["bar"]) -> plt.Axes:
    fig, ax = plt.subplots()
    ax.barh(index, values, color=color, edgecolor="#1f2937")
    ax.set_xlabel(xlabel)
    add_value_labels_h(ax, list(values), fmt="{:.2f}")
    ax.set_title(title)
    return ax

def detect_genre_cols(df: pd.DataFrame) -> list[str]:
    """Detecta colunas dummy (0/1) como gêneros."""
    candidates: list[str] = []
    for c in df.columns:
        lc = c.lower()
        if any(tok in lc for tok in ["oscar", "win", "nom", "star", "director", "certificate", "runtime"]):
            continue
        uniq = pd.Series(df[c]).dropna().unique()
        if len(uniq) > 0 and set(uniq).issubset({0, 1}):
            candidates.append(c)
    return sorted(candidates)

def melt_stars(df: pd.DataFrame) -> pd.DataFrame:
    stars = [c for c in STAR_COLS if c in df.columns]
    if not stars: return pd.DataFrame(columns=["actor","film",COL_RATING,COL_WINS])
    long = (pd.melt(
        df[[COL_TITLE, COL_RATING, COL_WINS] + stars],
        id_vars=[COL_TITLE, COL_RATING, COL_WINS],
        value_vars=stars, value_name="actor")
        .dropna(subset=["actor"]))
    long = long.rename(columns={COL_TITLE:"film"})
    return long[["actor","film",COL_RATING,COL_WINS]]


# ========================== VISUALIZAÇÕES =========================== #
def hist_panel(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if COL_RATING in df: axes[0,0].hist(pd.to_numeric(df[COL_RATING], errors="coerce").dropna(), bins=30, edgecolor="white"); axes[0,0].set_title("IMDB_Rating")
    if COL_META   in df: axes[0,1].hist(pd.to_numeric(df[COL_META], errors="coerce").dropna(),   bins=30, edgecolor="white"); axes[0,1].set_title("Meta_score")
    if COL_VOTES  in df: axes[1,0].hist(log1p_safe(df[COL_VOTES]).dropna(), bins=30, edgecolor="white"); axes[1,0].set_title("log1p(No_of_Votes)")
    if COL_GROSS  in df: axes[1,1].hist(log1p_safe(df[COL_GROSS]).dropna(), bins=30, edgecolor="white"); axes[1,1].set_title("log1p(Gross)")
    for ax in axes.flat: ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)

def viz_rating_by_genre(genre_summary: pd.DataFrame, out: Path, topn=10):
    top = (genre_summary.sort_values("n_films", ascending=False)
           .head(topn).sort_values("mean_rating"))
    ax = barh_simple(top["genre"], top["mean_rating"], "Nota média IMDB",
                     "Nota média por gênero (Top por contagem)")
    savefig(ax, out)

def viz_win_rate_by_genre(genre_summary: pd.DataFrame, out: Path, topn=10):
    top = (genre_summary.sort_values("oscar_win_rate", ascending=False)
           .head(topn).sort_values("oscar_win_rate"))
    ax = barh_simple(top["genre"], top["oscar_win_rate"],
                     "Taxa de vitórias (wins / filmes)",
                     "Top gêneros por taxa de vitórias no Oscar",
                     color=PALETTE["bar2"])
    savefig(ax, out)

def viz_gross_by_oscar_nomination(df: pd.DataFrame, out: Path):
    """
    Gráfico de barras comparando o faturamento médio de filmes
    que competiram vs. não competiram no Oscar.
    """
    if "oscar_nominations" not in df.columns:
        print("Aviso: coluna 'oscar_nominations' não encontrada. Pulando gráfico.")
        return

    # Tratar a coluna Gross
    df["Gross"] = pd.to_numeric(df["Gross"], errors="coerce").fillna(0)

    # Criar uma coluna binária para a competição no Oscar
    df["Competiu_Oscar"] = np.where(df["oscar_nominations"] > 0, "Competiu", "Não Competiu")

    # Calcular a média de faturamento por categoria
    gross_by_category = df.groupby("Competiu_Oscar")["Gross"].mean().sort_values()

    # Criar o gráfico de barras
    fig, ax = plt.subplots()
    ax.barh(gross_by_category.index, gross_by_category.values, color=[PALETTE["bar"], PALETTE["bar2"]])
    ax.set_xlabel("Faturamento Médio (Gross)")
    ax.set_title("Faturamento Médio de Filmes que Competiram vs. Não Competiram no Oscar")

    # Adicionar os rótulos de valor
    add_value_labels_h(ax, gross_by_category.values, fmt="${:,.0f}")
    
    savefig(ax, out)
    print(f"[ok] Gerado gráfico de faturamento por competição no Oscar: {out}")


def viz_box_rating_winners(df: pd.DataFrame, out: Path):
    """Boxplot IMDB_Rating de vencedores vs não, sem FutureWarning (usa hue)."""
    try:
        import seaborn as sns
    except Exception:
        print("[info] seaborn não disponível; pulando boxplot winners vs others.")
        return
    if COL_RATING not in df or COL_WINS not in df: return

    tmp = df.copy()
    tmp["winner_bin"] = (pd.to_numeric(tmp[COL_WINS], errors="coerce").fillna(0) > 0)\
                        .map({False:"Não-vencedor",True:"Vencedor"})

    fig, ax = plt.subplots(figsize=(8,6))
    sns.boxplot(
        data=tmp,
        x="winner_bin",
        y=COL_RATING,
        hue="winner_bin",                       # hue explícito
        legend=False,                           # sem legenda redundante
        ax=ax,
        palette={"Não-vencedor":PALETTE["bar"], "Vencedor":PALETTE["bar2"]},
        showcaps=True,
        fliersize=0
    )
    ax.set_xlabel(""); ax.set_ylabel("IMDB_Rating")
    savefig(ax, out, "Distribuição da nota IMDB: vencedores vs não-vencedores")

def viz_scatter_reg(df: pd.DataFrame, x, y, out: Path, logx=False, logy=False, title=None):
    xdata = pd.to_numeric(df[x], errors="coerce")
    ydata = pd.to_numeric(df[y], errors="coerce")
    m = xdata.notna() & ydata.notna()
    xdata, ydata = xdata[m], ydata[m]
    if len(xdata) < 5: return
    if logx: xdata = np.log1p(np.where(xdata < 0, 0, xdata))
    if logy: ydata = np.log1p(np.where(ydata < 0, 0, ydata))
    fig, ax = plt.subplots()
    ax.scatter(xdata, ydata, s=14, alpha=0.35, color="#93c5fd")
    coeff = np.polyfit(xdata, ydata, 1)
    xp = np.linspace(xdata.min(), xdata.max(), 100); yp = np.polyval(coeff, xp)
    ax.plot(xp, yp, color=PALETTE["line"], lw=2)
    r = np.corrcoef(xdata, ydata)[0, 1]
    savefig(ax, out, (title or f"{y} × {x}") + f"  (r={r:.2f})")

def viz_corr_heatmap(df: pd.DataFrame, out: Path, cols: list[str]):
    """Heatmap divergente (pink↔blue) centrado em 0, padronizado com o tema."""
    try:
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    except Exception:
        print("[info] seaborn/matplotlib extras indisponíveis; pulando heatmap.")
        return

    numdf = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = numdf.corr(numeric_only=True)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = LinearSegmentedColormap.from_list(
        "pink_blue_div", ["#f472b6", "#0f172a", "#60a5fa"]  # negativo, centro, positivo
    )
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    fig, ax = plt.subplots(figsize=(11, 10))
    import seaborn as sns
    sns.heatmap(
        corr, mask=mask, cmap=cmap, norm=norm, annot=True, fmt=".2f",
        linewidths=0.6, linecolor="#1e293b", square=True,
        cbar_kws={"shrink": 0.85, "ticks": [-1, -0.5, 0, 0.5, 1]}, ax=ax
    )
    ax.set_title("Correlação (variáveis numéricas)", pad=14, weight="semibold")
    ax.tick_params(axis="x", rotation=35); ax.tick_params(axis="y", rotation=0)
    savefig(ax, out)

def viz_top_actors_by_presence(df: pd.DataFrame, out: Path, topn=20):
    long = melt_stars(df)
    if long.empty: return
    cnt = (long.groupby("actor")["film"].nunique()
                 .sort_values().tail(topn))
    ax = barh_simple(cnt.index, cnt.values, "Qtde de filmes",
                     f"Top {topn} atores com mais filmes")
    savefig(ax, out)

def viz_top_actors_by_wins(df: pd.DataFrame, out: Path, topn=15):
    long = melt_stars(df)
    if long.empty: return
    long["_won"] = (pd.to_numeric(long[COL_WINS], errors="coerce").fillna(0) > 0)
    wins = (long[long["_won"]].groupby("actor")["film"].nunique()
            .sort_values().tail(topn))
    ax = barh_simple(wins.index, wins.values, "Filmes vencedores",
                     f"Top {topn} atores por filmes vencedores no Oscar",
                     color=PALETTE["bar2"])
    savefig(ax, out)

def viz_actor_mean_rating(df: pd.DataFrame, out: Path, min_films=3, topn=15):
    long = melt_stars(df)
    if long.empty: return
    long[COL_RATING] = pd.to_numeric(long[COL_RATING], errors="coerce")
    agg = (long.groupby("actor")
                .agg(n_films=("film","nunique"),
                     mean_rating=(COL_RATING,"mean"))
                .query("n_films >= @min_films")
                .sort_values(["mean_rating","n_films"], ascending=[False,False])
                .head(topn).sort_values("mean_rating"))
    if agg.empty: return
    ax = barh_simple(agg.index, agg["mean_rating"].values,
                     "Nota média IMDB",
                     f"Atores com maior nota média (mín. {min_films} filmes)")
    savefig(ax, out)

def viz_actor_genre_heatmap(df: pd.DataFrame, out: Path,
                            min_films=5, topk_actors=20, topk_genres=10):
    try:
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
    except Exception:
        print("[info] seaborn indisponível; pulando heatmap ator×gênero.")
        return
    genre_cols = detect_genre_cols(df)
    if not genre_cols: return
    long = melt_stars(df)
    if long.empty: return
    base = df[[COL_TITLE]+genre_cols].drop_duplicates()
    m = long.merge(base, left_on="film", right_on=COL_TITLE, how="left")
    grp = m.groupby("actor")[genre_cols].sum(min_count=1)
    grp["n_films"] = m.groupby("actor")["film"].nunique()
    grp = grp.query("n_films >= @min_films").drop(columns="n_films")
    if grp.empty: return
    top_genres = grp.sum().sort_values(ascending=False).head(topk_genres).index.tolist()
    grp = grp[top_genres]
    pct = grp.div(grp.sum(axis=1), axis=0).fillna(0)
    top_actors = pct.sum(axis=1).sort_values(ascending=False).head(topk_actors).index
    pct = pct.loc[top_actors]
    cmap = LinearSegmentedColormap.from_list("dark_blue", ["#0f172a","#1d4ed8","#60a5fa"])
    fig, ax = plt.subplots(figsize=(max(8,len(top_genres)*0.9), max(8,len(top_actors)*0.45)))
    sns.heatmap(pct, cmap=cmap, vmin=0, vmax=1, annot=False,
                cbar_kws={"shrink":0.85,"label":"participação (%)"})
    ax.set_xlabel("Gênero"); ax.set_ylabel("Ator")
    ax.set_title("Mix de gêneros por ator (proporção dos filmes)")
    savefig(ax, out)


# ============================== PIPELINE ============================== #
def run(
    inp: Path = INP_DEFAULT,
    out_viz: Path = OUT_VIZ_DIR,
    out_stats: Path = OUT_STATS_DIR,
    top_genres: int = 10,
) -> None:
    set_theme()
    ensure_dir(out_viz); ensure_dir(out_stats)

    if not inp.exists():
        raise FileNotFoundError(f"CSV não encontrado: {inp}")
    df = pd.read_csv(inp)
    print(f"[ok] Base: {df.shape[0]} linhas, {df.shape[1]} colunas")

    # 1) Distribuições
    hist_panel(df, out_viz / "distributions.png")

    # 2) Gêneros
    genre_cols = detect_genre_cols(df)
    genre_rows = []
    for g in genre_cols:
        sub = df[df[g]==1]
        genre_rows.append({
            "genre":g,
            "n_films":len(sub),
            "mean_rating":pd.to_numeric(sub.get(COL_RATING), errors="coerce").mean(),
            "mean_gross": pd.to_numeric(sub.get(COL_GROSS), errors="coerce").mean(),
            "mean_votes": pd.to_numeric(sub.get(COL_VOTES), errors="coerce").mean(),
            "oscar_win_rate": (pd.to_numeric(sub.get(COL_WINS,0), errors="coerce").fillna(0)>0).mean() if len(sub)>0 else 0.0,
        })
    genre_summary = pd.DataFrame(genre_rows).sort_values("n_films", ascending=False)
    genre_summary.to_csv(out_stats / "genres_summary.csv", index=False)
    if not genre_summary.empty:
        viz_rating_by_genre(genre_summary, out_viz/"rating_by_genre.png", topn=top_genres)
        viz_win_rate_by_genre(genre_summary, out_viz/"oscar_win_rate_by_genre.png", topn=top_genres)

    # Adicionando a nova visualização aqui
    if "oscar_nominations" in df.columns and "Gross" in df.columns:
        viz_gross_by_oscar_nomination(df, out_viz / "gross_by_oscar_nomination.png")
    
    # 3) Winners vs others
    if COL_RATING in df.columns and COL_WINS in df.columns:
        viz_box_rating_winners(df, out_viz/"box_imdb_winner_vs_other.png")

    # 4) Correlações / relações
    if COL_VOTES in df.columns and COL_GROSS in df.columns:
        viz_scatter_reg(df, COL_VOTES, COL_GROSS, out_viz/"scatter_log_votes_log_gross.png",
                        logx=True, logy=True, title="log(Gross) × log(Votes)")
    num_cols = [c for c in [COL_RATING,COL_META,COL_VOTES,COL_GROSS,"Runtime"] if c in df.columns]
    if len(num_cols)>=3:
        viz_corr_heatmap(df, out_viz/"corr_numeric.png", num_cols)

    # 5) Atores
    viz_top_actors_by_presence(df, out_viz/"actors_top_presence.png", topn=20)
    viz_top_actors_by_wins(df, out_viz/"actors_top_wins.png", topn=15)
    viz_actor_mean_rating(df, out_viz/"actors_best_mean_rating.png", min_films=3, topn=15)
    viz_actor_genre_heatmap(df, out_viz/"actors_genre_heatmap.png", min_films=5, topk_actors=20, topk_genres=10)

    # 6) HTML
    html = [
        "<html><head><meta charset='utf-8'><title>IMDB+Oscar — EDA</title>",
        "<style>",
        "body{font-family:system-ui,Arial;margin:22px;background:#0f172a;color:#e2e8f0}",
        "h1{margin:0 0 10px 0;font-weight:700}",
        "h2{margin-top:26px}",
        "img{max-width:100%;border:1px solid #1e293b;border-radius:8px}",
        "a{color:#60a5fa;text-decoration:none}",
        "</style></head><body>",
        "<h1>IMDB + Oscar — EDA</h1>",
        "<p>Resumo visual gerado automaticamente. Se alguma figura não aparece, provavelmente a coluna correspondente não estava na base.</p>",
    ]
    figs = [
        ("Distribuições", "distributions.png"),
        ("Nota média por gênero", "rating_by_genre.png"),
        ("Taxa de vitórias por gênero", "oscar_win_rate_by_genre.png"),
        ("Faturamento vs. Competição no Oscar", "gross_by_oscar_nomination.png"),
        ("Vencedores vs não-vencedores (IMDB_Rating)", "box_imdb_winner_vs_other.png"),
        ("log(Gross) × log(Votes)", "scatter_log_votes_log_gross.png"),
        ("Correlação numérica", "corr_numeric.png"),
        ("Atores – mais presentes", "actors_top_presence.png"),
        ("Atores – mais vitórias (filmes vencedores)", "actors_top_wins.png"),
        ("Atores – melhor nota média (mín. 3 filmes)", "actors_best_mean_rating.png"),
        ("Atores × Gênero – participação", "actors_genre_heatmap.png"),
    ]
    for title, fname in figs:
        p = out_viz / fname
        if p.exists():
            html += [f"<h2>{title}</h2>", f"<img src='{p.name}'/>"]
    html += [
        "<hr/><p style='opacity:.7'>Gerado por <code>src/eda_imdb_oscar.py</code></p>",
        "</body></html>"
    ]
    (out_viz / "index.html").write_text("\n".join(html), encoding="utf-8")

    print(f"[ok] figures → {out_viz}")
    print(f"[ok] tables  → {out_stats}")


# ================================ CLI ================================= #
def parse_args():
    ap = argparse.ArgumentParser(description="EDA do imdb_oscar: gêneros (dummies) e atores.")
    ap.add_argument("--inp", default=str(INP_DEFAULT))
    ap.add_argument("--out-viz", default=str(OUT_VIZ_DIR))
    ap.add_argument("--out-stats", default=str(OUT_STATS_DIR))
    ap.add_argument("--top-genres", type=int, default=10)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        inp=Path(args.inp),
        out_viz=Path(args.out_viz),
        out_stats=Path(args.out_stats),
        top_genres=args.top_genres,
    )