# src/eda_imdb.py
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ---------------------------------------------------------------------
# Configurações padrão
# ---------------------------------------------------------------------
INP_DEFAULT = Path("data/processed/imdb_clean.csv")
OUT_STATS = Path("reports/stats")
OUT_VIZ = Path("reports/viz")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def get_primary_genre(s: str | float) -> str:
    if isinstance(s, str) and s.strip():
        return s.split(",")[0].strip()
    return "Unknown"

def log1p_safe(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").fillna(0)
    x = x.where(x >= 0, 0)  # evita log de negativos
    return np.log1p(x)

def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Mapeia nomes canônicos -> nomes reais encontrados no DataFrame.
    Mantém flexibilidade para datasets com rótulos diferentes.
    """
    aliases = {
        "IMDB_Rating": ["IMDB_Rating", "imdb_rating", "Rating"],
        "Meta_score":  ["Meta_score", "metascore", "MetaScore"],
        "Gross":       ["Gross", "gross"],
        "No_of_Votes": ["No_of_Votes", "votes", "NumVotes"],
        "Certificate": ["Certificate", "certificate", "_cert"],
        "Genre":       ["Genre", "genre", "Genres"],
    }
    name_map: dict[str, str] = {}
    for canonical, alts in aliases.items():
        for c in alts:
            if c in df.columns:
                name_map[canonical] = c
                break
    return name_map

# ---------------------------------------------------------------------
# Estatística
# ---------------------------------------------------------------------
def anova_oneway(df: pd.DataFrame, target: str, factor: str) -> pd.DataFrame:
    """
    ANOVA: target ~ C(factor)
    Retorna tabela com eta^2 (tamanho de efeito).
    """
    model = ols(f"{target} ~ C({factor})", data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    # eta^2 = SS_efeito / SS_total
    try:
        ss_effect = table.loc[f"C({factor})", "sum_sq"]
        ss_total = table["sum_sq"].sum()
        eta2 = float(ss_effect) / float(ss_total) if ss_total else np.nan
    except Exception:
        eta2 = np.nan
    table.loc["eta_sq", ["sum_sq", "df", "F", "PR(>F)"]] = [eta2, np.nan, np.nan, np.nan]
    return table

def regression_simple(
    df: pd.DataFrame,
    x: str,
    y: str,
    add_const: bool = True
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """OLS simples y ~ x (limpa NA e checa variação)."""
    X = pd.to_numeric(df[x], errors="coerce")
    Y = pd.to_numeric(df[y], errors="coerce")
    m = X.notna() & Y.notna()
    X, Y = X[m], Y[m]
    if X.nunique() < 2 or Y.nunique() < 2:
        raise ValueError("Variáveis sem variação suficiente para OLS.")
    if add_const:
        X = sm.add_constant(X)
    return sm.OLS(Y, X).fit()

def chi2_independence(df: pd.DataFrame, a: str, b: str):
    """Qui-quadrado de independência; alerta se esperados < 5."""
    cont = pd.crosstab(df[a], df[b])
    chi2, p, dof, expected = chi2_contingency(cont)
    expected_min = expected.min()
    if expected_min < 5:
        logging.warning("Qui-quadrado: células com esperado < 5 (%.2f).", expected_min)
    return cont, chi2, p, dof, expected

# ---------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------
def plot_corr(df: pd.DataFrame, out: Path) -> Path | None:
    num = df.select_dtypes(include=["number"])
    if num.shape[1] < 2:
        return None
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(corr.shape[1])); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(corr.shape[0])); ax.set_yticklabels(corr.columns)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="correlação")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out

def plot_bar_genre_rating(
    df: pd.DataFrame, rating_col: str, genre_col: str, topn: int, out: Path
) -> Path | None:
    if genre_col not in df.columns or rating_col not in df.columns:
        return None
    tmp = df.assign(**{genre_col: df[genre_col].apply(get_primary_genre)})
    g = (tmp.groupby(genre_col)[rating_col]
           .agg(["count", "mean", "std"])
           .sort_values("count", ascending=False)
           .head(topn)
           .sort_values("mean"))
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(
        g.index,
        g["mean"],
        xerr=g["std"].fillna(0),
        color=plt.cm.viridis((g["mean"] - g["mean"].min()) / (g["mean"].max() - g["mean"].min() + 1e-9))
    )
    ax.set_xlabel("IMDB_Rating (média)"); ax.set_ylabel("Gênero (top por contagem)")
    ax.set_title("Média de IMDB_Rating por gênero (± DP)")
    for i, (c, m) in enumerate(zip(g["count"], g["mean"])):
        ax.text(m, i, f"  {m:.2f}  (n={c})", va="center")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out

def plot_scatter_with_fit(
    df: pd.DataFrame, xcol: str, ycol: str, out: Path, title: str, logx: bool=False, logy: bool=False
) -> Path | None:
    if xcol not in df.columns or ycol not in df.columns:
        return None
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    if logx: x = log1p_safe(x)
    if logy: y = log1p_safe(y)
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(x) < 5: return None
    coef = np.polyfit(x, y, 1); xp = np.linspace(x.min(), x.max(), 100); yp = np.polyval(coef, xp)
    r = np.corrcoef(x, y)[0, 1] if x.std() and y.std() else np.nan
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, alpha=0.35, s=18); ax.plot(xp, yp, lw=2)
    ax.set_xlabel(xcol + (" (log1p)" if logx else "")); ax.set_ylabel(ycol + (" (log1p)" if logy else ""))
    ax.set_title(f"{title}  (r={r:.2f})")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out

def plot_heatmap_contingency(
    df: pd.DataFrame, a: str, b: str, out: Path, min_group: int = 20
) -> Path | None:
    if a not in df.columns or b not in df.columns:
        return None
    A = df[a].copy()
    vc_a = A.value_counts(); vc_b = df[b].value_counts()
    A = A[A.isin(vc_a[vc_a >= min_group].index)]
    B = df.loc[A.index, b]
    B = B[B.isin(vc_b[vc_b >= min_group].index)]
    A = A.loc[B.index]
    if A.nunique() < 2 or B.nunique() < 2:
        return None
    cont = pd.crosstab(A, B).astype(float)
    cont_pct = cont.div(cont.sum(axis=1), axis=0) * 100.0
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(cont_pct.values, cmap="YlOrRd")
    ax.set_xticks(range(cont_pct.shape[1])); ax.set_xticklabels(cont_pct.columns, rotation=45, ha="right")
    ax.set_yticks(range(cont_pct.shape[0])); ax.set_yticklabels(cont_pct.index)
    for i in range(cont_pct.shape[0]):
        for j in range(cont_pct.shape[1]):
            ax.text(j, i, f"{cont_pct.values[i, j]:.0f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="% dentro do gênero")
    ax.set_title(f"Distribuição (%) de {b} por {a}")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out

def make_html(outdir: Path, figs: dict[str, Path]) -> None:
    html = [
        "<html><head><meta charset='utf-8'><title>IMDB – Visualizações</title>",
        "<style>body{font-family:system-ui,Arial;margin:24px} h2{margin-top:28px} img{max-width:100%;border:1px solid #ddd;border-radius:8px}</style>",
        "</head><body><h1>IMDB – Visualizações</h1>"
    ]
    order = [
        ("Correlação (numéricas)", "corr"),
        ("IMDB_Rating por gênero (média ± DP)", "bar_genre"),
        ("IMDB_Rating × Meta_score", "sc_imdb_meta"),
        ("log(Gross) × log(Votes)", "sc_gross_votes"),
        ("Heatmap: Genre × Certificate", "heatmap_gc"),
    ]
    for title, key in order:
        p = figs.get(key)
        if p:
            html += [f"<h2>{title}</h2>", f"<img src='{p.name}'/>"]
    html += ["</body></html>"]
    (outdir / "index.html").write_text("\n".join(html), encoding="utf-8")

# ---------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------
def run_all(
    path: str | Path = INP_DEFAULT,
    out_stats: str | Path = OUT_STATS,
    out_viz: str | Path = OUT_VIZ,
    do_anova: bool = True,
    do_reg: bool = True,
    do_chi2: bool = True,
    do_viz: bool = True,
    topn: int = 12,
    min_group_size: int = 20,
) -> dict[str, Path]:
    """
    Executa análises e (opcional) visualizações.
    Retorna dict com caminhos dos artefatos gerados.
    """
    warnings.filterwarnings("ignore")
    path = Path(path); out_stats = Path(out_stats); out_viz = Path(out_viz)
    ensure_dir(out_stats); 
    if do_viz: ensure_dir(out_viz)

    if not path.exists():
        raise FileNotFoundError(f"CSV de entrada não encontrado: {path}")

    df = pd.read_csv(path).copy()
    name = resolve_columns(df)

    # derivada de gênero
    if "Genre" in name and "Genre_primary" not in df.columns:
        df["Genre_primary"] = df[name["Genre"]].apply(get_primary_genre)

    artifacts: dict[str, Path] = {}

    # ------------------ ANOVA ------------------
    if do_anova and "IMDB_Rating" in name:
        targets = [
            ("Genre_primary", out_stats / "anova_genre.csv"),
            (name.get("Certificate"), out_stats / "anova_certificate.csv"),
        ]
        for factor, outp in targets:
            if factor and factor in df.columns:
                # filtra categorias raras (grupos pequenos geram ruído)
                vc = df[factor].value_counts()
                kept = vc[vc >= min_group_size].index
                sub = df[df[factor].isin(kept)]
                try:
                    if sub[factor].nunique() >= 2:
                        table = anova_oneway(sub, target=name["IMDB_Rating"], factor=factor)
                        table.to_csv(outp)
                        artifacts[f"anova_{factor}"] = outp
                        logging.info("ANOVA %s → %s", factor, outp)
                    else:
                        logging.warning("ANOVA pulada: %s tem < 2 grupos após filtro.", factor)
                except Exception as e:
                    logging.warning("ANOVA falhou (%s): %s", factor, e)

    # ---------------- Regressões ---------------
    if do_reg:
        # IMDB_Rating ~ Meta_score
        if "IMDB_Rating" in name and "Meta_score" in name:
            try:
                mdl = regression_simple(df, x=name["Meta_score"], y=name["IMDB_Rating"])
                outp = out_stats / "reg_imdb_vs_metascore.txt"
                outp.write_text(mdl.summary().as_text())
                artifacts["reg_imdb_vs_metascore"] = outp
                logging.info("Regressão IMDB_Rating ~ Meta_score → %s", outp)
            except Exception as e:
                logging.warning("Regressão (IMDB ~ Meta_score) falhou: %s", e)

        # log(Gross) ~ log(No_of_Votes)
        if "Gross" in name and "No_of_Votes" in name:
            try:
                df["_log_gross"] = log1p_safe(df[name["Gross"]])
                df["_log_votes"] = log1p_safe(df[name["No_of_Votes"]])
                mdl2 = regression_simple(df, x="_log_votes", y="_log_gross")
                outp = out_stats / "reg_log_gross_vs_log_votes.txt"
                outp.write_text(mdl2.summary().as_text())
                artifacts["reg_log_gross_vs_log_votes"] = outp
                logging.info("Regressão log(Gross) ~ log(Votes) → %s", outp)
            except Exception as e:
                logging.warning("Regressão (log Gross ~ log Votes) falhou: %s", e)

    # ---------------- Qui-quadrado -------------
    if do_chi2 and "Genre_primary" in df.columns and "Certificate" in name:
        try:
            a, b = "Genre_primary", name["Certificate"]
            sub = df[df[a].isin(df[a].value_counts()[lambda s: s >= min_group_size].index)]
            sub = sub[sub[b].isin(sub[b].value_counts()[lambda s: s >= min_group_size].index)]
            if sub[a].nunique() >= 2 and sub[b].nunique() >= 2:
                cont, chi2, p, dof, expected = chi2_independence(sub, a, b)
                cont_out = out_stats / "chi2_genre_certificate_contingency.csv"
                stats_out = out_stats / "chi2_genre_certificate_stats.csv"
                cont.to_csv(cont_out)
                pd.DataFrame({"chi2": [chi2], "p_value": [p], "dof": [dof]}).to_csv(stats_out, index=False)
                artifacts["chi2_contingency"] = cont_out
                artifacts["chi2_stats"] = stats_out
                logging.info("Qui-quadrado %s × %s → %s / %s", a, b, cont_out, stats_out)
            else:
                logging.warning("Qui-quadrado pulado: categorias insuficientes após filtro.")
        except Exception as e:
            logging.warning("Qui-quadrado falhou: %s", e)

    # ---------------- Visualizações ------------
    if do_viz:
        figs: dict[str, Path | None] = {}
        # 1) correlação
        figs["corr"] = plot_corr(df, OUT_VIZ / "corr_numeric.png")
        # 2) barras por gênero
        if "IMDB_Rating" in name:
            figs["bar_genre"] = plot_bar_genre_rating(
                df, name["IMDB_Rating"], "Genre_primary", topn, OUT_VIZ / "rating_by_genre.png"
            )
        # 3) scatter imdb vs meta
        if "IMDB_Rating" in name and "Meta_score" in name:
            figs["sc_imdb_meta"] = plot_scatter_with_fit(
                df, name["Meta_score"], name["IMDB_Rating"], OUT_VIZ / "imdb_vs_meta.png",
                title="IMDB_Rating × Meta_score"
            )
        # 4) scatter log gross vs log votes
        if "Gross" in name and "No_of_Votes" in name:
            tmp = df.copy()
            tmp["_log_gross"] = log1p_safe(tmp[name["Gross"]])
            tmp["_log_votes"] = log1p_safe(tmp[name["No_of_Votes"]])
            figs["sc_gross_votes"] = plot_scatter_with_fit(
                tmp, "_log_votes", "_log_gross", OUT_VIZ / "log_gross_vs_log_votes.png",
                title="log(Gross) × log(Votes)"
            )
        # 5) heatmap Genre × Certificate
        if "Certificate" in name and "Genre_primary" in df.columns:
            figs["heatmap_gc"] = plot_heatmap_contingency(
                df, "Genre_primary", name["Certificate"], OUT_VIZ / "heatmap_genre_certificate.png"
            )

        # monta HTML com o que foi gerado
        make_html(OUT_VIZ, {k: v for k, v in figs.items() if v is not None})
        logging.info("Visualizações em %s (abra o index.html)", OUT_VIZ)

    return artifacts

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="EDA estatística e visual do IMDB")
    p.add_argument("--inp", default=str(INP_DEFAULT), help="CSV de entrada")
    p.add_argument("--out-stats", default=str(OUT_STATS), help="Pasta de saída para tabelas")
    p.add_argument("--out-viz", default=str(OUT_VIZ), help="Pasta de saída para figuras/HTML")
    p.add_argument("--skip-anova", action="store_true")
    p.add_argument("--skip-reg", action="store_true")
    p.add_argument("--skip-chi2", action="store_true")
    p.add_argument("--skip-viz", action="store_true")
    p.add_argument("--topn", type=int, default=12, help="Top N gêneros por contagem (barras)")
    p.add_argument("--min-group-size", type=int, default=20, help="Tamanho mínimo por categoria (ANOVA/chi2)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_all(
        path=Path(args.inp),
        out_stats=Path(args.out_stats),
        out_viz=Path(args.out_viz),
        do_anova=not args.skip_anova,
        do_reg=not args.skip_reg,
        do_chi2=not args.skip_chi2,
        do_viz=not args.skip_viz,
        topn=args.topn,
        min_group_size=args.min_group_size,
    )
