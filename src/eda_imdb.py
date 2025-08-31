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

# ================================== config ==================================
INP_DEFAULT = Path("data/processed/imdb_clean.csv")
OUT_STATS   = Path("reports/stats")
OUT_VIZ     = Path("reports/viz")

# colunas conhecidas no imdb_clean.csv
COL_IMDB   = "IMDB_Rating"
COL_META   = "Meta_score"
COL_GROSS  = "Gross"
COL_VOTES  = "No_of_Votes"
COL_CERT   = "Certificate"
COL_GENRE  = "Genre"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# =================================== utils ==================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def get_primary_genre(s: str | float) -> str:
    return s.split(",")[0].strip() if isinstance(s, str) and s.strip() else "Unknown"

def log1p_safe(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").fillna(0)
    return np.log1p(np.where(x >= 0, x, 0))

def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")

# =================================== stats ==================================
def anova_oneway(df: pd.DataFrame, target: str, factor: str) -> pd.DataFrame:
    """ANOVA: target ~ C(factor) + eta² (tamanho de efeito)."""
    model = ols(f"{target} ~ C({factor})", data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    try:
        ss_effect = float(table.loc[f"C({factor})", "sum_sq"])
        eta2 = ss_effect / float(table["sum_sq"].sum())
    except Exception:
        eta2 = np.nan
    table.loc["eta_sq", ["sum_sq", "df", "F", "PR(>F)"]] = [eta2, np.nan, np.nan, np.nan]
    return table

def regression_simple(df: pd.DataFrame, x: str, y: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = pd.to_numeric(df[x], errors="coerce")
    Y = pd.to_numeric(df[y], errors="coerce")
    m = X.notna() & Y.notna()
    X, Y = X[m], Y[m]
    if X.nunique() < 2 or Y.nunique() < 2:
        raise ValueError("Sem variação suficiente para OLS.")
    X = sm.add_constant(X)
    return sm.OLS(Y, X).fit()

def chi2_independence(df: pd.DataFrame, a: str, b: str):
    cont = pd.crosstab(df[a], df[b])
    chi2, p, dof, expected = chi2_contingency(cont)
    if expected.min() < 5:
        logging.warning("Qui-quadrado: há células com esperado < 5.")
    return cont, chi2, p, dof

# ==================================== viz ===================================
def bar_genre_rating(df: pd.DataFrame, rating_col: str, topn: int, out: Path) -> Path | None:
    if "Genre_primary" not in df.columns or rating_col not in df.columns:
        return None
    g = (df.groupby("Genre_primary")[rating_col]
           .agg(["count", "mean", "std"])
           .sort_values("count", ascending=False)
           .head(topn).sort_values("mean"))
    fig, ax = plt.subplots(figsize=(11, 7))
    # escala de cor por média
    denom = (g["mean"].max() - g["mean"].min()) + 1e-9
    ax.barh(
        g.index, g["mean"], xerr=g["std"].fillna(0),
        color=plt.cm.viridis((g["mean"] - g["mean"].min()) / denom)
    )
    ax.set_xlabel("IMDB_Rating (média)"); ax.set_ylabel("Gênero (top por contagem)")
    ax.set_title("Média de IMDB_Rating por gênero (± DP)")
    for i, (c, m) in enumerate(zip(g["count"], g["mean"])):
        ax.text(m, i, f"  {m:.2f}  (n={c})", va="center")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig); return out

def scatter_fit(df: pd.DataFrame, xcol: str, ycol: str, out: Path, title: str, logx=False, logy=False) -> Path | None:
    if xcol not in df.columns or ycol not in df.columns:
        return None
    x = pd.to_numeric(df[xcol], errors="coerce"); y = pd.to_numeric(df[ycol], errors="coerce")
    if logx: x = log1p_safe(x)
    if logy: y = log1p_safe(y)
    m = x.notna() & y.notna(); x, y = x[m], y[m]
    if len(x) < 5: return None
    coef = np.polyfit(x, y, 1); xp = np.linspace(x.min(), x.max(), 100); yp = np.polyval(coef, xp)
    r = np.corrcoef(x, y)[0, 1] if x.std() and y.std() else np.nan
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, alpha=0.35, s=18); ax.plot(xp, yp, lw=2)
    ax.set_title(f"{title}  (r={r:.2f})"); ax.set_xlabel(xcol + (" (log1p)" if logx else "")); ax.set_ylabel(ycol + (" (log1p)" if logy else ""))
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig); return out

def heatmap_contingency(df: pd.DataFrame, a: str, b: str, out: Path, min_group: int = 20) -> Path | None:
    if a not in df.columns or b not in df.columns: return None
    A = df[a].copy()
    vc_a = A.value_counts(); vc_b = df[b].value_counts()
    A = A[A.isin(vc_a[vc_a >= min_group].index)]
    B = df.loc[A.index, b]; B = B[B.isin(vc_b[vc_b >= min_group].index)]
    A = A.loc[B.index]
    if A.nunique() < 2 or B.nunique() < 2: return None
    cont = pd.crosstab(A, B).astype(float)
    cont_pct = cont.div(cont.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(cont_pct.values, cmap="YlOrRd")
    ax.set_xticks(range(cont_pct.shape[1])); ax.set_xticklabels(cont_pct.columns, rotation=45, ha="right")
    ax.set_yticks(range(cont_pct.shape[0])); ax.set_yticklabels(cont_pct.index)
    for i in range(cont_pct.shape[0]):
        for j in range(cont_pct.shape[1]):
            ax.text(j, i, f"{cont_pct.values[i, j]:.0f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="% dentro do gênero")
    ax.set_title(f"Distribuição (%) de {b} por {a}")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig); return out

def make_html(outdir: Path, figs: dict[str, Path | None]) -> None:
    html = [
        "<html><head><meta charset='utf-8'><title>IMDB – Visualizações</title>",
        "<style>body{font-family:system-ui,Arial;margin:24px} h2{margin-top:28px} img{max-width:100%;border:1px solid #ddd;border-radius:8px}</style>",
        "</head><body><h1>IMDB – Visualizações</h1>"
    ]
    order = [
        ("IMDB_Rating por gênero (média ± DP)", "bar_genre"),
        ("IMDB_Rating  Meta_score", "sc_imdb_meta"),
        ("log(Gross)  log(Votes)", "sc_gross_votes"),
        ("Heatmap: Genre  Certificate", "heatmap_gc"),
    ]
    for title, key in order:
        p = figs.get(key)
        if p:
            html += [f"<h2>{title}</h2>", f"<img src='{Path(p).name}'/>"]
    html += ["</body></html>"]
    (outdir / "index.html").write_text("\n".join(html), encoding="utf-8")

# ================================ pipeline ===================================
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
    """Executa análises e (opcional) visualizações; retorna caminhos gerados."""
    warnings.filterwarnings("ignore")
    path, out_stats, out_viz = Path(path), Path(out_stats), Path(out_viz)
    ensure_dir(out_stats); 
    if do_viz: ensure_dir(out_viz)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path).copy()

    # derivadas
    if "Genre_primary" not in df.columns:
        df["Genre_primary"] = df[COL_GENRE].apply(get_primary_genre)

    artifacts: dict[str, Path] = {}

    # =============================== ANOVA ====================================
    if do_anova:
        for factor, fname in [("Genre_primary", "anova_genre.csv"),
                              (COL_CERT,          "anova_certificate.csv")]:
            if factor in df.columns:
                sub = df[df[factor].isin(df[factor].value_counts()[lambda s: s >= min_group_size].index)]
                try:
                    if sub[factor].nunique() >= 2:
                        outp = out_stats / fname
                        anova_oneway(sub, target=COL_IMDB, factor=factor).to_csv(outp)
                        artifacts[f"anova_{factor}"] = outp
                        logging.info("ANOVA %s → %s", factor, outp)
                    else:
                        logging.warning("ANOVA pulada: %s com <2 grupos após filtro.", factor)
                except Exception as e:
                    logging.warning("ANOVA falhou (%s): %s", factor, e)

    # ============================= Regressões =================================
    if do_reg:
        try:
            outp = out_stats / "reg_imdb_vs_metascore.txt"
            save_text(outp, regression_simple(df, COL_META, COL_IMDB).summary().as_text())
            artifacts["reg_imdb_vs_metascore"] = outp
            logging.info("Reg IMDB ~ Meta → %s", outp)
        except Exception as e:
            logging.warning("Reg IMDB ~ Meta falhou: %s", e)

        try:
            tmp = df.copy()
            tmp["_log_gross"] = log1p_safe(tmp[COL_GROSS])
            tmp["_log_votes"] = log1p_safe(tmp[COL_VOTES])
            outp = out_stats / "reg_log_gross_vs_log_votes.txt"
            save_text(outp, regression_simple(tmp, "_log_votes", "_log_gross").summary().as_text())
            artifacts["reg_log_gross_vs_log_votes"] = outp
            logging.info("Reg log(Gross) ~ log(Votes) → %s", outp)
        except Exception as e:
            logging.warning("Reg log(Gross) ~ log(Votes) falhou: %s", e)

    # ============================= Qui-quadrado ===============================
    if do_chi2:
        try:
            a, b = "Genre_primary", COL_CERT
            sub = df[df[a].isin(df[a].value_counts()[lambda s: s >= min_group_size].index)]
            sub = sub[sub[b].isin(sub[b].value_counts()[lambda s: s >= min_group_size].index)]
            if sub[a].nunique() >= 2 and sub[b].nunique() >= 2:
                cont, chi2, p, dof = chi2_independence(sub, a, b)
                cont.to_csv(out_stats / "chi2_genre_certificate_contingency.csv")
                pd.DataFrame({"chi2":[chi2], "p_value":[p], "dof":[dof]}).to_csv(
                    out_stats / "chi2_genre_certificate_stats.csv", index=False)
                artifacts["chi2_contingency"] = out_stats / "chi2_genre_certificate_contingency.csv"
                artifacts["chi2_stats"] = out_stats / "chi2_genre_certificate_stats.csv"
                logging.info("Chi² %s × %s → %s", a, b, artifacts["chi2_stats"])
            else:
                logging.warning("Chi² pulado: categorias insuficientes após filtro.")
        except Exception as e:
            logging.warning("Chi² falhou: %s", e)

    # ================================ visual ==================================
    if do_viz:
        figs: dict[str, Path | None] = {}
        figs["bar_genre"] = bar_genre_rating(df, COL_IMDB, topn, OUT_VIZ / "rating_by_genre.png")
        figs["sc_imdb_meta"] = scatter_fit(df, COL_META, COL_IMDB,
                                           OUT_VIZ / "imdb_vs_meta.png", "IMDB_Rating × Meta_score")
        tmp = df.copy()
        tmp["_log_gross"] = log1p_safe(tmp[COL_GROSS])
        tmp["_log_votes"] = log1p_safe(tmp[COL_VOTES])
        figs["sc_gross_votes"] = scatter_fit(tmp, "_log_votes", "_log_gross",
                                             OUT_VIZ / "log_gross_vs_log_votes.png",
                                             "log(Gross) × log(Votes)")
        figs["heatmap_gc"] = heatmap_contingency(df, "Genre_primary", COL_CERT,
                                                 OUT_VIZ / "heatmap_genre_certificate.png")
        make_html(OUT_VIZ, figs)
        logging.info("Visualizações em %s (index.html).", OUT_VIZ)

    return artifacts

# ==================================== CLI ====================================
def parse_args():
    p = argparse.ArgumentParser(description="EDA estatística/visual IMDB (colunas fixas)")
    p.add_argument("--inp", default=str(INP_DEFAULT))
    p.add_argument("--out-stats", default=str(OUT_STATS))
    p.add_argument("--out-viz", default=str(OUT_VIZ))
    p.add_argument("--skip-anova", action="store_true")
    p.add_argument("--skip-reg", action="store_true")
    p.add_argument("--skip-chi2", action="store_true")
    p.add_argument("--skip-viz", action="store_true")
    p.add_argument("--topn", type=int, default=12)
    p.add_argument("--min-group-size", type=int, default=20)
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
 