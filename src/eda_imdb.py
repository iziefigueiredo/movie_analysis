# src/eda_imdb.py
import os
import argparse
import warnings
import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols

INP_DEFAULT = "data/processed/imdb_clean.csv"
OUT_DIR = "reports/stats"

# ---------- utils ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def col_exists(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns

def get_primary_genre(s: str | float) -> str:
    if isinstance(s, str) and s.strip():
        return s.split(",")[0].strip()
    return "Unknown"

def log1p_safe(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return np.log1p(x.clip(lower=0))

# ---------- tests ----------
def anova_oneway(df: pd.DataFrame, target: str, factor: str) -> pd.DataFrame:
    """ANOVA: target ~ C(factor)"""
    model = ols(f"{target} ~ C({factor})", data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    return table

def regression_simple(df: pd.DataFrame, x: str, y: str, add_const=True):
    """OLS simples y ~ x (retorna summary string)"""
    X = pd.to_numeric(df[x], errors="coerce")
    Y = pd.to_numeric(df[y], errors="coerce")
    mask = X.notna() & Y.notna()
    X = X[mask]
    Y = Y[mask]
    if add_const:
        X = sm.add_constant(X)
    mdl = sm.OLS(Y, X).fit()
    return mdl

def chi2_independence(df: pd.DataFrame, a: str, b: str):
    contingency = pd.crosstab(df[a], df[b])
    chi2, p, dof, expected = chi2_contingency(contingency)
    return contingency, chi2, p, dof, expected

# ---------- pipeline ----------
def run_all(
    path: str = INP_DEFAULT,
    out_dir: str = OUT_DIR,
    do_anova: bool = True,
    do_reg: bool = True,
    do_chi2: bool = True
):
    warnings.filterwarnings("ignore")
    ensure_dir(out_dir)
    df = pd.read_csv(path)

    # checagens básicas de colunas
    needed_any = {
        "IMDB_Rating": ["IMDB_Rating", "imdb_rating", "Rating"],
        "Meta_score":  ["Meta_score", "metascore", "MetaScore"],
        "Gross":       ["Gross", "gross"],
        "No_of_Votes": ["No_of_Votes", "votes", "NumVotes"],
        "Certificate": ["Certificate", "certificate", "_cert"],
        "Genre":       ["Genre", "genre", "Genres"],
    }
    # mapear nomes existentes
    name_map = {}
    for canonical, alts in needed_any.items():
        for c in alts:
            if c in df.columns:
                name_map[canonical] = c
                break

    # criar Genre_primary para ANOVA/chi2
    if "Genre" in name_map:
        df["Genre_primary"] = df[name_map["Genre"]].apply(get_primary_genre)

    # ---------- ANOVA ----------
    if do_anova and "IMDB_Rating" in name_map:
        anova_targets = [("Genre_primary", "anova_genre.csv"),
                         (name_map.get("Certificate", None), "anova_certificate.csv")]
        for factor, fname in anova_targets:
            if factor and factor in df.columns:
                try:
                    table = anova_oneway(df, target=name_map["IMDB_Rating"], factor=factor)
                    table.to_csv(os.path.join(out_dir, fname))
                    print(f"[OK] ANOVA {factor} → {os.path.join(out_dir, fname)}")
                except Exception as e:
                    print(f"[WARN] ANOVA falhou ({factor}): {e}")

    # ---------- Regressões simples ----------
    if do_reg:
        # 1) IMDB_Rating ~ Meta_score
        if "IMDB_Rating" in name_map and "Meta_score" in name_map:
            try:
                mdl = regression_simple(df, x=name_map["Meta_score"], y=name_map["IMDB_Rating"])
                with open(os.path.join(out_dir, "reg_imdb_vs_metascore.txt"), "w") as f:
                    f.write(mdl.summary().as_text())
                print(f"[OK] Regressão IMDB_Rating ~ Meta_score → reports/stats/reg_imdb_vs_metascore.txt")
            except Exception as e:
                print(f"[WARN] Regressão (IMDB ~ Meta_score) falhou: {e}")

        # 2) log(Gross) ~ log(No_of_Votes)
        if "Gross" in name_map and "No_of_Votes" in name_map:
            try:
                df["_log_gross"] = log1p_safe(df[name_map["Gross"]])
                df["_log_votes"] = log1p_safe(df[name_map["No_of_Votes"]])
                mdl2 = regression_simple(df, x="_log_votes", y="_log_gross")
                with open(os.path.join(out_dir, "reg_log_gross_vs_log_votes.txt"), "w") as f:
                    f.write(mdl2.summary().as_text())
                print(f"[OK] Regressão log(Gross) ~ log(Votes) → reports/stats/reg_log_gross_vs_log_votes.txt")
            except Exception as e:
                print(f"[WARN] Regressão (log Gross ~ log Votes) falhou: {e}")

    # ---------- Qui-quadrado (categ x categ) ----------
    if do_chi2 and "Genre_primary" in df.columns and "Certificate" in name_map:
        try:
            cont, chi2, p, dof, expected = chi2_independence(df, "Genre_primary", name_map["Certificate"])
            cont.to_csv(os.path.join(out_dir, "chi2_genre_certificate_contingency.csv"))
            pd.DataFrame({"chi2":[chi2], "p_value":[p], "dof":[dof]}).to_csv(
                os.path.join(out_dir, "chi2_genre_certificate_stats.csv"), index=False
            )
            print(f"[OK] Qui-quadrado Genre x Certificate → reports/stats/chi2_*")
        except Exception as e:
            print(f"[WARN] Qui-quadrado falhou: {e}")

    # ---------- extra: salvar correlação numérica (tabela) ----------
    try:
        num_df = df.select_dtypes(include=["number"])
        corr = num_df.corr(numeric_only=True)
        corr.to_csv(os.path.join(out_dir, "corr_numeric.csv"))
        print(f"[OK] Matriz de correlação numérica → reports/stats/corr_numeric.csv")
    except Exception as e:
        print(f"[WARN] Correlação numérica falhou: {e}")

def parse_args():
    p = argparse.ArgumentParser(description="EDA estatística complementar ao ydata (IMDB)")
    p.add_argument("--inp", default=INP_DEFAULT)
    p.add_argument("--skip_anova", action="store_true")
    p.add_argument("--skip_reg", action="store_true")
    p.add_argument("--skip_chi2", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_all(
        path=args.inp,
        do_anova=not args.skip_anova,
        do_reg=not args.skip_reg,
        do_chi2=not args.skip_chi2
    )
