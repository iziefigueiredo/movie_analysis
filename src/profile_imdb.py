# src/profile_imdb.py
import os
import argparse
import pandas as pd
from ydata_profiling import ProfileReport

INP_DEFAULT = "data/processed/imdb_clean.csv"
OUT_DIR_DEFAULT = "reports"

def run_profile(
    inp: str = INP_DEFAULT,
    out_dir: str = OUT_DIR_DEFAULT,
    html_name: str = "eda_imdb_profile.html",
    pdf_name: str | None = None,
    sample: int | None = 3000,
    minimal: bool = True,
    calc_corr: bool = False,
    calc_interactions: bool = False,
    keep_missing_heatmap: bool = True
):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(inp)

    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)

    kwargs = {
        "title": "IMDB – EDA com ydata-profiling",
        "minimal": minimal,
        "explorative": not minimal,
        "interactions": {
            "continuous": calc_interactions,
            "targets": []
        },
        "missing_diagrams": {
            "heatmap": keep_missing_heatmap,
            "dendrogram": False
        },
    }

    kwargs["correlations"] = {
        "auto": {"calculate": calc_corr},
        "pearson": {"calculate": calc_corr},
        "spearman": {"calculate": calc_corr},
        "kendall": {"calculate": calc_corr},
        "phi_k": {"calculate": calc_corr},
        "cramers": {"calculate": calc_corr},
    }

    profile = ProfileReport(df, **kwargs)

    html_out = os.path.join(out_dir, html_name)
    profile.to_file(html_out)
    print(f"[OK] HTML salvo em: {html_out}")

    if pdf_name:
        try:
            from weasyprint import HTML
            pdf_out = os.path.join(out_dir, pdf_name)
            HTML(filename=html_out).write_pdf(pdf_out)
            print(f"[OK] PDF salvo em: {pdf_out}")
        except Exception as e:
            print(f"[INFO] PDF não gerado (opcional). Motivo: {e}")

def parse_args():
    p = argparse.ArgumentParser(description="Gera EDA do IMDB com ydata-profiling")
    p.add_argument("--inp", default=INP_DEFAULT)
    p.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    p.add_argument("--html_name", default="eda_imdb_profile.html")
    p.add_argument("--pdf_name", default=None)
    p.add_argument("--sample", type=int, default=3000)
    p.add_argument("--minimal", action="store_true", help="Usar modo minimalista")
    p.add_argument("--full", action="store_true", help="Modo explorative (desliga minimal)")
    p.add_argument("--corr", action="store_true", help="Calcular correlações")
    p.add_argument("--inter", action="store_true", help="Calcular interações")
    p.add_argument("--no_missing_heatmap", action="store_true", help="Não mostrar heatmap de missing")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_profile(
        inp=args.inp,
        out_dir=args.out_dir,
        html_name=args.html_name,
        pdf_name=args.pdf_name,
        sample=args.sample,
        minimal=not args.full,
        calc_corr=args.corr,
        calc_interactions=args.inter,
        keep_missing_heatmap=not args.no_missing_heatmap
    )
