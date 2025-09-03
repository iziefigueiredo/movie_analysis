# src/profile_imdb_oscar.py
import os
import argparse
import pandas as pd
from ydata_profiling import ProfileReport

# Default paths for input data and output directory
INP_DEFAULT = "data/processed/imdb_oscar.csv"
OUT_DIR_DEFAULT = "reports"

def run_profile(
    inp: str = INP_DEFAULT,
    out_dir: str = OUT_DIR_DEFAULT,
    html_name: str = "eda_imdb_oscar.html",
    pdf_name: str | None = None,
    sample: int | None = 3000,
    minimal: bool = True,
    calc_corr: bool = False,
    calc_interactions: bool = False,
    keep_missing_heatmap: bool = True
):
    """
    Generates a data profiling report using ydata-profiling.

    This report provides a comprehensive overview of the data, including statistics,
    distributions, correlations, and a heatmap of missing values.

   
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(inp)

    # Samples the data if a sample size is specified and the data is larger
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)

    # Configuration for the ProfileReport
    kwargs = {
        "title": "IMDB/OSCAR – EDA with ydata-profiling",
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

    # Save the HTML report
    html_out = os.path.join(out_dir, html_name)
    profile.to_file(html_out)
    print(f"[OK] HTML report saved to: {html_out}")

    # Generate and save the PDF report if a name is provided
    if pdf_name:
        try:
            from weasyprint import HTML
            pdf_out = os.path.join(out_dir, pdf_name)
            HTML(filename=html_out).write_pdf(pdf_out)
            print(f"[OK] PDF report saved to: {pdf_out}")
        except Exception as e:
            print(f"[INFO] PDF was not generated (optional). Reason: {e}")

def parse_args():
    """
    Defines and parses command-line arguments for the script.
    This allows the script to be configured from the terminal.
    """
    p = argparse.ArgumentParser(description="Generates an EDA report for IMDB/OSCAR data using ydata-profiling")
    p.add_argument("--inp", default=INP_DEFAULT)
    p.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    p.add_argument("--html_name", default="IMDB-OSCAR_profile.html")
    p.add_argument("--pdf_name", default=None)
    p.add_argument("--sample", type=int, default=3000)
    p.add_argument("--minimal", action="store_true", help="Use minimalist mode")
    p.add_argument("--full", action="store_true", help="Explorative mode (disables minimal)")
    p.add_argument("--corr", action="store_true", help="Calculate correlations")
    p.add_argument("--inter", action="store_true", help="Calculate interactions")
    p.add_argument("--no_missing_heatmap", action="store_true", help="Do not show missing values heatmap")
    return p.parse_args()

if __name__ == "__main__":
    """
    Main entry point for the script when executed from the command line.
    It parses arguments and runs the profiling function.
    """
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