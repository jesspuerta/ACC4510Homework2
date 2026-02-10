#!/usr/bin/env python3
"""Automated analysis for Research Question 1: awareness vs CPA intention."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf

REQUIRED_IDS = ["Q27", "Q29", "Q51", "Q53"]
OPTIONAL_IDS = ["Q31"]


def find_latest_csv(data_dir: Path) -> Path:
    csv_files = sorted(data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return csv_files[-1]


def detect_header_row(csv_path: Path, max_scan_rows: int = 6) -> int:
    rows: List[List[str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for _ in range(max_scan_rows):
            try:
                rows.append(next(reader))
            except StopIteration:
                break

    if not rows:
        raise ValueError("CSV appears empty.")

    best_idx = 0
    best_score = -1
    for idx, row in enumerate(rows):
        row_set = set(row)
        required_hits = sum(1 for qid in REQUIRED_IDS if qid in row_set)
        q_like = sum(1 for cell in row if cell.startswith("Q") and any(ch.isdigit() for ch in cell))
        score = required_hits * 10 + q_like
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def load_qualtrics_csv(csv_path: Path) -> pd.DataFrame:
    header_row = detect_header_row(csv_path)
    df = pd.read_csv(csv_path, header=header_row, dtype=str, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Drop obvious metadata rows sometimes present in Qualtrics exports.
    importid_mask = pd.Series(False, index=df.index)
    for col in [c for c in ["ResponseId", *REQUIRED_IDS, *OPTIONAL_IDS] if c in df.columns]:
        importid_mask = importid_mask | df[col].fillna("").str.contains(r"\{\"ImportId\"", regex=True)
    df = df.loc[~importid_mask].copy()

    # Remove rows with no response payload.
    payload_cols = [c for c in REQUIRED_IDS if c in df.columns]
    if payload_cols:
        has_payload = df[payload_cols].notna().any(axis=1)
        df = df.loc[has_payload].copy()

    return df


def normalize_awareness(value: str) -> Optional[str]:
    if pd.isna(value):
        return None
    v = str(value).strip().lower()
    if not v or "prefer" in v or "not say" in v or v in {"nan", "none"}:
        return None
    if v in {"yes", "y", "aware", "true"} or "yes" == v:
        return "Yes"
    if v in {"no", "n", "not aware", "false"} or "no" == v:
        return "No"
    if "yes" in v:
        return "Yes"
    if "no" in v:
        return "No"
    return None


def normalize_student_type(value: str) -> Optional[str]:
    if pd.isna(value):
        return None
    v = str(value).strip().lower()
    if not v or v in {"nan", "none"}:
        return None
    if "under" in v:
        return "Undergrad"
    if "grad" in v or "macc" in v or "master" in v:
        return "Graduate"
    return None


def clean_cpa_intention(value: str) -> Optional[str]:
    if pd.isna(value):
        return None
    v = str(value).strip()
    if not v or v.lower() in {"nan", "none"} or "importid" in v.lower():
        return None
    return v


def map_intent_binary(value: str) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    v = str(value).strip().lower()
    yes_markers = [
        "very likely",
        "somewhat likely",
        "likely",
        "yes",
        "definitely",
        "probably",
        "intend",
    ]
    no_markers = [
        "very unlikely",
        "somewhat unlikely",
        "unlikely",
        "no",
        "do not",
        "dont",
        "neither likely nor unlikely",
        "neutral",
        "unsure",
    ]
    if any(marker in v for marker in yes_markers):
        return 1
    if any(marker in v for marker in no_markers):
        return 0
    return None



def table_to_markdown(df: pd.DataFrame, decimals: int = 1) -> str:
    if df.empty:
        return "_No data available._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "")
    return formatted.to_markdown()


def save_overall_chart(crosstab: pd.DataFrame, out_path: Path) -> None:
    pct = crosstab.div(crosstab.sum(axis=1), axis=0).fillna(0) * 100
    categories = list(pct.columns)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(pct.index))
    bottom = np.zeros(len(pct.index))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(categories)))

    for i, cat in enumerate(categories):
        vals = pct[cat].values
        bars = ax.bar(x, vals, bottom=bottom, label=cat, color=colors[i], edgecolor="white")
        for j, b in enumerate(bars):
            if vals[j] >= 8:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    bottom[j] + vals[j] / 2,
                    f"{vals[j]:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(pct.index)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of respondents (%)")
    ax.set_title("CPA intention profile by awareness of alternative pathway")
    ax.legend(title="CPA intention", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_by_student_type_chart(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = (
        df.groupby(["student_type", "awareness", "intent_binary"]).size().reset_index(name="n")
    )
    if plot_df.empty:
        return
    total_df = plot_df.groupby(["student_type", "awareness"]) ["n"].transform("sum")
    plot_df["pct"] = (plot_df["n"] / total_df) * 100

    order_student = ["Undergrad", "Graduate"]
    order_awareness = ["No", "Yes"]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    x_labels = []
    x_pos = []
    bar_vals = []
    colors = []
    for s in order_student:
        for a in order_awareness:
            subset = plot_df[(plot_df["student_type"] == s) & (plot_df["awareness"] == a) & (plot_df["intent_binary"] == 1)]
            val = subset["pct"].iloc[0] if not subset.empty else 0
            x_labels.append(f"{s}\n{a}")
            x_pos.append(len(x_pos))
            bar_vals.append(val)
            colors.append("#1f77b4" if s == "Undergrad" else "#ff7f0e")

    bars = ax.bar(x_pos, bar_vals, color=colors, alpha=0.9)
    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%", ha="center", fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, max(bar_vals + [10]) + 15)
    ax.set_ylabel("% mapped to CPA intent = Yes")
    ax.set_title("Awareness-intention pattern by student type")
    ax.text(
        0.01,
        -0.22,
        "Bars show the share with binary CPA intent=Yes within each student type × awareness cell.",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_analysis(data_path: Path, outdir: Path) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    df_raw = load_qualtrics_csv(data_path)

    for needed in REQUIRED_IDS:
        if needed not in df_raw.columns:
            raise KeyError(f"Required column {needed} not found in dataset.")

    analysis_df = pd.DataFrame(
        {
            "student_type": df_raw["Q27"].apply(normalize_student_type),
            "cpa_intention": df_raw["Q29"].apply(clean_cpa_intention),
            "awareness": df_raw["Q53"].apply(normalize_awareness),
            "q51": df_raw["Q51"].apply(clean_cpa_intention) if "Q51" in df_raw.columns else None,
            "q31": df_raw["Q31"].apply(normalize_awareness) if "Q31" in df_raw.columns else None,
        }
    )

    n_total = len(analysis_df)
    missing_q27 = analysis_df["student_type"].isna().sum()
    missing_q29 = analysis_df["cpa_intention"].isna().sum()
    missing_q53 = analysis_df["awareness"].isna().sum()

    complete_mask = analysis_df[["student_type", "cpa_intention", "awareness"]].notna().all(axis=1)
    complete_df = analysis_df.loc[complete_mask].copy()

    ctab = pd.crosstab(complete_df["awareness"], complete_df["cpa_intention"]).sort_index()
    row_pct = ctab.div(ctab.sum(axis=1), axis=0) * 100
    col_pct = ctab.div(ctab.sum(axis=0), axis=1) * 100

    chi2 = p_value = dof = np.nan
    if ctab.shape[0] > 1 and ctab.shape[1] > 1:
        chi2, p_value, dof, _ = chi2_contingency(ctab)

    strat_lines: List[str] = []
    for stype in ["Undergrad", "Graduate"]:
        sdf = complete_df[complete_df["student_type"] == stype]
        sctab = pd.crosstab(sdf["awareness"], sdf["cpa_intention"]).sort_index()
        strat_lines.append(f"#### {stype}\n")
        strat_lines.append("Counts:\n")
        strat_lines.append(table_to_markdown(sctab, decimals=0))
        strat_lines.append("\nRow %:\n")
        strat_lines.append(table_to_markdown(sctab.div(sctab.sum(axis=1), axis=0) * 100))
        strat_lines.append("\n")

    complete_df["intent_binary"] = complete_df["cpa_intention"].apply(map_intent_binary)
    model_df = complete_df.dropna(subset=["intent_binary"]).copy()
    model_df["intent_binary"] = model_df["intent_binary"].astype(int)

    model_note = "Model not run: insufficient data."
    interaction_direction = "Not estimated"
    model_table_md = ""
    if len(model_df) >= 30 and model_df["intent_binary"].nunique() > 1:
        try:
            logit = smf.logit("intent_binary ~ C(awareness) * C(student_type)", data=model_df).fit(disp=False)
            coef = logit.params
            conf = logit.conf_int()
            model_table = pd.DataFrame(
                {
                    "coef": coef,
                    "odds_ratio": np.exp(coef),
                    "ci_low_or": np.exp(conf[0]),
                    "ci_high_or": np.exp(conf[1]),
                    "p_value": logit.pvalues,
                }
            )
            model_table_md = table_to_markdown(model_table.reset_index().rename(columns={"index": "term"}), decimals=3)

            interaction_terms = [t for t in coef.index if ":" in t]
            if interaction_terms:
                inter = interaction_terms[0]
                inter_coef = coef[inter]
                interaction_direction = "positive" if inter_coef > 0 else "negative"
                model_note = (
                    f"Interaction term `{inter}` is {interaction_direction} "
                    f"(coef={inter_coef:.3f}, p={logit.pvalues[inter]:.3f})."
                )
            else:
                model_note = "No interaction term estimated."
        except Exception as exc:  # noqa: BLE001
            model_note = f"Model failed to converge: {exc}"

    sensitivity_note = "Sensitivity model not run."
    if "q51" in model_df.columns and model_df["q51"].notna().sum() >= 30:
        model_df2 = model_df.dropna(subset=["q51"]).copy()
        if model_df2["q51"].nunique() > 1:
            try:
                sens = smf.logit(
                    "intent_binary ~ C(awareness) * C(student_type) + C(q51)",
                    data=model_df2,
                ).fit(disp=False)
                sensitivity_note = (
                    "Sensitivity model including Q51 ran successfully; "
                    f"awareness term p-values: {', '.join(f'{k}={v:.3f}' for k, v in sens.pvalues.items() if 'awareness' in k)}"
                )
            except Exception as exc:  # noqa: BLE001
                sensitivity_note = f"Sensitivity model failed: {exc}"

    overall_fig = outdir / "fig_awareness_intent_overall.png"
    by_type_fig = outdir / "fig_awareness_intent_by_student_type.png"
    save_overall_chart(ctab, overall_fig)
    save_by_student_type_chart(model_df, by_type_fig)

    # Difference-in-differences style interpretation using binary mapping.
    did_text = "Could not compute DiD-style contrast."
    did_value = np.nan
    if not model_df.empty:
        grp = (
            model_df.groupby(["student_type", "awareness"]) ["intent_binary"].mean().mul(100).unstack()
        )
        if set(["Yes", "No"]).issubset(grp.columns) and {"Undergrad", "Graduate"}.issubset(grp.index):
            ug_diff = grp.loc["Undergrad", "Yes"] - grp.loc["Undergrad", "No"]
            g_diff = grp.loc["Graduate", "Yes"] - grp.loc["Graduate", "No"]
            did_value = g_diff - ug_diff
            did_text = (
                f"Awareness lift in intent=Yes is {ug_diff:.1f} pp for undergrads and {g_diff:.1f} pp for graduates; "
                f"difference-in-differences style contrast is {did_value:.1f} pp."
            )

    report_path = outdir / "awareness_vs_cpa_intent.md"
    mapping_text = (
        "Binary mapping for Q29: 'Very likely'/'Somewhat likely'/'Likely'/'Yes'-like responses -> 1 (intent=Yes); "
        "'Neither likely nor unlikely', 'Somewhat unlikely', 'Very unlikely', and other no/neutral/unsure responses -> 0."
    )

    report = f"""# Awareness of Alternative CPA Pathway vs CPA Intention

## Data and quality checks
- Input file: `{data_path}`
- Total response rows analyzed (after dropping Qualtrics metadata rows): **{n_total}**
- Missingness:
  - Q27 (student type): **{missing_q27}**
  - Q29 (CPA intention): **{missing_q29}**
  - Q53 (awareness): **{missing_q53}**
- Complete cases for core analysis (Q27, Q29, Q53 all present): **{len(complete_df)}**

## Overall association: Q53 (awareness) × Q29 (CPA intention)

### Counts
{ctab.to_markdown()}

### Row percentages (within awareness)
{(row_pct).round(1).to_markdown()}

### Column percentages (within intention category)
{(col_pct).round(1).to_markdown()}

### Chi-square test
- χ² = **{chi2:.3f}**, df = **{int(dof) if pd.notna(dof) else 'NA'}**, p-value = **{p_value:.4f}**

![Overall awareness vs CPA intention](fig_awareness_intent_overall.png)

## Stratified by student type (Q27)
{''.join(strat_lines)}
### Difference-in-differences style interpretation
- {did_text}

## Model-based check (binary CPA intent)
- {mapping_text}
- Sample size for binary model: **{len(model_df)}**
- {model_note}

### Logistic model summary (intent_binary ~ awareness + student_type + awareness*student_type)
{model_table_md if model_table_md else '_Model output unavailable._'}

## Sensitivity (adding Q51 perceived impact)
- {sensitivity_note}

## Optional timing check for graduates (Q31)
"""

    if "q31" in analysis_df.columns and analysis_df["q31"].notna().any():
        grad_q31 = analysis_df[analysis_df["student_type"] == "Graduate"]["q31"].value_counts(dropna=True)
        report += "\n" + grad_q31.to_markdown() + "\n"
    else:
        report += "\n_No usable Q31 responses found._\n"

    report += "\n\n![By student type awareness-intention pattern](fig_awareness_intent_by_student_type.png)\n"

    report_path.write_text(report, encoding="utf-8")

    summary = {
        "data_file": str(data_path),
        "n_total": int(n_total),
        "missing_q27": int(missing_q27),
        "missing_q29": int(missing_q29),
        "missing_q53": int(missing_q53),
        "n_complete": int(len(complete_df)),
        "chi2_p_value": None if pd.isna(p_value) else float(p_value),
        "did_pp": None if pd.isna(did_value) else float(did_value),
        "interaction_direction": interaction_direction,
        "model_note": model_note,
    }

    summary_path = outdir / "awareness_vs_cpa_intent_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze awareness vs CPA intention from Qualtrics CSV")
    parser.add_argument("--data", type=str, default=None, help="Path to Qualtrics CSV (default: latest in data/)")
    parser.add_argument("--outdir", type=str, default="reports", help="Output directory for report + figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    if args.data:
        data_path = Path(args.data)
    else:
        data_path = find_latest_csv(Path("data"))

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    summary = run_analysis(data_path, outdir)
    print(
        " | ".join(
            [
                f"file={summary['data_file']}",
                f"n={summary['n_total']}",
                f"complete={summary['n_complete']}",
                f"missing(Q27/Q29/Q53)={summary['missing_q27']}/{summary['missing_q29']}/{summary['missing_q53']}",
                f"chi2_p={summary['chi2_p_value']}",
                f"interaction={summary['interaction_direction']}",
            ]
        )
    )


if __name__ == "__main__":
    main()
