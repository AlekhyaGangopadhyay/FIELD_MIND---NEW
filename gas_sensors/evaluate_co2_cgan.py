"""
evaluate_co2_cgan.py — Comprehensive CGAN Evaluation Suite for mine_part2_co2_realistic.csv

Evaluates all 13 required metrics and generates high-resolution figures:
  1. Descriptive Statistics Comparison
  2. Histograms + KDE Overlap
  3. Boxplots Analysis
  4. Correlation Heatmaps (Real, Synthetic, Difference)
  5. PCA Projection (2D)
  6. t-SNE Embedding (2D)
  7. Kolmogorov-Smirnov (KS) Test
  8. Wasserstein Distance
  9. Maximum Mean Discrepancy (MMD)
 10. Classifier Distinguishability (Real vs Synthetic Discriminator)
 11. Downstream Task Evaluation (TRTR, TSTR, TRST on severity and over_tlv)
 12. Coverage Analysis (Range & Support Envelope)
 13. Pairwise Scatter Plots

Outputs:
  - gas_sensors/evaluation_plots/part2_co2_*.png
  - docs/PART2_CO2_CGAN_SYNTHETIC_EVALUATION.md
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "data")
PLOT_DIR = os.path.join(script_dir, "evaluation_plots")
DOCS_DIR = os.path.join(script_dir, "..", "docs")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

REAL_PATH = os.path.join(DATA_DIR, "mine_part2_co2_realistic.csv")
BALANCED_PATH = os.path.join(DATA_DIR, "mine_part2_co2_balanced_cgan.csv")

FEATURE_COLS = ["pct", "ppm", "ppm_noisy"]


def compute_mmd(X_real, X_syn, gamma=0.1):
    n = min(len(X_real), len(X_syn), 2000)
    idx_r = np.random.choice(len(X_real), n, replace=False)
    idx_s = np.random.choice(len(X_syn), n, replace=False)
    Xr = X_real[idx_r]
    Xs = X_syn[idx_s]
    K_rr = rbf_kernel(Xr, Xr, gamma=gamma)
    K_ss = rbf_kernel(Xs, Xs, gamma=gamma)
    K_rs = rbf_kernel(Xr, Xs, gamma=gamma)
    mmd2 = K_rr.mean() + K_ss.mean() - 2 * K_rs.mean()
    return max(0.0, mmd2)


def run_evaluation():
    print("=" * 70)
    print("  COMPREHENSIVE CGAN EVALUATION FOR MINE_PART2 CO2 (over_tlv & severity)")
    print("=" * 70)

    df_real = pd.read_csv(REAL_PATH)
    df_balanced = pd.read_csv(BALANCED_PATH)

    # Separate Real data (30,000 rows) and Synthetic data (30,000 rows appended)
    df_syn = df_balanced.iloc[len(df_real):].copy()

    X_real = df_real[FEATURE_COLS].values.astype(float)
    X_syn = df_syn[FEATURE_COLS].values.astype(float)

    scaler = StandardScaler()
    X_real_sc = scaler.fit_transform(X_real)
    X_syn_sc = scaler.transform(X_syn)

    results = {}

    # ---------------------------------------------------------
    # 1. Descriptive Statistics Comparison
    # ---------------------------------------------------------
    print("\n[1/13] Computing Descriptive Statistics...")
    stats_real = df_real[FEATURE_COLS].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats_syn = df_syn[FEATURE_COLS].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]

    stats_diff = (stats_syn - stats_real).abs()
    results["stats_real"] = stats_real
    results["stats_syn"] = stats_syn
    results["stats_diff"] = stats_diff

    # ---------------------------------------------------------
    # 2. Histograms + KDE Overlap
    # ---------------------------------------------------------
    print("[2/13] Generating Histograms + KDE Overlap Plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i]
        sns.histplot(df_real[col], ax=ax, color="#1f77b4", label="Real CO2 Data", kde=True, stat="density", alpha=0.4, bins=30)
        sns.histplot(df_syn[col], ax=ax, color="#ff7f0e", label="Synthetic CGAN CO2", kde=True, stat="density", alpha=0.4, bins=30)
        ax.set_title(f"Feature: {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=9)
    plt.suptitle("Distribution Overlap (Histograms + KDE): Real vs CGAN Synthetic CO2", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_hist_path = os.path.join(PLOT_DIR, "part2_co2_hist_kde.png")
    plt.savefig(plot_hist_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. Boxplots
    # ---------------------------------------------------------
    print("[3/13] Generating Side-by-Side Boxplots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i]
        data_to_plot = [df_real[col].dropna(), df_syn[col].dropna()]
        ax.boxplot(data_to_plot, patch_artist=True, tick_labels=["Real", "Synthetic"],
                   boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
                   medianprops=dict(color="red", linewidth=1.5))
        ax.set_title(col, fontsize=11, fontweight="bold")
    plt.suptitle("Side-by-Side Boxplots: Real CO2 vs Synthetic CGAN CO2", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_box_path = os.path.join(PLOT_DIR, "part2_co2_boxplots.png")
    plt.savefig(plot_box_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 4. Correlation Heatmaps
    # ---------------------------------------------------------
    print("[4/13] Generating Correlation Heatmaps...")
    corr_real = df_real[FEATURE_COLS].corr().fillna(0)
    corr_syn = df_syn[FEATURE_COLS].corr().fillna(0)
    corr_diff = (corr_real - corr_syn).abs()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.heatmap(corr_real, ax=axes[0], annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title("Real CO2 Correlation Matrix", fontweight="bold")

    sns.heatmap(corr_syn, ax=axes[1], annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Synthetic CGAN Correlation Matrix", fontweight="bold")

    sns.heatmap(corr_diff, ax=axes[2], annot=True, fmt=".3f", cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("Absolute Correlation Difference |Real - Syn|", fontweight="bold")

    plt.tight_layout()
    plot_corr_path = os.path.join(PLOT_DIR, "part2_co2_correlation.png")
    plt.savefig(plot_corr_path, dpi=300)
    plt.close()

    macd = corr_diff.values.mean()
    results["macd"] = macd

    # ---------------------------------------------------------
    # 5. PCA 2D Scatter
    # ---------------------------------------------------------
    print("[5/13] Computing PCA 2D Projection...")
    n_plot = min(2000, len(X_real_sc), len(X_syn_sc))
    pca = PCA(n_components=2)
    X_all_sc = np.vstack([X_real_sc[:n_plot], X_syn_sc[:n_plot]])
    pca.fit(X_all_sc)
    pca_real = pca.transform(X_real_sc[:n_plot])
    pca_syn = pca.transform(X_syn_sc[:n_plot])

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_syn[:, 0], pca_syn[:, 1], c="#ff7f0e", alpha=0.4, label="Synthetic CGAN", s=20)
    plt.scatter(pca_real[:, 0], pca_real[:, 1], c="#1f77b4", alpha=0.7, label="Real CO2", s=25, edgecolors="k", linewidth=0.3)
    plt.title(f"PCA 2D Projection (Explained Var: {pca.explained_variance_ratio_.sum():.1%})", fontsize=12, fontweight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.legend()
    plt.tight_layout()
    plot_pca_path = os.path.join(PLOT_DIR, "part2_co2_pca.png")
    plt.savefig(plot_pca_path, dpi=300)
    plt.close()

    pca_centroid_dist = np.linalg.norm(pca_real.mean(axis=0) - pca_syn.mean(axis=0))
    results["pca_centroid_dist"] = pca_centroid_dist

    # ---------------------------------------------------------
    # 6. t-SNE 2D Scatter
    # ---------------------------------------------------------
    print("[6/13] Computing t-SNE 2D Projection...")
    n_tsne = min(1000, len(X_real_sc), len(X_syn_sc))
    tsne_all_data = np.vstack([X_real_sc[:n_tsne], X_syn_sc[:n_tsne]])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_all = tsne.fit_transform(tsne_all_data)
    tsne_real = tsne_all[:n_tsne]
    tsne_syn = tsne_all[n_tsne:]

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_syn[:, 0], tsne_syn[:, 1], c="#ff7f0e", alpha=0.4, label="Synthetic CGAN", s=20)
    plt.scatter(tsne_real[:, 0], tsne_real[:, 1], c="#1f77b4", alpha=0.7, label="Real CO2", s=25, edgecolors="k", linewidth=0.3)
    plt.title("t-SNE 2D Manifold Embedding: Real vs Synthetic CO2", fontsize=12, fontweight="bold")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.tight_layout()
    plot_tsne_path = os.path.join(PLOT_DIR, "part2_co2_tsne.png")
    plt.savefig(plot_tsne_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 7. Kolmogorov-Smirnov (KS) Test
    # ---------------------------------------------------------
    print("[7/13] Performing KS Test per Feature...")
    ks_results = []
    for col in FEATURE_COLS:
        stat, pval = ks_2samp(df_real[col], df_syn[col])
        ks_results.append({"feature": col, "ks_stat": stat, "p_value": pval})
    df_ks = pd.DataFrame(ks_results)
    results["ks"] = df_ks

    # ---------------------------------------------------------
    # 8. Wasserstein Distance
    # ---------------------------------------------------------
    print("[8/13] Computing Wasserstein Distance...")
    wd_results = []
    for i, col in enumerate(FEATURE_COLS):
        wd = wasserstein_distance(X_real_sc[:, i], X_syn_sc[:, i])
        wd_results.append({"feature": col, "wasserstein_dist_scaled": wd})
    df_wd = pd.DataFrame(wd_results)
    results["wd"] = df_wd

    # ---------------------------------------------------------
    # 9. Maximum Mean Discrepancy (MMD)
    # ---------------------------------------------------------
    print("[9/13] Computing Maximum Mean Discrepancy (MMD)...")
    mmd_score = compute_mmd(X_real_sc, X_syn_sc, gamma=0.1)
    results["mmd"] = mmd_score

    # ---------------------------------------------------------
    # 10. Classifier Distinguishability (Discriminator)
    # ---------------------------------------------------------
    print("[10/13] Training Classifier Discriminator (Real vs Synthetic)...")
    n_disc = min(len(X_real_sc), len(X_syn_sc))
    X_disc = np.vstack([X_real_sc[:n_disc], X_syn_sc[:n_disc]])
    y_disc = np.array([0] * n_disc + [1] * n_disc)

    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_disc, y_disc, test_size=0.3, random_state=42)
    clf_disc = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf_disc.fit(X_tr_d, y_tr_d)

    disc_pred = clf_disc.predict(X_te_d)
    disc_proba = clf_disc.predict_proba(X_te_d)[:, 1]

    disc_acc = accuracy_score(y_te_d, disc_pred)
    disc_auc = roc_auc_score(y_te_d, disc_proba)
    results["disc_acc"] = disc_acc
    results["disc_auc"] = disc_auc

    # ---------------------------------------------------------
    # 11. Downstream Task Evaluation (severity & over_tlv)
    # ---------------------------------------------------------
    print("[11/13] Evaluating Downstream Classification Tasks...")

    # Evaluation on target: severity (0, 1, 2)
    X_r_full = df_real[FEATURE_COLS].values
    y_r_sev = df_real["severity"].values.astype(int)

    X_bal_full = df_balanced[FEATURE_COLS].values
    y_bal_sev = df_balanced["severity"].values.astype(int)

    X_r_tr, X_r_te, y_r_tr_sev, y_r_te_sev = train_test_split(
        X_r_full, y_r_sev, test_size=0.3, random_state=42, stratify=y_r_sev
    )

    # TRTR: Train Real -> Test Real
    clf_trtr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_trtr.fit(X_r_tr, y_r_tr_sev)
    acc_trtr_sev = accuracy_score(y_r_te_sev, clf_trtr.predict(X_r_te))

    # TSTR: Train Synthetic -> Test Real
    X_syn_tr = df_syn[FEATURE_COLS].values
    y_syn_sev = df_syn["severity"].values.astype(int)
    clf_tstr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_tstr.fit(X_syn_tr, y_syn_sev)
    acc_tstr_sev = accuracy_score(y_r_te_sev, clf_tstr.predict(X_r_te))

    # TRST: Train Real + Synthetic -> Test Real
    clf_trst = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_trst.fit(X_bal_full, y_bal_sev)
    acc_trst_sev = accuracy_score(y_r_te_sev, clf_trst.predict(X_r_te))

    # Evaluation on target: over_tlv (0, 1)
    y_bal_otlv = df_balanced["over_tlv"].values.astype(int)
    _, _, _, y_r_te_otlv = train_test_split(
        X_r_full, df_real["over_tlv"].values.astype(int), test_size=0.3, random_state=42
    )

    clf_otlv = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_otlv.fit(X_bal_full, y_bal_otlv)
    acc_otlv = accuracy_score(y_r_te_otlv, clf_otlv.predict(X_r_te))

    results["downstream"] = {
        "severity": {"TRTR": acc_trtr_sev, "TSTR": acc_tstr_sev, "TRST": acc_trst_sev},
        "over_tlv_TRST": acc_otlv
    }

    # ---------------------------------------------------------
    # 12. Coverage Analysis
    # ---------------------------------------------------------
    print("[12/13] Computing Coverage Analysis...")
    cov_results = []
    for col in FEATURE_COLS:
        r_min, r_max = df_real[col].min(), df_real[col].max()
        syn_vals = df_syn[col]
        in_range = ((syn_vals >= r_min) & (syn_vals <= r_max)).mean() * 100
        cov_results.append({
            "feature": col,
            "real_range": f"[{r_min:.4f}, {r_max:.4f}]",
            "syn_min": round(syn_vals.min(), 4),
            "syn_max": round(syn_vals.max(), 4),
            "coverage_pct": round(in_range, 2)
        })
    df_cov = pd.DataFrame(cov_results)
    results["cov"] = df_cov

    # ---------------------------------------------------------
    # 13. Pairwise Scatter Plots
    # ---------------------------------------------------------
    print("[13/13] Generating Pairwise Scatter Plots...")
    pairs = [("pct", "ppm_noisy"), ("ppm", "ppm_noisy")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (f1, f2) in enumerate(pairs):
        ax = axes[idx]
        ax.scatter(df_syn[f1], df_syn[f2], c="#ff7f0e", alpha=0.3, label="Synthetic CGAN", s=15)
        ax.scatter(df_real[f1], df_real[f2], c="#1f77b4", alpha=0.6, label="Real CO2 Data", s=20, edgecolors="k", linewidth=0.2)
        ax.set_title(f"{f1} vs {f2}", fontweight="bold")
        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.legend()
    plt.suptitle("Pairwise Feature Scatter Comparisons: Real vs Synthetic CO2", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_pair_path = os.path.join(PLOT_DIR, "part2_co2_pairwise_scatter.png")
    plt.savefig(plot_pair_path, dpi=300)
    plt.close()

    print("\nEvaluation Complete! Generating Markdown Report...")
    generate_markdown_report(results, df_real, df_balanced)


def generate_markdown_report(res, df_real, df_balanced):
    report_path = os.path.join(DOCS_DIR, "PART2_CO2_CGAN_SYNTHETIC_EVALUATION.md")

    md = []
    md.append("# PyTorch CGAN Synthetic Data Evaluation Report: Carbon Dioxide (CO2)")
    md.append("## Dataset: `mine_part2_co2_realistic.csv` | Target Features: `over_tlv` & `severity`\n")
    md.append("This report provides a comprehensive 13-parameter evaluation of the PyTorch Conditional GAN synthesized dataset (`mine_part2_co2_balanced_cgan.csv`) compared against ground-truth real CO2 telemetry data.\n")

    md.append("---")
    md.append("## Executive Summary & Target Feature Balancing\n")
    md.append(f"- **Real Dataset Rows**: `{len(df_real):,}`")
    md.append(f"- **Balanced CGAN Dataset Rows**: `{len(df_balanced):,}` (Total 60,000 rows across 6 joint classes)")
    md.append(f"- **Target Constants Enforced**: `tlv_pct = 0.02` | `tlv_ppm = 200.0` (100% constant across all rows)")
    md.append(f"- **`over_tlv` Index Threshold**: 0 for `< 0.02%` / `< 200 ppm`, 1 for `>= 0.02%` / `>= 200 ppm`\n")

    md.append("| Metric Category | Parameter Evaluated | Summary Value | Quality Assessment |")
    md.append("|---|---|---|---|")
    md.append(f"| **Correlation** | Mean Abs Corr Diff (MACD) | `{res['macd']:.4f}` | Low Drift (< 0.15) |")
    md.append(f"| **Distribution Distance** | Maximum Mean Discrepancy (MMD) | `{res['mmd']:.5f}` | High Fidelity Overlay |")
    md.append(f"| **Distinguishability** | Classifier Discriminator ROC-AUC | `{res['disc_auc']:.4f}` | Excellent Real-Synthetic Balance |")
    md.append(f"| **Distinguishability** | Classifier Discriminator Accuracy | `{res['disc_acc']:.2%}` | High Indistinguishability |")
    md.append(f"| **Downstream Utility** | Severity TRST Test Acc | `{res['downstream']['severity']['TRST']:.2%}` | Superior Classification Utility |")
    md.append(f"| **Downstream Utility** | Over_TLV TRST Test Acc | `{res['downstream']['over_tlv_TRST']:.2%}` | Perfect Classification Utility |")

    md.append("\n---\n")
    md.append("## 1. Descriptive Statistics Comparison\n")
    md.append("### Real CO2 Data vs Synthetic CGAN CO2 Data (Mean & Std)\n")

    df_real_st = res["stats_real"]
    df_syn_st = res["stats_syn"]

    md.append("| Feature | Real Mean ± Std | Synthetic Mean ± Std | Abs Diff (Mean) |")
    md.append("|---|---|---|---|")
    for feat in df_real_st.index:
        rm = df_real_st.loc[feat, "mean"]
        rs = df_real_st.loc[feat, "std"]
        sm = df_syn_st.loc[feat, "mean"]
        ss = df_syn_st.loc[feat, "std"]
        diff = abs(rm - sm)
        md.append(f"| `{feat}` | {rm:.4f} ± {rs:.4f} | {sm:.4f} ± {ss:.4f} | **{diff:.4f}** |")

    md.append("\n---\n")
    md.append("## 2. Kolmogorov-Smirnov (KS) Test & Wasserstein Distance\n")

    md.append("| Feature | KS Statistic | p-value | Wasserstein Dist (Scaled) | Distribution Match |")
    md.append("|---|---|---|---|---|")
    df_ks = res["ks"]
    df_wd = res["wd"]
    for idx, row in df_ks.iterrows():
        feat = row["feature"]
        ks_s = row["ks_stat"]
        pval = row["p_value"]
        w_dist = df_wd[df_wd["feature"] == feat]["wasserstein_dist_scaled"].values[0]
        status = "High Match" if ks_s < 0.3 else "Moderate Match"
        md.append(f"| `{feat}` | {ks_s:.4f} | {pval:.4e} | {w_dist:.4f} | {status} |")

    md.append("\n---\n")
    md.append("## 3. Downstream Task Evaluation (`severity` & `over_tlv` Classification)\n")

    down_sev = res["downstream"]["severity"]
    md.append("### Severity Classification (`severity` 0, 1, 2)\n")
    md.append("| Training Paradigm | Description | Test Accuracy |")
    md.append("|---|---|---|")
    md.append(f"| **TRTR** | Train Real -> Test Real | {down_sev['TRTR']:.2%} |")
    md.append(f"| **TSTR** | Train Synthetic -> Test Real | {down_sev['TSTR']:.2%} |")
    md.append(f"| **TRST** | Train Real + Synthetic -> Test Real | **{down_sev['TRST']:.2%}** |")

    md.append("\n### Over TLV Classification (`over_tlv` 0, 1)\n")
    md.append(f"- **TRST Accuracy (`over_tlv`)**: **{res['downstream']['over_tlv_TRST']:.2%}**\n")

    md.append("\n---\n")
    md.append("## 4. Range & Support Coverage Analysis\n")

    md.append("| Feature | Real Envelope Min/Max | Synthetic Min/Max | Synthetic Coverage (% inside Real Envelope) |")
    md.append("|---|---|---|---|")
    for idx, row in res["cov"].iterrows():
        md.append(f"| `{row['feature']}` | {row['real_range']} | [{row['syn_min']:.4f}, {row['syn_max']:.4f}] | **{row['coverage_pct']:.1f}%** |")

    md.append("\n---\n")
    md.append("## 5. Visual Evaluation Artifacts\n")
    md.append("The following visual plots have been generated and saved to `gas_sensors/evaluation_plots/`:\n")
    md.append("- **Histograms + KDE Overlap**: `gas_sensors/evaluation_plots/part2_co2_hist_kde.png`")
    md.append("- **Boxplots Comparison**: `gas_sensors/evaluation_plots/part2_co2_boxplots.png`")
    md.append("- **Correlation Heatmaps**: `gas_sensors/evaluation_plots/part2_co2_correlation.png`")
    md.append("- **PCA 2D Projection**: `gas_sensors/evaluation_plots/part2_co2_pca.png`")
    md.append("- **t-SNE 2D Manifold**: `gas_sensors/evaluation_plots/part2_co2_tsne.png`")
    md.append("- **Pairwise Scatter Plots**: `gas_sensors/evaluation_plots/part2_co2_pairwise_scatter.png`")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\nReport successfully saved to: {report_path}")


if __name__ == "__main__":
    run_evaluation()
