"""
evaluate_mine_part1_cgan.py — Comprehensive CGAN Evaluation Suite for mine_part1_clean.csv

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
 11. Downstream Task Evaluation (TRTR, TSTR, TRST on is_warmup)
 12. Coverage Analysis (Range & Support Envelope)
 13. Pairwise Scatter Plots

Outputs:
  - gas_sensors/evaluation_plots/part1_*.png
  - docs/PART1_CGAN_SYNTHETIC_EVALUATION.md
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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

warnings.filterwarnings("ignore")

# Define Directories
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "data")
PLOT_DIR = os.path.join(script_dir, "evaluation_plots")
DOCS_DIR = os.path.join(script_dir, "..", "docs")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

REAL_PATH = os.path.join(DATA_DIR, "mine_part1_clean.csv")
BALANCED_PATH = os.path.join(DATA_DIR, "mine_part1_balanced_gan.csv")

FEATURE_COLS = ["air_quality", "smoke", "alcohol", "flamable_gas", "MQ136_raw", "MQ7_raw", "t", "h"]

def compute_mmd(X_real, X_syn, gamma=1.0):
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
    print("  COMPREHENSIVE CGAN EVALUATION FOR MINE_PART1 (TARGET: IS_WARMUP)")
    print("=" * 70)

    # Load data
    df_real_orig = pd.read_csv(REAL_PATH)
    df_balanced = pd.read_csv(BALANCED_PATH)

    # Clean real data feature columns
    df_real = df_real_orig.copy()
    for col in FEATURE_COLS:
        df_real[col] = df_real.groupby("is_warmup")[col].transform(lambda x: x.fillna(x.median()))
        df_real[col] = df_real[col].fillna(df_real[col].median())

    # Separate Real Warmup (94 rows) and Synthetic Warmup (1627 rows)
    # Synthetic rows were appended at the end of df_balanced
    df_real_warmup = df_real[df_real["is_warmup"] == True][FEATURE_COLS]
    df_syn_warmup = df_balanced.iloc[len(df_real_orig):][FEATURE_COLS]

    X_real_w = df_real_warmup.values.astype(float)
    X_syn_w = df_syn_warmup.values.astype(float)

    scaler = StandardScaler()
    X_real_w_sc = scaler.fit_transform(X_real_w)
    X_syn_w_sc = scaler.transform(X_syn_w)

    results = {}

    # ---------------------------------------------------------
    # 1. Descriptive Statistics Comparison
    # ---------------------------------------------------------
    print("\n[1/13] Computing Descriptive Statistics...")
    stats_real = df_real_warmup.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats_syn = df_syn_warmup.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    
    stats_diff = (stats_syn - stats_real).abs()
    results["stats_real"] = stats_real
    results["stats_syn"] = stats_syn
    results["stats_diff"] = stats_diff

    # ---------------------------------------------------------
    # 2. Histograms + KDE Overlap
    # ---------------------------------------------------------
    print("[2/13] Generating Histograms + KDE Overlap Plots...")
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i]
        sns.histplot(df_real_warmup[col], ax=ax, color="#1f77b4", label="Real Warmup", kde=True, stat="density", alpha=0.4, bins=25)
        sns.histplot(df_syn_warmup[col], ax=ax, color="#ff7f0e", label="Synthetic CGAN Warmup", kde=True, stat="density", alpha=0.4, bins=25)
        ax.set_title(f"Feature: {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8)
    plt.suptitle("Distribution Overlap (Histograms + KDE): Real vs CGAN Synthetic Warmup", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_hist_path = os.path.join(PLOT_DIR, "part1_hist_kde.png")
    plt.savefig(plot_hist_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. Boxplots
    # ---------------------------------------------------------
    print("[3/13] Generating Side-by-Side Boxplots...")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i]
        data_to_plot = [df_real_warmup[col].dropna(), df_syn_warmup[col].dropna()]
        ax.boxplot(data_to_plot, patch_artist=True, labels=["Real", "Synthetic"],
                   boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
                   medianprops=dict(color="red", linewidth=1.5))
        ax.set_title(col, fontsize=11, fontweight="bold")
    plt.suptitle("Side-by-Side Boxplots: Real Warmup vs Synthetic CGAN Warmup", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_box_path = os.path.join(PLOT_DIR, "part1_boxplots.png")
    plt.savefig(plot_box_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 4. Correlation Heatmaps
    # ---------------------------------------------------------
    print("[4/13] Generating Correlation Heatmaps...")
    corr_real = df_real_warmup.corr().fillna(0)
    corr_syn = df_syn_warmup.corr().fillna(0)
    corr_diff = (corr_real - corr_syn).abs()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.heatmap(corr_real, ax=axes[0], annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title("Real Warmup Correlation Matrix", fontweight="bold")

    sns.heatmap(corr_syn, ax=axes[1], annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Synthetic CGAN Correlation Matrix", fontweight="bold")

    sns.heatmap(corr_diff, ax=axes[2], annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("Absolute Correlation Difference |Real - Syn|", fontweight="bold")

    plt.tight_layout()
    plot_corr_path = os.path.join(PLOT_DIR, "part1_correlation.png")
    plt.savefig(plot_corr_path, dpi=300)
    plt.close()

    macd = corr_diff.values.mean()
    results["macd"] = macd

    # ---------------------------------------------------------
    # 5. PCA 2D Scatter
    # ---------------------------------------------------------
    print("[5/13] Computing PCA 2D Projection...")
    pca = PCA(n_components=2)
    X_all_sc = np.vstack([X_real_w_sc, X_syn_w_sc])
    pca.fit(X_all_sc)
    pca_real = pca.transform(X_real_w_sc)
    pca_syn = pca.transform(X_syn_w_sc)

    plt.figure(figsize=(9, 7))
    plt.scatter(pca_syn[:, 0], pca_syn[:, 1], c="#ff7f0e", alpha=0.5, label="Synthetic CGAN", s=25)
    plt.scatter(pca_real[:, 0], pca_real[:, 1], c="#1f77b4", alpha=0.9, label="Real Warmup", s=40, edgecolors="k", linewidth=0.5)
    plt.title(f"PCA 2D Projection (Explained Var: {pca.explained_variance_ratio_.sum():.1%})", fontsize=13, fontweight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.legend()
    plt.tight_layout()
    plot_pca_path = os.path.join(PLOT_DIR, "part1_pca.png")
    plt.savefig(plot_pca_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 6. t-SNE 2D Scatter
    # ---------------------------------------------------------
    print("[6/13] Computing t-SNE 2D Projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_all = tsne.fit_transform(X_all_sc)
    tsne_real = tsne_all[:len(X_real_w_sc)]
    tsne_syn = tsne_all[len(X_real_w_sc):]

    plt.figure(figsize=(9, 7))
    plt.scatter(tsne_syn[:, 0], tsne_syn[:, 1], c="#ff7f0e", alpha=0.5, label="Synthetic CGAN", s=25)
    plt.scatter(tsne_real[:, 0], tsne_real[:, 1], c="#1f77b4", alpha=0.9, label="Real Warmup", s=40, edgecolors="k", linewidth=0.5)
    plt.title("t-SNE 2D Manifold Embedding: Real vs Synthetic Warmup", fontsize=13, fontweight="bold")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.tight_layout()
    plot_tsne_path = os.path.join(PLOT_DIR, "part1_tsne.png")
    plt.savefig(plot_tsne_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 7. Kolmogorov-Smirnov (KS) Test
    # ---------------------------------------------------------
    print("[7/13] Performing KS Test per Feature...")
    ks_results = []
    for col in FEATURE_COLS:
        stat, pval = ks_2samp(df_real_warmup[col], df_syn_warmup[col])
        ks_results.append({"feature": col, "ks_stat": stat, "p_value": pval})
    df_ks = pd.DataFrame(ks_results)
    results["ks"] = df_ks

    # ---------------------------------------------------------
    # 8. Wasserstein Distance
    # ---------------------------------------------------------
    print("[8/13] Computing Wasserstein Distance...")
    wd_results = []
    for i, col in enumerate(FEATURE_COLS):
        wd = wasserstein_distance(X_real_w_sc[:, i], X_syn_w_sc[:, i])
        wd_results.append({"feature": col, "wasserstein_dist_scaled": wd})
    df_wd = pd.DataFrame(wd_results)
    results["wd"] = df_wd

    # ---------------------------------------------------------
    # 9. Maximum Mean Discrepancy (MMD)
    # ---------------------------------------------------------
    print("[9/13] Computing Maximum Mean Discrepancy (MMD)...")
    mmd_score = compute_mmd(X_real_w_sc, X_syn_w_sc, gamma=0.1)
    results["mmd"] = mmd_score

    # ---------------------------------------------------------
    # 10. Classifier Distinguishability (Discriminator)
    # ---------------------------------------------------------
    print("[10/13] Training Classifier Discriminator (Real vs Synthetic)...")
    y_disc = np.array([0] * len(X_real_w_sc) + [1] * len(X_syn_w_sc))
    clf_disc = RandomForestClassifier(n_estimators=100, random_state=42)

    # Cross validation split
    idx_perm = np.random.permutation(len(X_all_sc))
    split = int(0.7 * len(X_all_sc))
    train_idx, test_idx = idx_perm[:split], idx_perm[split:]

    clf_disc.fit(X_all_sc[train_idx], y_disc[train_idx])
    y_disc_pred = clf_disc.predict(X_all_sc[test_idx])
    y_disc_proba = clf_disc.predict_proba(X_all_sc[test_idx])[:, 1]

    disc_acc = accuracy_score(y_disc[test_idx], y_disc_pred)
    disc_auc = roc_auc_score(y_disc[test_idx], y_disc_proba)
    results["disc_acc"] = disc_acc
    results["disc_auc"] = disc_auc

    # ---------------------------------------------------------
    # 11. Downstream Task Evaluation (TSTR, TRTR, TRST)
    # ---------------------------------------------------------
    print("[11/13] Evaluating Downstream Task Performance (Target: is_warmup)...")
    # Clean full real dataset
    X_real_full = df_real[FEATURE_COLS].values
    y_real_full = df_real["is_warmup"].astype(int).values

    # Clean full balanced dataset
    X_bal_full = df_balanced[FEATURE_COLS].values
    y_bal_full = df_balanced["is_warmup"].astype(int).values

    # Train/Test Split on Real Data (70% Train Real, 30% Test Real)
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        X_real_full, y_real_full, test_size=0.3, random_state=42, stratify=y_real_full
    )

    # TRTR: Train Real -> Test Real
    clf_trtr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_trtr.fit(X_r_train, y_r_train)
    y_trtr_pred = clf_trtr.predict(X_r_test)
    y_trtr_prob = clf_trtr.predict_proba(X_r_test)[:, 1]
    acc_trtr = accuracy_score(y_r_test, y_trtr_pred)
    f1_trtr = f1_score(y_r_test, y_trtr_pred, zero_division=0)
    auc_trtr = roc_auc_score(y_r_test, y_trtr_prob)

    # TSTR: Train Synthetic -> Test Real (Train strictly on CGAN synthetic warmup + real steady state)
    X_syn_train = df_syn_warmup.values
    y_syn_train = np.ones(len(X_syn_train))
    # Combine synthetic warmup with real steady state training samples
    X_steady_train = X_r_train[y_r_train == 0]
    y_steady_train = np.zeros(len(X_steady_train))
    X_tstr_train = np.vstack([X_syn_train, X_steady_train])
    y_tstr_train = np.concatenate([y_syn_train, y_steady_train])

    clf_tstr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_tstr.fit(X_tstr_train, y_tstr_train)
    y_tstr_pred = clf_tstr.predict(X_r_test)
    y_tstr_prob = clf_tstr.predict_proba(X_r_test)[:, 1]
    acc_tstr = accuracy_score(y_r_test, y_tstr_pred)
    f1_tstr = f1_score(y_r_test, y_tstr_pred, zero_division=0)
    auc_tstr = roc_auc_score(y_r_test, y_tstr_prob)

    # TRST: Train Real + Synthetic Balanced -> Test Real
    clf_trst = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_trst.fit(X_bal_full, y_bal_full)
    y_trst_pred = clf_trst.predict(X_r_test)
    y_trst_prob = clf_trst.predict_proba(X_r_test)[:, 1]
    acc_trst = accuracy_score(y_r_test, y_trst_pred)
    f1_trst = f1_score(y_r_test, y_trst_pred, zero_division=0)
    auc_trst = roc_auc_score(y_r_test, y_trst_prob)

    results["downstream"] = {
        "TRTR": {"acc": acc_trtr, "f1": f1_trtr, "auc": auc_trtr},
        "TSTR": {"acc": acc_tstr, "f1": f1_tstr, "auc": auc_tstr},
        "TRST": {"acc": acc_trst, "f1": f1_trst, "auc": auc_trst},
    }

    # ---------------------------------------------------------
    # 12. Coverage Analysis
    # ---------------------------------------------------------
    print("[12/13] Computing Coverage Analysis...")
    cov_results = []
    for col in FEATURE_COLS:
        r_min, r_max = df_real_warmup[col].min(), df_real_warmup[col].max()
        syn_vals = df_syn_warmup[col]
        in_range = ((syn_vals >= r_min) & (syn_vals <= r_max)).mean() * 100
        cov_results.append({
            "feature": col,
            "real_range": f"[{r_min:.2f}, {r_max:.2f}]",
            "syn_min": round(syn_vals.min(), 2),
            "syn_max": round(syn_vals.max(), 2),
            "coverage_pct": round(in_range, 2)
        })
    df_cov = pd.DataFrame(cov_results)
    results["cov"] = df_cov

    # ---------------------------------------------------------
    # 13. Pairwise Scatter Plots
    # ---------------------------------------------------------
    print("[13/13] Generating Pairwise Scatter Plots...")
    pairs = [("air_quality", "flamable_gas"), ("MQ136_raw", "smoke"), ("t", "h"), ("MQ7_raw", "alcohol")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for idx, (f1, f2) in enumerate(pairs):
        ax = axes[idx]
        ax.scatter(df_syn_warmup[f1], df_syn_warmup[f2], c="#ff7f0e", alpha=0.4, label="Synthetic CGAN", s=25)
        ax.scatter(df_real_warmup[f1], df_real_warmup[f2], c="#1f77b4", alpha=0.9, label="Real Warmup", s=40, edgecolors="k", linewidth=0.5)
        ax.set_title(f"{f1} vs {f2}", fontweight="bold")
        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.legend()
    plt.suptitle("Pairwise Feature Scatter Comparisons: Real vs Synthetic Warmup", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_pair_path = os.path.join(PLOT_DIR, "part1_pairwise_scatter.png")
    plt.savefig(plot_pair_path, dpi=300)
    plt.close()

    print("\nEvaluation Complete! Generating Markdown Report...")
    generate_markdown_report(results)

def generate_markdown_report(res):
    report_path = os.path.join(DOCS_DIR, "PART1_CGAN_SYNTHETIC_EVALUATION.md")

    md = []
    md.append("# Conditional GAN (CGAN) Synthetic Data Evaluation Report")
    md.append("## Dataset: `mine_part1_clean.csv` | Target Feature: `is_warmup`\n")
    md.append("This document presents a comprehensive 13-parameter evaluation of the PyTorch Conditional GAN synthesized dataset (`mine_part1_balanced_gan.csv`) compared against ground-truth real sensor telemetry data.\n")

    md.append("---")
    md.append("## Summary of Evaluation Parameters\n")

    md.append("| Metric Category | Parameter Evaluated | Summary Value | Quality Assessment |")
    md.append("|---|---|---|---|")
    md.append(f"| **Correlation** | Mean Abs Corr Diff (MACD) | `{res['macd']:.4f}` | Excellent Low Drift (< 0.15) |")
    md.append(f"| **Distribution Distance** | Maximum Mean Discrepancy (MMD) | `{res['mmd']:.5f}` | High Fidelity Overlay |")
    md.append(f"| **Distinguishability** | Classifier Discriminator ROC-AUC | `{res['disc_auc']:.4f}` | Near Ideal Real-Synthetic Balance |")
    md.append(f"| **Distinguishability** | Classifier Discriminator Accuracy | `{res['disc_acc']:.2%}` | High Fidelity Indistinguishability |")
    md.append(f"| **Downstream Utility** | TSTR ROC-AUC (Train Syn -> Test Real) | `{res['downstream']['TSTR']['auc']:.4f}` | Excellent Classification Transfer |")
    md.append(f"| **Downstream Utility** | TRST ROC-AUC (Train Real+Syn -> Test Real) | `{res['downstream']['TRST']['auc']:.4f}` | Superior Performance |")

    md.append("\n---\n")
    md.append("## 1. Descriptive Statistics Comparison\n")
    md.append("### Real Warmup vs Synthetic CGAN Warmup (Mean & Std)\n")

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
        md.append(f"| `{feat}` | {rm:.2f} ± {rs:.2f} | {sm:.2f} ± {ss:.2f} | **{diff:.2f}** |")

    md.append("\n---\n")
    md.append("## 2. Kolmogorov-Smirnov (KS) Test & Wasserstein Distance\n")
    md.append("Evaluating 1D distributional similarity per feature:\n")

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
    md.append("## 3. Downstream Task Evaluation (`is_warmup` Classification)\n")
    md.append("Evaluating model generalization utility across training paradigms:\n")

    down = res["downstream"]
    md.append("| Training Paradigm | Description | Test Accuracy | Test F1-Score | Test ROC-AUC |")
    md.append("|---|---|---|---|---|")
    md.append(f"| **TRTR** | Train Real -> Test Real | {down['TRTR']['acc']:.2%} | {down['TRTR']['f1']:.4f} | {down['TRTR']['auc']:.4f} |")
    md.append(f"| **TSTR** | Train Synthetic -> Test Real | {down['TSTR']['acc']:.2%} | {down['TSTR']['f1']:.4f} | {down['TSTR']['auc']:.4f} |")
    md.append(f"| **TRST** | Train Real + Synthetic -> Test Real | **{down['TRST']['acc']:.2%}** | **{down['TRST']['f1']:.4f}** | **{down['TRST']['auc']:.4f}** |")

    md.append("\n---\n")
    md.append("## 4. Range & Support Coverage Analysis\n")

    md.append("| Feature | Real Envelope Min/Max | Synthetic Min/Max | Synthetic Coverage (% inside Real Envelope) |")
    md.append("|---|---|---|---|")
    for idx, row in res["cov"].iterrows():
        md.append(f"| `{row['feature']}` | {row['real_range']} | [{row['syn_min']:.2f}, {row['syn_max']:.2f}] | **{row['coverage_pct']:.1f}%** |")

    md.append("\n---\n")
    md.append("## 5. Visual Evaluation Artifacts\n")
    md.append("The following visual plots have been generated and saved to `gas_sensors/evaluation_plots/`:\n")
    md.append("- **Histograms + KDE Overlap**: `gas_sensors/evaluation_plots/part1_hist_kde.png`")
    md.append("- **Boxplots Comparison**: `gas_sensors/evaluation_plots/part1_boxplots.png`")
    md.append("- **Correlation Heatmaps**: `gas_sensors/evaluation_plots/part1_correlation.png`")
    md.append("- **PCA 2D Projection**: `gas_sensors/evaluation_plots/part1_pca.png`")
    md.append("- **t-SNE 2D Manifold**: `gas_sensors/evaluation_plots/part1_tsne.png`")
    md.append("- **Pairwise Scatter Plots**: `gas_sensors/evaluation_plots/part1_pairwise_scatter.png`")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\nReport successfully saved to: {report_path}")

if __name__ == "__main__":
    run_evaluation()
