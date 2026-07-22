"""
evaluate_cgan_synthetics.py — Comprehensive PyTorch CGAN Synthetic Data Evaluation Suite

Evaluates all 13 metrics for CGAN-balanced synthetic datasets vs real data:
  1. Histograms / KDE Overlap
  2. Boxplots Analysis
  3. Statistical Summary (Mean, Std, Min, Max, Median)
  4. Correlation Heatmaps
  5. PCA Projection
  6. t-SNE Embedding
  7. Kolmogorov-Smirnov (KS) Test
  8. Wasserstein Distance
  9. Maximum Mean Discrepancy (MMD)
  10. TSTR (Train Synthetic -> Test Real)
  11. TRTS (Train Real -> Test Synthetic)
  12. Classifier Distinguishability
  13. Time-Series Autocorrelation

Outputs:
  - gas_sensors/evaluation_plots/   (PNG plots)
  - docs/CGAN_SYNTHETIC_EVALUATION.md  (Markdown report)
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
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = script_dir
PROJECT_ROOT = os.path.join(script_dir, "..", "..")
PLOT_DIR = os.path.join(DATA_DIR, "..", "evaluation_plots")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

FEATURE_COLS = ["ppm", "ppm_noisy", "pct", "tlv_pct", "tlv_ppm"]

DATASETS = [
    {
        "gas": "CH4",
        "real":      "mine_part2_ch4_realistic.csv",
        "synthetic": "mine_part2_ch4_balanced_cgan.csv",
        "target":    "severity",
    },
    {
        "gas": "CO",
        "real":      "mine_part2_co_realistic.csv",
        "synthetic": "mine_part2_co_balanced_cgan.csv",
        "target":    "severity",
    },
    {
        "gas": "CO2",
        "real":      "mine_part2_co2_realistic.csv",
        "synthetic": "mine_part2_co2_balanced_cgan.csv",
        "target":    "severity",
    },
    {
        "gas": "H2",
        "real":      "mine_part2_h2_realistic.csv",
        "synthetic": "mine_part2_h2_balanced_cgan.csv",
        "target":    "severity",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: MMD
# ─────────────────────────────────────────────────────────────────────────────
def compute_mmd(X_real, X_syn, gamma=1.0):
    from sklearn.metrics.pairwise import rbf_kernel
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

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
all_results = []

for ds in DATASETS:
    gas = ds["gas"]
    print(f"\n{'='*65}")
    print(f"  EVALUATING: {gas}")
    print(f"{'='*65}")

    real_path = os.path.join(DATA_DIR, ds["real"])
    syn_path  = os.path.join(DATA_DIR, ds["synthetic"])

    df_real = pd.read_csv(real_path)
    df_syn  = pd.read_csv(syn_path)

    # Only keep existing feature cols
    feats = [c for c in FEATURE_COLS if c in df_real.columns and c in df_syn.columns]
    target = ds["target"]

    X_real = df_real[feats].values.astype(float)
    X_syn  = df_syn[feats].values.astype(float)
    y_real = df_real[target].values.astype(int)
    y_syn  = df_syn[target].values.astype(int)

    scaler = StandardScaler()
    scaler.fit(X_real)
    Xr_sc = scaler.transform(X_real)
    Xs_sc = scaler.transform(X_syn)

    result = {"gas": gas, "n_real": len(X_real), "n_syn": len(X_syn)}

    # ── 1. KDE / Histogram Plots ──────────────────────────────────────────
    print("  [1] KDE / Histogram plots...")
    n_feats = len(feats)
    fig, axes = plt.subplots(1, n_feats, figsize=(5 * n_feats, 4))
    if n_feats == 1:
        axes = [axes]
    for ax, feat in zip(axes, feats):
        ax.hist(df_real[feat].values, bins=50, alpha=0.5, density=True, label="Real", color="#2196F3")
        ax.hist(df_syn[feat].values,  bins=50, alpha=0.5, density=True, label="Synthetic", color="#FF9800")
        try:
            df_real[feat].plot.kde(ax=ax, color="#1565C0", linewidth=2)
            df_syn[feat].plot.kde(ax=ax, color="#E65100", linewidth=2)
        except Exception:
            pass
        ax.set_title(f"{gas} | {feat}")
        ax.legend(fontsize=8)
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
    plt.suptitle(f"{gas}: Real vs CGAN Synthetic KDE / Histogram", fontsize=13, fontweight="bold")
    plt.tight_layout()
    kde_path = os.path.join(PLOT_DIR, f"{gas}_kde_hist.png")
    plt.savefig(kde_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {kde_path}")

    # ── 2. Boxplots ───────────────────────────────────────────────────────
    print("  [2] Boxplot plots...")
    box_data_r = pd.DataFrame(df_real[feats], columns=feats)
    box_data_r["Source"] = "Real"
    box_data_s = pd.DataFrame(df_syn[feats], columns=feats)
    box_data_s["Source"] = "Synthetic"
    box_combined = pd.concat([box_data_r, box_data_s], ignore_index=True)
    box_melted = box_combined.melt(id_vars="Source", var_name="Feature", value_name="Value")

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(data=box_melted, x="Feature", y="Value", hue="Source",
                palette={"Real": "#2196F3", "Synthetic": "#FF9800"}, ax=ax, linewidth=1.2)
    ax.set_title(f"{gas}: Real vs CGAN Synthetic Boxplots", fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Value")
    plt.tight_layout()
    box_path = os.path.join(PLOT_DIR, f"{gas}_boxplots.png")
    plt.savefig(box_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {box_path}")

    # ── 3. Statistical Summary ─────────────────────────────────────────────
    print("  [3] Statistical summary...")
    stat_rows = []
    for feat in feats:
        r_vals = df_real[feat].values
        s_vals = df_syn[feat].values
        mean_bias = abs(np.mean(s_vals) - np.mean(r_vals)) / (abs(np.mean(r_vals)) + 1e-9) * 100
        stat_rows.append({
            "Feature": feat,
            "Real_Mean": round(np.mean(r_vals), 3),
            "Syn_Mean": round(np.mean(s_vals), 3),
            "Real_Std": round(np.std(r_vals), 3),
            "Syn_Std": round(np.std(s_vals), 3),
            "Real_Median": round(np.median(r_vals), 3),
            "Syn_Median": round(np.median(s_vals), 3),
            "Mean_Bias_%": round(mean_bias, 2),
        })
    stat_df = pd.DataFrame(stat_rows)
    result["stats"] = stat_df

    # ── 4. Correlation Heatmaps ────────────────────────────────────────────
    print("  [4] Correlation heatmaps...")
    corr_real = df_real[feats].corr()
    corr_syn  = df_syn[feats].corr()
    diff_corr = (corr_real - corr_syn).abs()
    frob_norm = np.linalg.norm(diff_corr.values)
    result["frob_norm"] = round(frob_norm, 4)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(corr_real, annot=True, fmt=".2f", cmap="Blues", ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title(f"{gas}: Real Correlation", fontweight="bold")
    sns.heatmap(corr_syn, annot=True, fmt=".2f", cmap="Oranges", ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title(f"{gas}: Synthetic Correlation", fontweight="bold")
    sns.heatmap(diff_corr, annot=True, fmt=".2f", cmap="Reds", ax=axes[2], vmin=0, vmax=1)
    axes[2].set_title(f"{gas}: Absolute Difference (Frob={frob_norm:.3f})", fontweight="bold")
    plt.suptitle(f"{gas}: Correlation Heatmap Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    heat_path = os.path.join(PLOT_DIR, f"{gas}_correlation_heatmaps.png")
    plt.savefig(heat_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {heat_path}")

    # ── 5. PCA Projection ──────────────────────────────────────────────────
    print("  [5] PCA projection...")
    n_plot = min(2000, len(X_real), len(X_syn))
    pca = PCA(n_components=2, random_state=42)
    Xr_pca = pca.fit_transform(Xr_sc[:n_plot])
    Xs_pca = pca.transform(Xs_sc[:n_plot])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Xr_pca[:, 0], Xr_pca[:, 1], alpha=0.3, s=10, c="#2196F3", label="Real")
    ax.scatter(Xs_pca[:, 0], Xs_pca[:, 1], alpha=0.3, s=10, c="#FF9800", label="Synthetic")
    ax.set_title(f"{gas}: PCA Projection (2D)\nExplained Var: {pca.explained_variance_ratio_.sum()*100:.1f}%",
                 fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend()
    plt.tight_layout()
    pca_path = os.path.join(PLOT_DIR, f"{gas}_pca.png")
    plt.savefig(pca_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {pca_path}")

    pca_centroid_dist = np.linalg.norm(Xr_pca.mean(axis=0) - Xs_pca.mean(axis=0))
    result["pca_centroid_dist"] = round(pca_centroid_dist, 4)
    result["pca_explained_var"] = round(pca.explained_variance_ratio_.sum() * 100, 2)

    # ── 6. t-SNE Embedding ─────────────────────────────────────────────────
    print("  [6] t-SNE embedding (this may take a moment)...")
    n_tsne = min(1000, len(X_real), len(X_syn))
    idx_r = np.random.choice(len(Xr_sc), n_tsne, replace=False)
    idx_s = np.random.choice(len(Xs_sc), n_tsne, replace=False)
    combined_tsne = np.vstack([Xr_sc[idx_r], Xs_sc[idx_s]])
    labels_tsne = np.array(["Real"] * n_tsne + ["Synthetic"] * n_tsne)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500)
    embedded = tsne.fit_transform(combined_tsne)

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl, color in [("Real", "#2196F3"), ("Synthetic", "#FF9800")]:
        mask = labels_tsne == lbl
        ax.scatter(embedded[mask, 0], embedded[mask, 1], alpha=0.4, s=12, c=color, label=lbl)
    ax.set_title(f"{gas}: t-SNE Embedding (Real vs Synthetic)", fontweight="bold")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend()
    plt.tight_layout()
    tsne_path = os.path.join(PLOT_DIR, f"{gas}_tsne.png")
    plt.savefig(tsne_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {tsne_path}")

    tsne_centroid_r = embedded[:n_tsne].mean(axis=0)
    tsne_centroid_s = embedded[n_tsne:].mean(axis=0)
    tsne_dist = np.linalg.norm(tsne_centroid_r - tsne_centroid_s)
    result["tsne_centroid_dist"] = round(tsne_dist, 4)

    # ── 7. KS Test ────────────────────────────────────────────────────────
    print("  [7] KS Test...")
    ks_rows = []
    for feat in feats:
        d_stat, p_val = ks_2samp(df_real[feat].values, df_syn[feat].values)
        ks_rows.append({"Feature": feat, "KS_D_Statistic": round(d_stat, 4), "KS_p_value": round(p_val, 4),
                        "Pass (p>0.05)": "PASS" if p_val > 0.05 else "FAIL"})
    ks_df = pd.DataFrame(ks_rows)
    result["ks_df"] = ks_df
    print(ks_df.to_string(index=False))

    # ── 8. Wasserstein Distance ───────────────────────────────────────────
    print("  [8] Wasserstein Distance...")
    wass_rows = []
    for feat in feats:
        w_dist = wasserstein_distance(df_real[feat].values, df_syn[feat].values)
        wass_rows.append({"Feature": feat, "Wasserstein_Distance": round(w_dist, 4)})
    wass_df = pd.DataFrame(wass_rows)
    result["wass_df"] = wass_df
    print(wass_df.to_string(index=False))

    # ── 9. MMD ────────────────────────────────────────────────────────────
    print("  [9] MMD (RBF Kernel)...")
    mmd2 = compute_mmd(Xr_sc, Xs_sc)
    result["mmd2"] = round(mmd2, 6)
    print(f"     MMD^2 = {mmd2:.6f}  ({'PASS' if mmd2 < 0.01 else 'REVIEW'})")

    # ── 10. TSTR ─────────────────────────────────────────────────────────
    print("  [10] TSTR (Train Synthetic, Test Real)...")
    X_syn_tr, y_syn_tr = Xs_sc, y_syn
    X_real_te, y_real_te = Xr_sc, y_real
    clf_tstr = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
    clf_tstr.fit(X_syn_tr, y_syn_tr)
    tstr_acc = accuracy_score(y_real_te, clf_tstr.predict(X_real_te))
    result["tstr_acc"] = round(tstr_acc * 100, 2)
    print(f"     TSTR Accuracy: {tstr_acc*100:.2f}%")

    # ── 11. TRTS ─────────────────────────────────────────────────────────
    print("  [11] TRTS (Train Real, Test Synthetic)...")
    clf_trts = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
    clf_trts.fit(Xr_sc, y_real)
    trts_acc = accuracy_score(y_syn, clf_trts.predict(Xs_sc))
    result["trts_acc"] = round(trts_acc * 100, 2)
    print(f"     TRTS Accuracy: {trts_acc*100:.2f}%")

    # ── 12. Classifier Distinguishability ─────────────────────────────────
    print("  [12] Classifier Distinguishability (Real vs Synthetic)...")
    n_disc = min(len(X_real), len(X_syn))
    Xr_disc = Xr_sc[:n_disc]
    Xs_disc = Xs_sc[:n_disc]
    X_disc = np.vstack([Xr_disc, Xs_disc])
    y_disc = np.array([0] * n_disc + [1] * n_disc)

    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_disc, y_disc, test_size=0.3, random_state=42)
    clf_disc = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf_disc.fit(X_tr_d, y_tr_d)
    disc_proba = clf_disc.predict_proba(X_te_d)[:, 1]
    disc_auc = roc_auc_score(y_te_d, disc_proba)
    disc_acc = accuracy_score(y_te_d, clf_disc.predict(X_te_d))
    result["disc_auc"] = round(disc_auc, 4)
    result["disc_acc"] = round(disc_acc * 100, 2)
    print(f"     Discriminator AUC: {disc_auc:.4f}  (0.50=indistinguishable)")
    print(f"     Discriminator Accuracy: {disc_acc*100:.2f}%")

    # ── 13. Time-Series Autocorrelation ───────────────────────────────────
    print("  [13] Time-Series Autocorrelation comparison...")
    feat_ac = feats[0]  # primary feature (ppm)
    r_vals = df_real[feat_ac].values[:2000]
    s_vals = df_syn[feat_ac].values[:2000]
    max_lag = 40

    acf_r = [np.corrcoef(r_vals[:-k], r_vals[k:])[0, 1] if k < len(r_vals) else 0 for k in range(1, max_lag+1)]
    acf_s = [np.corrcoef(s_vals[:-k], s_vals[k:])[0, 1] if k < len(s_vals) else 0 for k in range(1, max_lag+1)]

    acf_r = np.nan_to_num(acf_r)
    acf_s = np.nan_to_num(acf_s)

    acf_mae = np.mean(np.abs(np.array(acf_r) - np.array(acf_s)))
    result["acf_mae"] = round(acf_mae, 4)

    fig, ax = plt.subplots(figsize=(10, 4))
    lags = list(range(1, max_lag + 1))
    ax.plot(lags, acf_r, marker="o", markersize=4, linewidth=1.5, color="#2196F3", label="Real")
    ax.plot(lags, acf_s, marker="s", markersize=4, linewidth=1.5, color="#FF9800", label="Synthetic")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.fill_between(lags, -1.96/np.sqrt(len(r_vals)), 1.96/np.sqrt(len(r_vals)), alpha=0.15, color="gray", label="95% CI")
    ax.set_title(f"{gas}: Autocorrelation Function Comparison (ACF MAE={acf_mae:.4f})", fontweight="bold")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend()
    plt.tight_layout()
    acf_path = os.path.join(PLOT_DIR, f"{gas}_autocorrelation.png")
    plt.savefig(acf_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {acf_path}")

    all_results.append(result)
    print(f"  [{gas}] All 13 evaluations complete.")

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE MARKDOWN REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n\nGenerating CGAN evaluation report...")
REPORT_PATH = os.path.join(DOCS_DIR, "CGAN_SYNTHETIC_EVALUATION.md")

plot_dir_rel = "../gas_sensors/evaluation_plots"

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("# FIELD-MIND: PyTorch CGAN Synthetic Data Evaluation Report\n\n")
    f.write("Comprehensive evaluation of CGAN-generated synthetic mine gas sensor telemetry across all 13 statistical, distributional, and machine-learning-utility metrics.\n\n")
    f.write("---\n\n")

    # Summary Table
    f.write("## Executive Summary Table\n\n")
    f.write("| Gas | Real Rows | Syn Rows | KS Pass Rate | Frob Norm | MMD^2 | TSTR Acc% | TRTS Acc% | Disc AUC | ACF MAE |\n")
    f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for r in all_results:
        if "ks_df" in r:
            ks_pass = (r["ks_df"]["Pass (p>0.05)"] == "PASS").sum()
            ks_total = len(r["ks_df"])
        else:
            ks_pass, ks_total = 0, 0
        f.write(f"| **{r['gas']}** | {r['n_real']:,} | {r['n_syn']:,} | {ks_pass}/{ks_total} | {r.get('frob_norm','N/A')} | {r.get('mmd2','N/A')} | {r.get('tstr_acc','N/A')} | {r.get('trts_acc','N/A')} | {r.get('disc_auc','N/A')} | {r.get('acf_mae','N/A')} |\n")
    f.write("\n---\n\n")

    for r in all_results:
        gas = r["gas"]
        f.write(f"## {gas} Gas — Full Evaluation\n\n")

        f.write("### 1. Histogram / KDE Overlap\n\n")
        f.write(f"![{gas} KDE Histogram]({plot_dir_rel}/{gas}_kde_hist.png)\n\n")

        f.write("### 2. Boxplots Analysis\n\n")
        f.write(f"![{gas} Boxplots]({plot_dir_rel}/{gas}_boxplots.png)\n\n")

        f.write("### 3. Statistical Summary\n\n")
        if "stats" in r:
            f.write(r["stats"].to_markdown(index=False))
            f.write("\n\n")

        f.write("### 4. Correlation Heatmaps\n\n")
        f.write(f"![{gas} Correlation Heatmaps]({plot_dir_rel}/{gas}_correlation_heatmaps.png)\n\n")
        f.write(f"**Frobenius Norm of Correlation Difference**: `{r.get('frob_norm', 'N/A')}` (threshold ≤ 0.15)\n\n")

        f.write("### 5. PCA Projection (2D)\n\n")
        f.write(f"![{gas} PCA]({plot_dir_rel}/{gas}_pca.png)\n\n")
        f.write(f"**PCA Explained Variance**: {r.get('pca_explained_var', 'N/A')}%  \n")
        f.write(f"**PCA Centroid Distance**: `{r.get('pca_centroid_dist', 'N/A')}`\n\n")

        f.write("### 6. t-SNE Embedding\n\n")
        f.write(f"![{gas} t-SNE]({plot_dir_rel}/{gas}_tsne.png)\n\n")
        f.write(f"**t-SNE Centroid Distance**: `{r.get('tsne_centroid_dist', 'N/A')}`\n\n")

        f.write("### 7. Kolmogorov-Smirnov (KS) Test\n\n")
        if "ks_df" in r:
            f.write(r["ks_df"].to_markdown(index=False))
            f.write("\n\n")

        f.write("### 8. Wasserstein Distance\n\n")
        if "wass_df" in r:
            f.write(r["wass_df"].to_markdown(index=False))
            f.write("\n\n")

        f.write("### 9. Maximum Mean Discrepancy (MMD)\n\n")
        mmd = r.get("mmd2", "N/A")
        status = "PASS (Good)" if isinstance(mmd, float) and mmd < 0.01 else "REVIEW"
        f.write(f"| Metric | Value | Threshold | Status |\n")
        f.write(f"| :--- | :---: | :---: | :---: |\n")
        f.write(f"| MMD^2 (RBF kernel) | `{mmd}` | ≤ 0.01 | **{status}** |\n\n")

        f.write("### 10. TSTR — Train Synthetic, Test Real\n\n")
        f.write("| Metric | Value |\n| :--- | :---: |\n")
        f.write(f"| TSTR Accuracy | **{r.get('tstr_acc', 'N/A')}%** |\n\n")

        f.write("### 11. TRTS — Train Real, Test Synthetic\n\n")
        f.write("| Metric | Value |\n| :--- | :---: |\n")
        f.write(f"| TRTS Accuracy | **{r.get('trts_acc', 'N/A')}%** |\n\n")

        f.write("### 12. Classifier Distinguishability\n\n")
        disc_auc = r.get("disc_auc", "N/A")
        disc_status = "EXCELLENT (Indistinguishable)" if isinstance(disc_auc, float) and disc_auc < 0.60 else "REVIEW"
        f.write("| Metric | Value | Target | Status |\n| :--- | :---: | :---: | :---: |\n")
        f.write(f"| Discriminator AUC | `{disc_auc}` | ~0.50 | **{disc_status}** |\n")
        f.write(f"| Discriminator Accuracy | `{r.get('disc_acc', 'N/A')}%` | ~50% | - |\n\n")

        f.write("### 13. Time-Series Autocorrelation\n\n")
        f.write(f"![{gas} Autocorrelation]({plot_dir_rel}/{gas}_autocorrelation.png)\n\n")
        acf = r.get("acf_mae", "N/A")
        acf_status = "PASS (Good)" if isinstance(acf, float) and acf < 0.05 else "REVIEW"
        f.write("| Metric | Value | Threshold | Status |\n| :--- | :---: | :---: | :---: |\n")
        f.write(f"| ACF MAE | `{acf}` | ≤ 0.05 | **{acf_status}** |\n\n")

        f.write("---\n\n")

print(f"\nReport generated: {REPORT_PATH}")
print("Done! All 13 evaluations complete across all gas datasets.")
