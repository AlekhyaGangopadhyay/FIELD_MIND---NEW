"""
generate_combined_cgan_plots.py — Master Visual Subplot Grid Generator for CGAN Evaluation

Generates 5 unified high-resolution multi-panel subplot grid figures across all 5 datasets:
  1. Part 1 (mine_part1_clean.csv vs mine_part1_balanced_gan.csv)
  2. Part 2 CH4 (mine_part2_ch4_realistic.csv vs mine_part2_ch4_balanced_cgan.csv)
  3. Part 2 CO (mine_part2_co_realistic.csv vs mine_part2_co_balanced_cgan.csv)
  4. Part 2 CO2 (mine_part2_co2_realistic.csv vs mine_part2_co2_balanced_cgan.csv)
  5. Part 2 H2 (mine_part2_h2_realistic.csv vs mine_part2_h2_balanced_cgan.csv)

Generated Subplot Figures:
  - gas_sensors/evaluation_plots/combined_cgan_hist_kde.png
  - gas_sensors/evaluation_plots/combined_cgan_boxplots.png
  - gas_sensors/evaluation_plots/combined_cgan_correlation.png
  - gas_sensors/evaluation_plots/combined_cgan_pca_tsne.png
  - gas_sensors/evaluation_plots/combined_cgan_pairwise.png
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

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "data")
PLOT_DIR = os.path.join(script_dir, "evaluation_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DATASETS = [
    {
        "name": "Part 1 (is_warmup)",
        "tag": "part1",
        "real": os.path.join(DATA_DIR, "mine_part1_clean.csv"),
        "syn": os.path.join(DATA_DIR, "mine_part1_balanced_gan.csv"),
        "feats": ["air_quality", "smoke", "alcohol", "flamable_gas", "MQ136_raw", "MQ7_raw", "t", "h"],
        "target": "is_warmup",
        "is_part1": True,
    },
    {
        "name": "Part 2 Methane (CH4)",
        "tag": "ch4",
        "real": os.path.join(DATA_DIR, "mine_part2_ch4_realistic.csv"),
        "syn": os.path.join(DATA_DIR, "mine_part2_ch4_balanced_cgan.csv"),
        "feats": ["pct", "ppm", "ppm_noisy"],
        "target": "severity",
        "is_part1": False,
    },
    {
        "name": "Part 2 Carbon Monoxide (CO)",
        "tag": "co",
        "real": os.path.join(DATA_DIR, "mine_part2_co_realistic.csv"),
        "syn": os.path.join(DATA_DIR, "mine_part2_co_balanced_cgan.csv"),
        "feats": ["pct", "ppm", "ppm_noisy"],
        "target": "severity",
        "is_part1": False,
    },
    {
        "name": "Part 2 Carbon Dioxide (CO2)",
        "tag": "co2",
        "real": os.path.join(DATA_DIR, "mine_part2_co2_realistic.csv"),
        "syn": os.path.join(DATA_DIR, "mine_part2_co2_balanced_cgan.csv"),
        "feats": ["pct", "ppm", "ppm_noisy"],
        "target": "severity",
        "is_part1": False,
    },
    {
        "name": "Part 2 Hydrogen (H2)",
        "tag": "h2",
        "real": os.path.join(DATA_DIR, "mine_part2_h2_realistic.csv"),
        "syn": os.path.join(DATA_DIR, "mine_part2_h2_balanced_cgan.csv"),
        "feats": ["pct", "ppm", "ppm_noisy"],
        "target": "severity",
        "is_part1": False,
    },
]


def load_dataset_pair(ds_info):
    df_real_orig = pd.read_csv(ds_info["real"])
    df_syn_orig = pd.read_csv(ds_info["syn"])

    feats = ds_info["feats"]

    if ds_info["is_part1"]:
        df_real = df_real_orig[df_real_orig["is_warmup"] == True][feats].dropna()
        df_syn = df_syn_orig.iloc[len(df_real_orig):][feats].dropna()
    else:
        df_real = df_real_orig[feats].dropna()
        df_syn = df_syn_orig.iloc[len(df_real_orig):][feats].dropna()

    return df_real, df_syn


def generate_combined_hist_kde():
    print("[1/5] Generating Combined Histograms + KDE Overlap Subplot Grid...")
    fig, axes = plt.subplots(5, 4, figsize=(20, 18))

    for row_idx, ds in enumerate(DATASETS):
        df_r, df_s = load_dataset_pair(ds)
        feats = ds["feats"][:4]  # Top 4 features for layout uniformity

        for col_idx, feat in enumerate(feats):
            ax = axes[row_idx, col_idx]
            sns.histplot(df_r[feat], ax=ax, color="#1f77b4", label="Real", kde=True, stat="density", alpha=0.35, bins=25)
            sns.histplot(df_s[feat], ax=ax, color="#ff7f0e", label="Synthetic CGAN", kde=True, stat="density", alpha=0.35, bins=25)
            ax.set_title(f"{ds['name']} | {feat}", fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Density" if col_idx == 0 else "")
            ax.legend(fontsize=8)

        # Blank out remaining columns if fewer than 4 features
        for col_idx in range(len(feats), 4):
            axes[row_idx, col_idx].axis("off")

    plt.suptitle("Master Subplot: Distribution Overlap (Histograms + KDE) Across All 5 Telemetry Datasets", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(PLOT_DIR, "combined_cgan_hist_kde.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")


def generate_combined_boxplots():
    print("[2/5] Generating Combined Boxplots Subplot Grid...")
    fig, axes = plt.subplots(5, 4, figsize=(20, 18))

    for row_idx, ds in enumerate(DATASETS):
        df_r, df_s = load_dataset_pair(ds)
        feats = ds["feats"][:4]

        for col_idx, feat in enumerate(feats):
            ax = axes[row_idx, col_idx]
            data_to_plot = [df_r[feat].dropna(), df_s[feat].dropna()]
            ax.boxplot(data_to_plot, patch_artist=True, tick_labels=["Real", "Synthetic"],
                       boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
                       medianprops=dict(color="red", linewidth=1.5))
            ax.set_title(f"{ds['name']} | {feat}", fontsize=10, fontweight="bold")

        for col_idx in range(len(feats), 4):
            axes[row_idx, col_idx].axis("off")

    plt.suptitle("Master Subplot: Side-by-Side Boxplots Comparison Across All 5 Telemetry Datasets", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(PLOT_DIR, "combined_cgan_boxplots.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")


def generate_combined_correlation():
    print("[3/5] Generating Combined Correlation Heatmaps Subplot Grid...")
    fig, axes = plt.subplots(5, 3, figsize=(18, 22))

    for row_idx, ds in enumerate(DATASETS):
        df_r, df_s = load_dataset_pair(ds)
        feats = ds["feats"]

        corr_r = df_r[feats].corr().fillna(0)
        corr_s = df_s[feats].corr().fillna(0)
        corr_diff = (corr_r - corr_s).abs()
        frob = np.linalg.norm(corr_diff.values)

        # Real Matrix
        sns.heatmap(corr_r, ax=axes[row_idx, 0], annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=False)
        axes[row_idx, 0].set_title(f"{ds['name']} Real Correlation", fontsize=10, fontweight="bold")

        # Synthetic Matrix
        sns.heatmap(corr_s, ax=axes[row_idx, 1], annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=False)
        axes[row_idx, 1].set_title(f"{ds['name']} Synthetic Correlation", fontsize=10, fontweight="bold")

        # Difference Matrix
        sns.heatmap(corr_diff, ax=axes[row_idx, 2], annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1, cbar=False)
        axes[row_idx, 2].set_title(f"{ds['name']} |Real - Syn| (Frob={frob:.3f})", fontsize=10, fontweight="bold")

    plt.suptitle("Master Subplot: Correlation Heatmaps & Absolute Drift Across All 5 Telemetry Datasets", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(PLOT_DIR, "combined_cgan_correlation.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")


def generate_combined_pca_tsne():
    print("[4/5] Generating Combined PCA & t-SNE Projections Subplot Grid...")
    fig, axes = plt.subplots(5, 2, figsize=(16, 22))

    for row_idx, ds in enumerate(DATASETS):
        df_r, df_s = load_dataset_pair(ds)
        feats = ds["feats"]

        Xr = df_r[feats].values.astype(float)
        Xs = df_s[feats].values.astype(float)

        scaler = StandardScaler()
        Xr_sc = scaler.fit_transform(Xr)
        Xs_sc = scaler.transform(Xs)

        n_plot = min(1500, len(Xr_sc), len(Xs_sc))
        Xr_sub = Xr_sc[:n_plot]
        Xs_sub = Xs_sc[:n_plot]

        # PCA 2D
        pca = PCA(n_components=2, random_state=42)
        X_all_pca = pca.fit_transform(np.vstack([Xr_sub, Xs_sub]))
        r_pca = X_all_pca[:n_plot]
        s_pca = X_all_pca[n_plot:]

        ax_pca = axes[row_idx, 0]
        ax_pca.scatter(s_pca[:, 0], s_pca[:, 1], c="#ff7f0e", alpha=0.4, label="Synthetic", s=18)
        ax_pca.scatter(r_pca[:, 0], r_pca[:, 1], c="#1f77b4", alpha=0.7, label="Real", s=22, edgecolors="k", linewidth=0.3)
        ax_pca.set_title(f"{ds['name']} | PCA 2D (Var: {pca.explained_variance_ratio_.sum():.1%})", fontsize=10, fontweight="bold")
        ax_pca.set_xlabel("PC1")
        ax_pca.set_ylabel("PC2")
        ax_pca.legend(fontsize=8)

        # t-SNE 2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_all_tsne = tsne.fit_transform(np.vstack([Xr_sub[:800], Xs_sub[:800]]))
        r_tsne = X_all_tsne[:800]
        s_tsne = X_all_tsne[800:]

        ax_tsne = axes[row_idx, 1]
        ax_tsne.scatter(s_tsne[:, 0], s_tsne[:, 1], c="#ff7f0e", alpha=0.4, label="Synthetic", s=18)
        ax_tsne.scatter(r_tsne[:, 0], r_tsne[:, 1], c="#1f77b4", alpha=0.7, label="Real", s=22, edgecolors="k", linewidth=0.3)
        ax_tsne.set_title(f"{ds['name']} | t-SNE 2D Embedding", fontsize=10, fontweight="bold")
        ax_tsne.set_xlabel("t-SNE Dim 1")
        ax_tsne.set_ylabel("t-SNE Dim 2")
        ax_tsne.legend(fontsize=8)

    plt.suptitle("Master Subplot: PCA & t-SNE 2D Manifold Overlays Across All 5 Telemetry Datasets", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(PLOT_DIR, "combined_cgan_pca_tsne.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")


def generate_combined_pairwise():
    print("[5/5] Generating Combined Pairwise Scatter Subplot Grid...")
    fig, axes = plt.subplots(5, 2, figsize=(16, 22))

    pair_map = {
        "part1": [("air_quality", "flamable_gas"), ("MQ136_raw", "smoke")],
        "ch4": [("pct", "ppm_noisy"), ("ppm", "ppm_noisy")],
        "co": [("pct", "ppm_noisy"), ("ppm", "ppm_noisy")],
        "co2": [("pct", "ppm_noisy"), ("ppm", "ppm_noisy")],
        "h2": [("pct", "ppm_noisy"), ("ppm", "ppm_noisy")],
    }

    for row_idx, ds in enumerate(DATASETS):
        df_r, df_s = load_dataset_pair(ds)
        pairs = pair_map[ds["tag"]]

        for col_idx, (f1, f2) in enumerate(pairs):
            ax = axes[row_idx, col_idx]
            ax.scatter(df_s[f1], df_s[f2], c="#ff7f0e", alpha=0.3, label="Synthetic", s=15)
            ax.scatter(df_r[f1], df_r[f2], c="#1f77b4", alpha=0.6, label="Real", s=20, edgecolors="k", linewidth=0.2)
            ax.set_title(f"{ds['name']} | {f1} vs {f2}", fontsize=10, fontweight="bold")
            ax.set_xlabel(f1)
            ax.set_ylabel(f2)
            ax.legend(fontsize=8)

    plt.suptitle("Master Subplot: Pairwise Feature Relationship Scatter Comparisons Across All 5 Telemetry Datasets", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(PLOT_DIR, "combined_cgan_pairwise.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")


def run_all():
    print("=" * 70)
    print("  GENERATING COMPOSITE CGAN MASTER EVALUATION SUBPLOT FIGURES")
    print("=" * 70)
    generate_combined_hist_kde()
    generate_combined_boxplots()
    generate_combined_correlation()
    generate_combined_pca_tsne()
    generate_combined_pairwise()
    print("\nAll 5 master subplot grid figures successfully generated!")


if __name__ == "__main__":
    run_all()
