"""
Generates publication-quality evaluation figures:
1. results/figures/stale_passage_rate.png
2. results/figures/conflict_rate.png
3. results/figures/poincare_disk_projection.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch

from models import HyperbolicRetriever, exp_map_origin

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGS_DIR = os.path.join(RESULTS_DIR, "figures")

def plot_stale_rate(results_data):
    """Figure 1: Stale passage rate by system, grouped by version stated vs unstated."""
    systems = [
        "A_BM25",
        "B_Euclidean",
        "B_trained_384d",
        "C_Euclidean_Filter",
        "D_Hyperbolic_384d",
        "E_Hyperbolic_Filter",
        "F_RRF_B_D"
    ]
    labels = [
        "A: BM25",
        "B: Euclidean",
        "B: Trained",
        "C: Filter",
        "D: Hyperbolic",
        "E: Hyp+Filter",
        "F: RRF"
    ]
    
    stated_rates = [results_data["systems"][s]["version_stated"]["stale_rate@5"] * 100 for s in systems]
    unstated_rates = [results_data["systems"][s]["version_unstated"]["stale_rate@5"] * 100 for s in systems]
    
    x = np.arange(len(systems))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    rects1 = ax.bar(x - width/2, unstated_rates, width, label="Version Unstated (Headline)", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, stated_rates, width, label="Version Stated", color="#d95f02")
    
    ax.set_ylabel("Stale Passage Rate @ 5 (%)", fontsize=12)
    ax.set_title("Stale Passage Retrieval Rate by System and Query Type", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Value labels
    for rect in rects1:
        h = rect.get_height()
        if h > 0.05:
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        if h > 0.05:
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(FIGS_DIR, "stale_passage_rate.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def plot_conflict_rate(results_data):
    """Figure 2: Conflict rate @ 5 by system."""
    systems = [
        "A_BM25",
        "B_Euclidean",
        "B_trained_384d",
        "C_Euclidean_Filter",
        "D_Hyperbolic_384d",
        "E_Hyperbolic_Filter",
        "F_RRF_B_D",
        "B_trained_32d",
        "D_Hyperbolic_32d"
    ]
    labels = [
        "A: BM25",
        "B: Euclidean",
        "B: Trained (384d)",
        "C: Filter",
        "D: Hyp (384d)",
        "E: Hyp+Filter",
        "F: RRF (384d)",
        "B: Trained (32d)",
        "D: Hyp (32d)"
    ]
    
    overall_conflicts = [results_data["systems"][s]["overall"]["conflict_rate@5"] * 100 for s in systems]
    unstated_conflicts = [results_data["systems"][s]["version_unstated"]["conflict_rate@5"] * 100 for s in systems]
    
    x = np.arange(len(systems))
    width = 0.38
    
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    r1 = ax.bar(x - width/2, unstated_conflicts, width, label="Version Unstated", color="#c0392b")
    r2 = ax.bar(x + width/2, overall_conflicts, width, label="Overall", color="#7f8c8d")
    
    ax.set_ylabel("Conflict Rate @ 5 (%)", fontsize=12)
    ax.set_title("Conflict Rate @ 5: Queries with Both Gold and Stale Passages in Top 5", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for r in r1:
        h = r.get_height()
        if h > 0.05:
            ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for r in r2:
        h = r.get_height()
        if h > 0.05:
            ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(FIGS_DIR, "conflict_rate.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def plot_poincare_disk():
    """
    Figure 3: 2D Poincare disk plot of Nextcloud passages across versions 28, 29, 30.
    Uses PCA in tangent space followed by Poincare ball projection.
    """
    with open(os.path.join(PROCESSED_DIR, "passages.jsonl"), "r", encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    p_data = torch.load(os.path.join(PROCESSED_DIR, "passage_embeddings.pt"), map_location="cpu")
    passage_embs = p_data["embeddings"]

    # Filter Nextcloud passages
    nc_indices = [i for i, p in enumerate(passages) if p["product"] == "Nextcloud"]
    nc_passages = [passages[i] for i in nc_indices]
    nc_embs = passage_embs[nc_indices]

    # Sample up to 150 passages per version for visual clarity
    by_version = {"28": [], "29": [], "30": []}
    for idx, p in enumerate(nc_passages):
        v = p["version"]
        if v in by_version and len(by_version[v]) < 120:
            by_version[v].append(idx)

    sampled_indices = []
    sampled_versions = []
    for v in ["28", "29", "30"]:
        for idx in by_version[v]:
            sampled_indices.append(idx)
            sampled_versions.append(v)

    sampled_embs = nc_embs[sampled_indices]

    # Load trained model to get tangent space projection
    weights_path = os.path.join(RESULTS_DIR, "hyperbolic_model_384.pt")
    state_dict = torch.load(weights_path, map_location="cpu")
    num_tax = state_dict["taxonomy_emb.weight"].shape[0]
    model = HyperbolicRetriever(in_dim=384, proj_dim=384, num_taxonomy_nodes=num_tax)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        tangent_vecs = model.project_tangent(sampled_embs).numpy()

    # PCA down to 2 dimensions in tangent space
    pca = PCA(n_components=2, random_state=42)
    tangent_2d = pca.fit_transform(tangent_vecs)

    # Poincare exponential map in 2D: y = tanh(r / 2) * (v / r)
    norms = np.linalg.norm(tangent_2d, axis=1, keepdims=True)
    norms_clamped = np.clip(norms, 1e-7, 8.0)
    poincare_2d = np.tanh(norms_clamped / 2.0) * (tangent_2d / norms_clamped)

    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=300)
    
    # Draw boundary circle
    circle = plt.Circle((0, 0), 1.0, color="#2c3e50", fill=False, linestyle="-", linewidth=2.0)
    ax.add_artist(circle)
    
    # Internal geodesic reference rings
    for r in [0.33, 0.66]:
        ref_circle = plt.Circle((0, 0), r, color="#bdc3c7", fill=False, linestyle=":", linewidth=1.0)
        ax.add_artist(ref_circle)

    colors = {"28": "#e74c3c", "29": "#f39c12", "30": "#27ae60"}
    labels = {"28": "Nextcloud 28 (Older Stale)", "29": "Nextcloud 29 (Intermediate Stale)", "30": "Nextcloud 30 (Target/Newest)"}

    for v in ["28", "29", "30"]:
        v_mask = [ver == v for ver in sampled_versions]
        pts = poincare_2d[v_mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=colors[v], label=labels[v], alpha=0.75, s=32, edgecolors="none")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("2D Poincaré Disk Projection (Nextcloud Documentation Passages)\n[2D PCA in Tangent Space Projected to Poincaré Ball for Visualisation]",
                 fontsize=11.5, fontweight="bold", pad=15)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), frameon=True, fontsize=9.5, ncol=1)

    plt.tight_layout()
    out_path = os.path.join(FIGS_DIR, "poincare_disk_projection.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def main():
    with open(os.path.join(RESULTS_DIR, "results.json"), "r", encoding="utf-8") as f:
        results_data = json.load(f)

    print("Generating figures...")
    plot_stale_rate(results_data)
    plot_conflict_rate(results_data)
    plot_poincare_disk()
    print("All figures generated successfully.")

if __name__ == "__main__":
    main()
