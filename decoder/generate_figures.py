"""
Warm Editorial Publication & Documentation Figure Generator
Adheres to the Ivory, Charcoal, Stone, and Dusty Rose design system.
Outputs high-resolution scientific plates for GitHub README and publications:
1. horizon_accuracy_curve.png
2. confusion_matrix.png
3. neuron_importance.png
4. neural_raster_preview.png
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

from decoder.connectome_model import ACTION_CLASSES, NEURON_NAMES

# Editorial Palette Constants
IVORY_BG = "#FAF7F2"
SURFACE_BG = "#F7F3EE"
INK_PRIMARY = "#24211E"
CHARCOAL = "#48443F"
MUTED_TEXT = "#6C6761"
STONE_GRID = "#E1D6CB"
STONE_BORDER = "#D6C7BD"
BLUSH_DARK = "#844D43"
DUSTY_ROSE = "#9E6D60"
MUTED_ROSE = "#B2887B"
WARM_ROSE = "#C2A193"
SAGE_GREEN = "#557864"


def set_editorial_style():
    """Configures matplotlib for a quiet luxury, warm editorial scientific aesthetic."""
    plt.rcParams.update({
        'figure.facecolor': IVORY_BG,
        'axes.facecolor': IVORY_BG,
        'savefig.facecolor': IVORY_BG,
        'font.sans-serif': ['Plus Jakarta Sans', 'Inter', 'Helvetica Neue', 'sans-serif'],
        'font.family': 'sans-serif',
        'text.color': INK_PRIMARY,
        'axes.labelcolor': CHARCOAL,
        'xtick.color': CHARCOAL,
        'ytick.color': CHARCOAL,
        'axes.edgecolor': STONE_BORDER,
        'axes.linewidth': 1.0,
        'grid.color': STONE_GRID,
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.85
    })


def generate_all_figures(
    metrics_path: str = "assets/models/benchmark_metrics.json",
    web_data_path: str = "web/data/test_neural_stream.json",
    output_dir: str = "assets/figures"
):
    os.makedirs(output_dir, exist_ok=True)
    set_editorial_style()

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # -------------------------------------------------------------
    # 1. Lookahead Horizon Prediction Curve
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    horizons = [item["horizon_ms"] for item in metrics["horizon_results"]]
    accuracies = [item["accuracy"] * 100 for item in metrics["horizon_results"]]

    # Curve with warm blush-dark styling
    ax.plot(horizons, accuracies, marker='o', markersize=7.5, color=BLUSH_DARK, 
            linewidth=2.2, label='Decoder Accuracy across Lead Times', zorder=4)
    
    # Lead window indicator
    ax.axvline(x=100, color=DUSTY_ROSE, linestyle='--', linewidth=1.4, alpha=0.85, 
               label='Optimal Pre-Motor Horizon (100 ms)')
    ax.scatter([100], [accuracies[horizons.index(100.0)]], color=BLUSH_DARK, s=110, 
               edgecolor=IVORY_BG, linewidth=2, zorder=5)

    ax.set_title("Drosophila Action Prediction Accuracy vs. Lead Time", 
                 fontsize=13, fontweight='bold', pad=16, color=INK_PRIMARY)
    ax.set_xlabel("Lookahead Horizon Prior to Physical Movement (ms)", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Predictive Classification Accuracy (%)", fontsize=10.5, labelpad=8)
    ax.set_ylim(88, 100)
    ax.grid(True)
    
    legend = ax.legend(frameon=True, facecolor=SURFACE_BG, edgecolor=STONE_BORDER, fontsize=9.5, loc='lower left')
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    horizon_path = os.path.join(output_dir, "horizon_accuracy_curve.png")
    plt.savefig(horizon_path)
    plt.close()
    print(f"📊 Saved {horizon_path}")

    # -------------------------------------------------------------
    # 2. Confusion Matrix (Warm Terracotta / Rose Colormap)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    cm = np.array(metrics["confusion_matrix"])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Custom editorial colormap: Ivory -> Warm Rose -> Dusty Rose -> Blush Dark
    editorial_cmap = mcolors.LinearSegmentedColormap.from_list(
        "editorial_rose",
        [IVORY_BG, "#EBDCD3", WARM_ROSE, DUSTY_ROSE, BLUSH_DARK, INK_PRIMARY]
    )

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap=editorial_cmap,
        xticklabels=ACTION_CLASSES,
        yticklabels=ACTION_CLASSES,
        cbar=True,
        ax=ax,
        linewidths=1.0,
        linecolor=STONE_GRID,
        annot_kws={'fontsize': 9, 'fontweight': '500'}
    )

    ax.set_title("Action Decoder Confusion Matrix (100 ms Pre-Motor Window)", 
                 fontsize=13, fontweight='bold', pad=16, color=INK_PRIMARY)
    ax.set_xlabel("Predicted Action State", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Ground Truth Motor State", fontsize=10.5, labelpad=8)
    plt.xticks(rotation=35, ha='right', fontsize=9, color=CHARCOAL)
    plt.yticks(rotation=0, fontsize=9, color=CHARCOAL)

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"📊 Saved {cm_path}")

    # -------------------------------------------------------------
    # 3. Descending Neuron Feature Importance
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    imp_dict = metrics["neuron_importance"]
    sorted_neurons = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_neurons]
    scores = [x[1] * 100 for x in sorted_neurons]

    # Colors: top command neurons highlighted with Blush Dark and Dusty Rose
    bar_colors = [BLUSH_DARK if 'MDN' in n or 'GF' in n or 'DNp' in n or 'DNg' in n else DUSTY_ROSE for n in names]
    bars = ax.barh(names[::-1], scores[::-1], color=bar_colors[::-1], height=0.62, edgecolor=STONE_BORDER, linewidth=0.8)

    ax.set_title("Descending Command Channel Predictive Weight (Feature Importance %)", 
                 fontsize=13, fontweight='bold', pad=16, color=INK_PRIMARY)
    ax.set_xlabel("Relative Gini Importance Score (%)", fontsize=10.5, labelpad=8)
    ax.grid(True, axis='x')

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.35, bar.get_y() + bar.get_height() / 2.0, f'{w:.1f}%', 
                va='center', fontsize=9, color=CHARCOAL, fontweight='500')

    plt.tight_layout()
    feat_path = os.path.join(output_dir, "neuron_importance.png")
    plt.savefig(feat_path)
    plt.close()
    print(f"📊 Saved {feat_path}")

    # -------------------------------------------------------------
    # 4. Neural Activity Stream & Action Timing Sample
    # -------------------------------------------------------------
    with open(web_data_path, "r") as f:
        web_data = json.load(f)

    stream_sample = web_data["stream"][:350]
    times = [s["t"] for s in stream_sample]
    actual_actions = [s["actual_action"] for s in stream_sample]
    pred_actions = [s["predicted_next_action"] for s in stream_sample]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=300, sharex=True, 
                                   gridspec_kw={'height_ratios': [2.4, 1]})

    tracked_indices = [0, 1, 2, 4, 5]  # DNp09, MDN, DNg02_L, aDN, GF
    line_colors = [SAGE_GREEN, BLUSH_DARK, DUSTY_ROSE, MUTED_ROSE, "#945338"]

    for idx, col in zip(tracked_indices, line_colors):
        rates = [s["rates"][idx] for s in stream_sample]
        ax1.plot(times, rates, label=NEURON_NAMES[idx], color=col, linewidth=1.6)

    ax1.set_title("Population Firing Rates Across Key Descending Channels (Hz)", 
                  fontsize=12, fontweight='bold', pad=12, color=INK_PRIMARY)
    ax1.set_ylabel("Spike Rate (Hz)", fontsize=10, labelpad=8)
    ax1.grid(True)
    leg = ax1.legend(loc='upper right', frameon=True, facecolor=SURFACE_BG, 
                     edgecolor=STONE_BORDER, fontsize=8.5, ncol=3)
    leg.get_frame().set_linewidth(0.8)

    ax2.plot(times, actual_actions, label='Current Motor Action', color=MUTED_TEXT, linewidth=2.0, alpha=0.7)
    ax2.plot(times, pred_actions, label='Decoded 100 ms Forecast', color=BLUSH_DARK, linestyle='--', linewidth=1.6)
    ax2.set_yticks(range(len(ACTION_CLASSES)))
    ax2.set_yticklabels(ACTION_CLASSES, fontsize=7.5, color=CHARCOAL)
    ax2.set_xlabel("Elapsed Time (seconds)", fontsize=10, labelpad=8)
    ax2.grid(True)
    leg2 = ax2.legend(loc='lower right', frameon=True, facecolor=SURFACE_BG, 
                      edgecolor=STONE_BORDER, fontsize=8.5)
    leg2.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    raster_path = os.path.join(output_dir, "neural_raster_preview.png")
    plt.savefig(raster_path)
    plt.close()
    print(f"📊 Saved {raster_path}")


if __name__ == "__main__":
    generate_all_figures()
