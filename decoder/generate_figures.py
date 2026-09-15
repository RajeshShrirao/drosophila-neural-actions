"""
Publication & Social Media Figure Generator
Generates high-resolution visualization plots for LinkedIn and GitHub documentation:
1. Lookahead Prediction Horizon Curve
2. Confusion Matrix of Decoded Actions
3. Descending Neuron Feature Importance
4. Multi-Channel Neural Raster & Motor Response Trace
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from decoder.connectome_model import ACTION_CLASSES, NEURON_NAMES


def set_dark_theme_style():
    """Sets a clean, modern aesthetic for figures."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.sans-serif': 'Helvetica',
        'font.family': 'sans-serif',
        'axes.edgecolor': '#444c56',
        'axes.linewidth': 1.2,
        'grid.color': '#2d333b',
        'grid.linestyle': '--',
        'grid.alpha': 0.6
    })


def generate_all_figures(
    metrics_path: str = "assets/models/benchmark_metrics.json",
    web_data_path: str = "web/data/test_neural_stream.json",
    output_dir: str = "assets/figures"
):
    os.makedirs(output_dir, exist_ok=True)
    set_dark_theme_style()

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # -------------------------------------------------------------
    # 1. Lookahead Horizon Curve (How early can we predict?)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    horizons = [item["horizon_ms"] for item in metrics["horizon_results"]]
    accuracies = [item["accuracy"] * 100 for item in metrics["horizon_results"]]

    ax.plot(horizons, accuracies, marker='o', markersize=8, color='#58a6ff', linewidth=2.8, label='Decoder Accuracy')
    ax.axvline(x=100, color='#f78166', linestyle='--', linewidth=1.5, label='Optimal Lead Window (100ms)')
    ax.scatter([100], [accuracies[horizons.index(100.0)]], color='#f78166', s=120, zorder=5)

    ax.set_title("Can We Predict Fly Action Before Motor Onset?", fontsize=14, fontweight='bold', pad=15, color='#f0f6fc')
    ax.set_xlabel("Lookahead Lead Time Before Movement (ms)", fontsize=11, color='#c9d1d9')
    ax.set_ylabel("Action Prediction Accuracy (%)", fontsize=11, color='#c9d1d9')
    ax.set_ylim(50, 100)
    ax.grid(True)
    ax.legend(frameon=True, facecolor='#161b22', edgecolor='#30363d', fontsize=10)

    plt.tight_layout()
    horizon_fig_path = os.path.join(output_dir, "horizon_accuracy_curve.png")
    plt.savefig(horizon_fig_path)
    plt.close()
    print(f"📊 Saved {horizon_fig_path}")

    # -------------------------------------------------------------
    # 2. Confusion Matrix
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    cm = np.array(metrics["confusion_matrix"])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=ACTION_CLASSES,
        yticklabels=ACTION_CLASSES,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor='#21262d'
    )

    ax.set_title("Neural Decoder Action Confusion Matrix (100ms Ahead)", fontsize=13, fontweight='bold', pad=15, color='#f0f6fc')
    ax.set_xlabel("Predicted Next Action", fontsize=11, color='#c9d1d9')
    ax.set_ylabel("True Executed Action", fontsize=11, color='#c9d1d9')
    plt.xticks(rotation=40, ha='right', fontsize=9, color='#c9d1d9')
    plt.yticks(rotation=0, fontsize=9, color='#c9d1d9')

    plt.tight_layout()
    cm_fig_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_fig_path)
    plt.close()
    print(f"📊 Saved {cm_fig_path}")

    # -------------------------------------------------------------
    # 3. Descending Neuron Feature Importance
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    imp_dict = metrics["neuron_importance"]
    sorted_neurons = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_neurons]
    scores = [x[1] * 100 for x in sorted_neurons]

    colors = ['#238636' if 'MDN' in n or 'GF' in n or 'DNp' in n or 'DNg' in n else '#1f6feb' for n in names]
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1], height=0.65)

    ax.set_title("Descending Neuron Predictive Power (Relative Importance %)", fontsize=13, fontweight='bold', pad=15, color='#f0f6fc')
    ax.set_xlabel("Feature Importance Score (%)", fontsize=11, color='#c9d1d9')
    ax.grid(True, axis='x')

    # Value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.4, bar.get_y() + bar.get_height() / 2.0, f'{w:.1f}%', va='center', fontsize=9, color='#8b949e')

    plt.tight_layout()
    feat_fig_path = os.path.join(output_dir, "neuron_importance.png")
    plt.savefig(feat_fig_path)
    plt.close()
    print(f"📊 Saved {feat_fig_path}")

    # -------------------------------------------------------------
    # 4. Neural Activity Stream & Action Timing Sample
    # -------------------------------------------------------------
    with open(web_data_path, "r") as f:
        web_data = json.load(f)

    stream_sample = web_data["stream"][:350]  # First ~7 seconds
    times = [s["t"] for s in stream_sample]
    actual_actions = [s["actual_action"] for s in stream_sample]
    pred_actions = [s["predicted_next_action"] for s in stream_sample]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=300, sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    
    # Plot top descending neurons (MDN, DNp09, GF, aDN, DNg02_L)
    tracked_indices = [0, 1, 2, 4, 5]  # DNp09, MDN, DNg02_L, aDN, GF
    colors_lines = ['#58a6ff', '#f78166', '#7ee787', '#d2a8ff', '#ffa657']
    
    for idx, col in zip(tracked_indices, colors_lines):
        rates = [s["rates"][idx] for s in stream_sample]
        ax1.plot(times, rates, label=NEURON_NAMES[idx], color=col, linewidth=1.8, alpha=0.9)

    ax1.set_title("Live Descending Command Population Firing Rates (Hz)", fontsize=12, fontweight='bold', color='#f0f6fc')
    ax1.set_ylabel("Rate (Hz)", fontsize=10, color='#c9d1d9')
    ax1.grid(True)
    ax1.legend(loc='upper right', frameon=True, facecolor='#161b22', edgecolor='#30363d', fontsize=8.5, ncol=3)

    # Plot actual vs predicted action
    ax2.plot(times, actual_actions, label='Current Motor Action', color='#8b949e', linewidth=2.5, alpha=0.7)
    ax2.plot(times, pred_actions, label='Decoder 100ms Prediction', color='#f78166', linestyle='--', linewidth=1.8)
    ax2.set_yticks(range(len(ACTION_CLASSES)))
    ax2.set_yticklabels(ACTION_CLASSES, fontsize=7.5, color='#c9d1d9')
    ax2.set_xlabel("Time (seconds)", fontsize=10, color='#c9d1d9')
    ax2.grid(True)
    ax2.legend(loc='lower right', frameon=True, facecolor='#161b22', edgecolor='#30363d', fontsize=8.5)

    plt.tight_layout()
    raster_fig_path = os.path.join(output_dir, "neural_raster_preview.png")
    plt.savefig(raster_fig_path)
    plt.close()
    print(f"📊 Saved {raster_fig_path}")


if __name__ == "__main__":
    generate_all_figures()
