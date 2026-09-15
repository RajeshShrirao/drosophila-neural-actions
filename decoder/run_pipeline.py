"""
Master Orchestration Script for Drosophila Neural Action Decoding Pipeline.
Executes training, evaluation, horizon benchmarking, figure generation, and web data export.
"""

import sys
import time
from decoder.train_decoder import train_and_evaluate
from decoder.generate_figures import generate_all_figures


def main():
    start_time = time.time()
    print("=" * 70)
    print("🪰🧠 DROSOPHILA NEURAL ACTION PREDICTION PIPELINE")
    print("Janelia Drosophila Male CNS Connectomics & Descending Command Decoding")
    print("=" * 70)

    # 1. Train and evaluate
    metrics = train_and_evaluate(
        output_dir="assets/models",
        save_web_data_path="web/data/test_neural_stream.json"
    )

    # 2. Generate publication and social media figures
    print("\n🎨 Generating Figures for LinkedIn Post & Documentation...")
    generate_all_figures(
        metrics_path="assets/models/benchmark_metrics.json",
        web_data_path="web/data/test_neural_stream.json",
        output_dir="assets/figures"
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✅ Pipeline Completed Successfully in {elapsed:.1f}s!")
    print(f"   • Best Random Forest 100ms Accuracy : {metrics['accuracies']['random_forest']*100:.2f}%")
    print(f"   • Deep Neural Decoder 100ms Accuracy : {metrics['accuracies']['deep_neural_decoder']*100:.2f}%")
    print("   • Figures generated in assets/figures/")
    print("   • Web simulation data saved to web/data/test_neural_stream.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
