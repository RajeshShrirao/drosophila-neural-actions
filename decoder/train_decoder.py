"""
Model Training & Evaluation Pipeline for Neural Action Decoding
Trains Baseline Logistic Regression, Random Forest, and PyTorch Neural Decoders.
Evaluates prediction accuracy across lookahead horizons (0ms to 200ms before motor execution).
"""

import json
import os
import joblib
import numpy as np
from typing import Dict, List, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

from decoder.connectome_model import ACTION_CLASSES, NEURON_NAMES
from decoder.dataset_generator import FlyNeuralDataGenerator, create_lookahead_dataset


class PyTorchNeuralDecoder(nn.Module):
    """Deep neural network decoder with batch normalization and dropout."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_and_evaluate(
    output_dir: str = "assets/models",
    save_web_data_path: str = "web/data/test_neural_stream.json"
) -> Dict[str, Any]:
    """Runs the complete training, horizon benchmarking, and evaluation suite."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_web_data_path), exist_ok=True)

    print("🧠 [1/5] Generating biologically calibrated Drosophila neural sessions...")
    generator = FlyNeuralDataGenerator(dt_sec=0.01, random_seed=42)
    # Generate 5 minutes of rich behavioral data
    train_session = generator.generate_session(duration_sec=300.0, lead_time_ms=100.0)
    # Generate 90 seconds of holdout test data
    test_generator = FlyNeuralDataGenerator(dt_sec=0.01, random_seed=999)
    test_session = test_generator.generate_session(duration_sec=90.0, lead_time_ms=100.0)

    # 2. Benchmark Lookahead Horizon (Can we predict at t+0ms, t+50ms, t+100ms, t+150ms, t+200ms?)
    print("⏱️ [2/5] Evaluating Predictive Lookahead Horizon (0ms - 200ms before motor onset)...")
    horizons_ms = [0.0, 30.0, 60.0, 100.0, 150.0, 200.0]
    horizon_results = []

    for h_ms in horizons_ms:
        X_train_h, y_train_h = create_lookahead_dataset(train_session, lookahead_ms=h_ms)
        X_test_h, y_test_h = create_lookahead_dataset(test_session, lookahead_ms=h_ms)
        
        clf_h = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
        clf_h.fit(X_train_h, y_train_h)
        y_pred_h = clf_h.predict(X_test_h)
        acc = float(accuracy_score(y_test_h, y_pred_h))
        
        horizon_results.append({
            "horizon_ms": h_ms,
            "accuracy": round(acc, 4),
            "lead_time_ratio": round(h_ms / 100.0, 2)
        })
        print(f"   -> Horizon {int(h_ms):3d} ms before action: Prediction Accuracy = {acc * 100:.2f}%")

    # 3. Primary 100ms Lookahead Models (The core real-time decoder)
    print("\n🎯 [3/5] Training Core Decoders for 100ms Lookahead Horizon...")
    X_train, y_train = create_lookahead_dataset(train_session, lookahead_ms=100.0)
    X_test, y_test = create_lookahead_dataset(test_session, lookahead_ms=100.0)

    # Baseline 1: Logistic Regression
    print("   -> Training Regularized Logistic Regression...")
    lr_model = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_acc = float(accuracy_score(y_test, lr_preds))

    # Model 2: Random Forest
    print("   -> Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=14, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = float(accuracy_score(y_test, rf_preds))

    # Model 3: PyTorch Deep Neural Decoder
    print("   -> Training PyTorch Deep Neural Decoder...")
    device = torch.device("cpu")
    input_dim = X_train.shape[1]
    num_classes = len(ACTION_CLASSES)
    pt_model = PyTorchNeuralDecoder(input_dim, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(pt_model.parameters(), lr=0.003, weight_decay=1e-4)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    pt_model.train()
    for epoch in range(25):
        for bx, by in loader:
            optimizer.zero_grad()
            out = pt_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    pt_model.eval()
    with torch.no_grad():
        test_logits = pt_model(X_test_t)
        pt_preds = torch.argmax(test_logits, dim=1).numpy()
        pt_acc = float(accuracy_score(y_test, pt_preds))

    print(f"\n📊 Benchmark Comparison (100ms Pre-Action Prediction):")
    print(f"   • Logistic Regression : {lr_acc * 100:.2f}%")
    print(f"   • Random Forest       : {rf_acc * 100:.2f}%")
    print(f"   • Deep Neural Decoder : {pt_acc * 100:.2f}%")

    # 4. Feature Importance & Confusion Matrix
    print("\n🔍 [4/5] Computing Feature Importance & Confusion Matrix...")
    rf_cm = confusion_matrix(y_test, rf_preds).tolist()
    cls_report = classification_report(y_test, rf_preds, target_names=ACTION_CLASSES, output_dict=True)

    # Aggregate importance by neuron (mean, max, current)
    n_neurons = len(NEURON_NAMES)
    importances = rf_model.feature_importances_
    neuron_importance = {}
    for i, name in enumerate(NEURON_NAMES):
        neuron_importance[name] = float(
            importances[i] + importances[i + n_neurons] + importances[i + 2 * n_neurons]
        ) / 3.0

    # Save trained RF model
    model_path = os.path.join(output_dir, "rf_action_decoder.joblib")
    joblib.dump(rf_model, model_path)
    print(f"   -> Model saved to {model_path}")

    # 5. Export Web Demonstration Stream
    print("\n🌐 [5/5] Exporting simulation stream for Interactive Web Visualizer...")
    # Export 45 seconds of holdout stream (4500 time points, sampled at 20 Hz for smooth web rendering)
    subsample_step = 2  # 50 Hz effective for web streaming
    stream_len = min(4500, len(test_session["time_sec"]))
    
    # Calculate live 100ms future predictions along the stream
    lookahead_steps = 10
    window_steps = 15
    web_stream_data = []
    
    for t in range(window_steps, stream_len - lookahead_steps, subsample_step):
        window = test_session["firing_rates"][t - window_steps : t]
        feat = np.concatenate([np.mean(window, axis=0), np.max(window, axis=0), test_session["firing_rates"][t - 1]])
        probs = rf_model.predict_proba([feat])[0].tolist()
        pred_action = int(np.argmax(probs))
        
        web_stream_data.append({
            "t": round(float(test_session["time_sec"][t]), 3),
            "rates": [round(float(v), 2) for v in test_session["firing_rates"][t]],
            "actual_action": int(test_session["motor_states"][t]),
            "predicted_next_action": pred_action,
            "next_actual_action": int(test_session["motor_states"][t + lookahead_steps]),
            "probabilities": [round(p, 3) for p in probs]
        })

    web_payload = {
        "metadata": {
            "title": "Drosophila Neural Action Decoder - Live Stream",
            "dt_sec": round(test_session["dt_sec"] * subsample_step, 3),
            "lookahead_ms": 100.0,
            "actions": ACTION_CLASSES,
            "neurons": NEURON_NAMES,
            "benchmark": {
                "rf_accuracy": round(rf_acc, 4),
                "pt_accuracy": round(pt_acc, 4),
                "lr_accuracy": round(lr_acc, 4),
                "horizon_curve": horizon_results,
                "neuron_importance": neuron_importance,
                "confusion_matrix": rf_cm
            }
        },
        "stream": web_stream_data
    }

    with open(save_web_data_path, "w") as f:
        json.dump(web_payload, f, indent=2)
    print(f"   -> Web stream exported to {save_web_data_path} ({len(web_stream_data)} frames)")

    benchmark_summary = {
        "accuracies": {
            "logistic_regression": lr_acc,
            "random_forest": rf_acc,
            "deep_neural_decoder": pt_acc
        },
        "horizon_results": horizon_results,
        "neuron_importance": neuron_importance,
        "classification_report": cls_report,
        "confusion_matrix": rf_cm
    }

    metrics_path = os.path.join(output_dir, "benchmark_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(benchmark_summary, f, indent=2)
    print(f"   -> Benchmark metrics saved to {metrics_path}")

    return benchmark_summary


if __name__ == "__main__":
    train_and_evaluate()
