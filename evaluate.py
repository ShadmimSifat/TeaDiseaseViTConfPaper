"""
Evaluation script — Section V of the paper.

Produces:
  - Table II equivalent: per-class precision/recall/F1 + accuracy/macro/weighted avg
  - Fig. 3 equivalent: training vs. test/val loss curve
  - Fig. 4 equivalent: confusion matrix heatmap

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.keras --data_dir data/
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from config import Config
from dataset import build_datasets
import model as _model_module  # noqa: F401  (import registers custom ViT layers for deserialization)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.keras")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--training_log", type=str, default="outputs/training_log.csv")
    return p.parse_args()


def plot_loss_curve(csv_path, out_path):
    if not os.path.exists(csv_path):
        print(f"[warn] training log not found at {csv_path}, skipping loss curve")
        return
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["loss"], label="Training Loss", color="tab:blue")
    if "val_loss" in df.columns:
        plt.plot(df["epoch"], df["val_loss"], label="Validation/Test Loss", color="tab:orange")
    plt.title("Training and Test Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved loss curve to {out_path}")


def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(8, 7))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                      ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


def main():
    args = parse_args()
    cfg = Config()
    if args.data_dir:
        cfg.data_dir = args.data_dir

    os.makedirs(cfg.output_dir, exist_ok=True)

    _, _, test_seq, (test_files, test_labels) = build_datasets(cfg)

    print(f"Loading model from {args.checkpoint}")
    model = tf.keras.models.load_model(args.checkpoint)

    y_true, y_pred = [], []
    for i in range(len(test_seq)):
        X_batch, y_batch = test_seq[i]
        preds = model.predict(X_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Table II equivalent
    report = classification_report(
        y_true, y_pred, target_names=cfg.class_names, digits=2, zero_division=0
    )
    print("\n=== Classification Performance on Test Set ===")
    print(report)
    with open(os.path.join(cfg.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Fig. 4 equivalent
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, cfg.class_names, os.path.join(cfg.output_dir, "confusion_matrix.png"))

    # Fig. 3 equivalent
    plot_loss_curve(args.training_log, os.path.join(cfg.output_dir, "loss_curve.png"))

    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
