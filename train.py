"""
Training script — Section IV-D of the paper.

  - Loss: categorical cross-entropy                       (Eq. 15)
  - Optimizer: Adam, lr = 1e-4, gradient clipping at 1.0
  - Epochs: 50, batch size: 32
  - Multi-GPU: tf.distribute.MirroredStrategy
  - Callbacks: ModelCheckpoint (best val_loss), ReduceLROnPlateau, CSVLogger

Usage:
    python train.py --data_dir data/ --epochs 50
"""

import os
import argparse
import tensorflow as tf

from config import Config
from dataset import build_datasets
from model import build_vit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    train_seq, val_seq, test_seq, _ = build_datasets(cfg)

    # Multi-GPU support via MirroredStrategy, as described in the paper
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices for MirroredStrategy: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        model = build_vit(cfg)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.learning_rate,
            clipnorm=cfg.grad_clip_norm,   # gradient clipping at 1.0
        )
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",   # Eq. 15
            metrics=["accuracy"],
        )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(cfg.checkpoint_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(cfg.output_dir, "training_log.csv")
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
    ]

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(os.path.join(cfg.checkpoint_dir, "final_model.keras"))
    print("Training complete. Best model saved to", cfg.checkpoint_dir)
    return history, model, test_seq, cfg


if __name__ == "__main__":
    main()
