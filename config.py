"""
Configuration for the Vision Transformer tea-leaf-disease pipeline.
All values are taken directly from the paper:
  "Vision Transformer-Based Tea Leaf Disease Classification Using a Novel Dataset"
  (Alom, Roy, Hassan, Sifat — QPAIN 2025)
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ---- Data ----
    data_dir: str = "data"          # expects data/<class_name>/*.jpg  (ImageFolder style)
    class_names: List[str] = field(default_factory=lambda: [
        "Anthracnose", "Algal Leaf", "Bird Eye Spot", "Brown Blight",
        "Healthy", "Red Leaf Spot", "White Spot",
    ])
    image_size: int = 200            # H = W = 200          (Eq. 2)
    channels: int = 3
    patch_size: int = 25             # P = 25                (Eq. 3)
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42

    # ---- ViT architecture ----
    hidden_dim: int = 768            # D                     (Eq. 7)
    num_layers: int = 12             # L                     (Sec IV-C4)
    num_heads: int = 12              # h                     (Sec IV-C4)
    mlp_dim: int = 3072              # D_mlp                 (Sec IV-C4)
    dropout_rate: float = 0.1

    # ---- Training ----
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    grad_clip_norm: float = 1.0
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2   # N_p = 64   (Eq. 3)

    @property
    def patch_dim(self) -> int:
        return self.patch_size * self.patch_size * self.channels  # 1875  (Eq. 4)
