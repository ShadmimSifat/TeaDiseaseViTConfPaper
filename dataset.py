"""
Dataset preparation for tea leaf disease classification.

Implements:
  - 80:10:10 stratified train/val/test split with a fixed seed  (Sec IV-A, Eq. 1)
  - Resize to 200x200, normalize to [0,1]                       (Sec IV-B, Eq. 2)
  - Non-overlapping 25x25 patch extraction via `patchify`        (Sec IV-B, Eq. 3-5)

Expected directory layout (standard ImageFolder style):

    data/
      Anthracnose/*.jpg
      Algal Leaf/*.jpg
      Bird Eye Spot/*.jpg
      Brown Blight/*.jpg
      Healthy/*.jpg
      Red Leaf Spot/*.jpg
      White Spot/*.jpg
"""

import os
import glob
import numpy as np
import tensorflow as tf

try:
    from patchify import patchify
    _HAS_PATCHIFY = True
except ImportError:
    _HAS_PATCHIFY = False


def extract_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Split a normalized (H, W, C) image into non-overlapping flattened patches.
    Matches Eq. 3-5 of the paper: N_p = (H/P)*(W/P) patches, each flattened to P*P*C.

    Uses the `patchify` library if available (as in the paper), else falls back
    to an equivalent reshape-based implementation.
    """
    H, W, C = image.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        "Image dimensions must be divisible by patch_size"

    if _HAS_PATCHIFY:
        patches = patchify(image, (patch_size, patch_size, C), step=patch_size)
        # patches shape: (H/P, W/P, 1, P, P, C) -> flatten grid then each patch
        patches = patches.reshape(-1, patch_size, patch_size, C)
    else:
        n_h, n_w = H // patch_size, W // patch_size
        patches = image.reshape(n_h, patch_size, n_w, patch_size, C)
        patches = patches.transpose(0, 2, 1, 3, 4).reshape(-1, patch_size, patch_size, C)

    num_patches = patches.shape[0]
    patches = patches.reshape(num_patches, -1)  # (N_p, P*P*C)  -> (64, 1875)
    return patches.astype(np.float32)


def load_image(path: str, image_size: int) -> np.ndarray:
    """Load an image, resize to (image_size, image_size), normalize to [0,1]."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_size, image_size], method="bilinear")
    img = img.numpy().astype(np.float32) / 255.0  # Eq. 2
    return img


def index_dataset(data_dir: str, class_names, seed: int, train_split: float, val_split: float):
    """
    Build stratified (per-class) train/val/test file lists using a fixed seed,
    matching the paper's 80:10:10 split described in Eq. 1.
    """
    rng = np.random.RandomState(seed)
    train_files, val_files, test_files = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        files = sorted(
            glob.glob(os.path.join(class_dir, "*.jpg"))
            + glob.glob(os.path.join(class_dir, "*.jpeg"))
            + glob.glob(os.path.join(class_dir, "*.png"))
        )
        if not files:
            print(f"[warn] no images found for class '{class_name}' in {class_dir}")
            continue

        rng.shuffle(files)
        n = len(files)
        n_train = int(round(train_split * n))
        n_val = int(round(val_split * n))

        train_files += files[:n_train]
        val_files += files[n_train:n_train + n_val]
        test_files += files[n_train + n_val:]

        train_labels += [label_idx] * n_train
        val_labels += [label_idx] * (min(n_val, n - n_train))
        test_labels += [label_idx] * (n - n_train - min(n_val, n - n_train))

    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)


class TeaLeafSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence that yields (patch_batch, one_hot_label_batch).
    patch_batch shape: (batch, N_p, P*P*C)   e.g. (32, 64, 1875)
    """

    def __init__(self, files, labels, cfg, shuffle=True, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.files = files
        self.labels = labels
        self.cfg = cfg
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(files))
        self.on_epoch_end()

    def __len__(self):
        return max(1, int(np.ceil(len(self.files) / self.cfg.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _augment(self, img):
        if np.random.rand() < 0.5:
            img = np.fliplr(img)
        if np.random.rand() < 0.3:
            img = np.flipud(img)
        return img

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.cfg.batch_size:(idx + 1) * self.cfg.batch_size]
        batch_patches, batch_labels = [], []

        for i in batch_idx:
            img = load_image(self.files[i], self.cfg.image_size)
            if self.augment:
                img = self._augment(img)
            patches = extract_patches(img, self.cfg.patch_size)
            batch_patches.append(patches)
            batch_labels.append(self.labels[i])

        X = np.stack(batch_patches, axis=0)
        y = tf.keras.utils.to_categorical(batch_labels, num_classes=self.cfg.num_classes)
        return X, y


def build_datasets(cfg):
    """Return train/val/test Keras Sequences plus the raw file/label splits."""
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = index_dataset(
        cfg.data_dir, cfg.class_names, cfg.random_seed, cfg.train_split, cfg.val_split
    )

    print(f"Train: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}")

    train_seq = TeaLeafSequence(train_files, train_labels, cfg, shuffle=True, augment=True)
    val_seq = TeaLeafSequence(val_files, val_labels, cfg, shuffle=False, augment=False)
    test_seq = TeaLeafSequence(test_files, test_labels, cfg, shuffle=False, augment=False)

    return train_seq, val_seq, test_seq, (test_files, test_labels)
