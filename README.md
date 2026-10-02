# Tea Leaf Disease Classification — Vision Transformer

Implementation of the pipeline described in *"Vision Transformer-Based Tea Leaf
Disease Classification Using a Novel Dataset"* (Alom, Roy, Hassan, Sifat —
QPAIN 2025), built from the paper's equations and reported hyperparameters.

## What's implemented

| Paper section | File | Detail |
|---|---|---|
| IV-A Dataset Prep | `dataset.py` | Stratified 80:10:10 split, fixed seed |
| IV-B Preprocessing | `dataset.py` | Resize to 200×200, normalize to [0,1], 25×25 non-overlapping patches via `patchify` (Eq. 2–5) |
| IV-C ViT Architecture | `model.py` | Patch embedding (D=768), learnable positional encoding, class token, 12-layer / 12-head pre-LN transformer encoder (MLP dim 3072), classification head (Eq. 7–14) |
| IV-D Training | `train.py` | Categorical cross-entropy, Adam (lr=1e-4), gradient clipping (1.0), 50 epochs, batch size 32, `MirroredStrategy` for multi-GPU, checkpointing + LR reduction on plateau |
| V Results | `evaluate.py` | Per-class precision/recall/F1 (Table II), confusion matrix (Fig. 4), training/test loss curve (Fig. 3) |

## Setup

```bash
pip install -r requirements.txt
```

## Expected data layout

Download the "Tea Sickness Dataset" (or your own equivalent) and arrange it as:

```
data/
  Anthracnose/*.jpg
  Algal Leaf/*.jpg
  Bird Eye Spot/*.jpg
  Brown Blight/*.jpg
  Healthy/*.jpg
  Red Leaf Spot/*.jpg
  White Spot/*.jpg
```

Class names must match `config.py:Config.class_names` exactly (edit if your
folder names differ).

## Train

```bash
python train.py --data_dir data/ --epochs 50 --batch_size 32
```

This will:
- split the dataset 80:10:10 with a fixed seed,
- extract 25×25 patches from each 200×200 image on the fly,
- train the ViT with Adam (lr=1e-4, grad-clip 1.0),
- save the best model (by validation loss) to `checkpoints/best_model.keras`,
- log per-epoch metrics to `outputs/training_log.csv`.

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best_model.keras --data_dir data/
```

This will print/save:
- `outputs/classification_report.txt` — precision/recall/F1 per class
- `outputs/confusion_matrix.png`
- `outputs/loss_curve.png`

## Notes on scale

The model as specified in the paper (D=768, L=12, h=12, MLP=3072) is a
ViT-Base-scale network (~86M parameters). On a dataset of only ~3,300 images
this is a large model relative to the data, so in practice you may want to:

- reduce `hidden_dim` / `num_layers` in `config.py`, or
- initialize from ImageNet-pretrained ViT weights and fine-tune, or
- rely more heavily on data augmentation (light augmentation is already
  included in `dataset.py`'s `TeaLeafSequence`).

The code faithfully follows the paper's stated architecture and training
recipe as written; these are optional practical adjustments if you find the
full-size model overfits on your data.
