"""
Vision Transformer (ViT) architecture, implemented to match Section IV-C of the paper:

  1) Patch Embedding      (Eq. 7)   E = P @ W_e
  2) Positional Encoding  (Eq. 8-9) Z0 = E + Pos  (learnable position embeddings)
  3) Class Token          (Eq. 10)  Z0' = [x_cls ; Z0]
  4) Transformer Encoder  (Eq. 11-13) L=12 layers, h=12 heads, D=768, D_mlp=3072
  5) Classification Head  (Eq. 14)  softmax(W_c . LN(Z_L[0]) + b_c)

Input to the model is the pre-extracted patch matrix P in R^(N_p x (P^2*C)),
i.e. shape (batch, 64, 1875), produced by dataset.extract_patches.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="TeaViT")
class PatchEmbedding(layers.Layer):
    """Eq. 7: linear projection of flattened patches to hidden dim D."""

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.proj = layers.Dense(hidden_dim, name="patch_embedding_dense")

    def call(self, patches):
        return self.proj(patches)  # (batch, N_p, D)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


@register_keras_serializable(package="TeaViT")
class PositionalEncoding(layers.Layer):
    """Eq. 8-9: learnable positional embedding table, added to patch embeddings."""

    def __init__(self, num_patches, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.pos_embedding = self.add_weight(
            name="position_embedding",
            shape=(1, num_patches, hidden_dim),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, x):
        return x + self.pos_embedding

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "hidden_dim": self.hidden_dim})
        return config


@register_keras_serializable(package="TeaViT")
class ClassToken(layers.Layer):
    """Eq. 10: learnable class token, prepended to the patch sequence."""

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="class_token",
            shape=(1, 1, self.hidden_dim),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.hidden_dim])
        return tf.concat([cls_tokens, x], axis=1)  # (batch, N_p+1, D)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


@register_keras_serializable(package="TeaViT")
class TransformerEncoderLayer(layers.Layer):
    """
    Eq. 11-13: one pre-LN transformer encoder block.
      Z'_l = MSA(LN(Z_{l-1})) + Z_{l-1}
      Z_l  = MLP(LN(Z'_l)) + Z'_l ,   MLP(x) = W2 . GELU(W1 x + b1) + b2
    """

    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.msa = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_dim // num_heads, dropout=dropout_rate
        )
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(hidden_dim),
            layers.Dropout(dropout_rate),
        ])

    def call(self, z, training=False):
        z_norm = self.ln1(z)
        attn_out = self.msa(z_norm, z_norm, training=training)   # Attention(Q,K,V), Eq. 13
        z_prime = attn_out + z                                    # residual, Eq. 11

        z_prime_norm = self.ln2(z_prime)
        mlp_out = self.mlp(z_prime_norm, training=training)
        z_out = mlp_out + z_prime                                 # residual, Eq. 12
        return z_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


def build_vit(cfg) -> tf.keras.Model:
    """Assemble the full ViT model per Fig. 2 / Section IV-C."""
    patch_input = layers.Input(shape=(cfg.num_patches, cfg.patch_dim), name="patches")

    x = PatchEmbedding(cfg.hidden_dim)(patch_input)          # Eq. 7  -> (B, 64, 768)
    x = ClassToken(cfg.hidden_dim)(x)                         # Eq. 10 -> (B, 65, 768)
    x = PositionalEncoding(cfg.num_patches + 1, cfg.hidden_dim)(x)  # Eq. 8-9

    for i in range(cfg.num_layers):                           # L = 12 layers
        x = TransformerEncoderLayer(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.mlp_dim,
            dropout_rate=cfg.dropout_rate,
            name=f"encoder_layer_{i}",
        )(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)
    cls_output = x[:, 0]  # Z_L[0], the class token position   (Eq. 14)
    logits = layers.Dense(cfg.num_classes, name="classification_head")(cls_output)
    probs = layers.Activation("softmax", name="softmax")(logits)

    model = tf.keras.Model(inputs=patch_input, outputs=probs, name="TeaLeafViT")
    return model


if __name__ == "__main__":
    from config import Config
    cfg = Config()
    model = build_vit(cfg)
    model.summary()
