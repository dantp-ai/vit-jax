from typing import Tuple
from typing import Union

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array


class PatchEmbedding(nn.Module):

    patch_size: Tuple[int, int]
    img_size: Tuple[int, int]
    n_features: int

    def setup(self):
        self.n_patches = (self.img_size[0] // self.patch_size[0]) * (
            self.img_size[1] // self.patch_size[1]
        )

        self.patch_embedding = nn.Conv(
            self.n_features,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="VALID",
        )

    @nn.compact
    def __call__(self, X: Array) -> Array:

        X = self.patch_embedding(X)

        batch_size, height, width, channels = X.shape
        X = jnp.reshape(X, (batch_size, height * width, channels))

        return X


class MLPBlock(nn.Module):

    mlp_n_hidden: int
    out_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, X: Array, *, deterministic: bool = False) -> Array:
        X = nn.Dense(self.mlp_n_hidden)(X)
        X = nn.gelu(X)
        X = nn.Dropout(self.dropout_rate, deterministic=deterministic)(X)
        X = nn.Dense(self.out_dim)(X)
        X = nn.Dropout(self.dropout_rate, deterministic=deterministic)(X)

        return X


class EncoderBlock(nn.Module):

    n_hidden: int
    mlp_n_hidden: int
    n_attn_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, X: Array, *, deterministic: bool = False) -> Array:

        X = X + nn.MultiHeadDotProductAttention(
            num_heads=self.n_attn_heads,
            qkv_features=self.n_hidden,
            deterministic=deterministic,
            dropout_rate=self.dropout_rate,
        )(
            inputs_q=nn.LayerNorm()(X),
            inputs_kv=nn.LayerNorm()(X),
        )

        out_mlp = MLPBlock(
            mlp_n_hidden=self.mlp_n_hidden,
            out_dim=self.n_hidden,
            dropout_rate=self.dropout_rate,
        )(nn.LayerNorm()(X), deterministic=deterministic)

        return X + out_mlp


class VisionTransformer(nn.Module):

    patch_size: Union[int, Tuple[int, int]]
    img_size: Union[int, Tuple[int, int]]
    n_hidden: int
    mlp_n_hidden: int
    n_attn_heads: int
    n_blocks: int
    n_classes: int
    emb_dropout_rate: float = 0.1
    dropout_rate: float = 0.1

    def setup(self):

        self.patch_emb = PatchEmbedding(
            patch_size=self.patch_size,
            img_size=self.img_size,
            n_features=self.n_hidden,
        )

        self.cls_token = self.param(
            "cls_token", nn.initializers.zeros, (1, 1, self.n_hidden)
        )

        n_steps = self.patch_emb.n_patches + 1

        self.position_embedding = self.param(
            "pos_embed", nn.initializers.normal(), (1, n_steps, self.n_hidden)
        )
        self.blocks = [
            EncoderBlock(
                n_hidden=self.n_hidden,
                mlp_n_hidden=self.mlp_n_hidden,
                n_attn_heads=self.n_attn_heads,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.n_blocks)
        ]
        self.head = nn.Sequential([nn.LayerNorm(), nn.Dense(self.n_classes)])

    @nn.compact
    def __call__(self, X: Array, deterministic: bool = False) -> Array:

        X = self.patch_emb(X)
        X = jnp.concatenate((jnp.tile(self.cls_token, (X.shape[0], 1, 1)), X), 1)
        X = nn.Dropout(self.emb_dropout_rate, deterministic=deterministic)(
            X + self.position_embedding
        )

        for block in self.blocks:
            X = block(X, deterministic=deterministic)

        return self.head(X[:, 0])
