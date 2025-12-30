"""
Adapted from https://github.com/baofff/U-ViT/blob/main/libs/timm.py
"""

import math

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from genpp.models.cgm.fm.base import ConditionalVectorField
from genpp.models.cgm.fm.helpers import Mlp, trunc_normal_
from genpp.models.layers import FourierEncoder, PixelEmbedder

if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    ATTENTION_MODE = "flash"
else:
    try:
        import xformers
        import xformers.ops

        ATTENTION_MODE = "xformers"
    except ImportError:
        ATTENTION_MODE = "math"


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size: tuple[int, int]):
    x = einops.rearrange(
        imgs, "B C (h p1) (w p2) -> B (h w) (p1 p2 C)", p1=patch_size[0], p2=patch_size[1]
    )
    return x


def unpatchify(x, patch_size: tuple[int, int], image_size: tuple[int, int], channels: int = 2):
    h, w = image_size
    p1, p2 = patch_size
    h = h // p1
    w = w // p2
    assert h * w == x.shape[1] and p1 * p2 * channels == x.shape[2]
    x = einops.rearrange(
        x, "B (h w) (p1 p2 C) -> B C (h p1) (w p2)", h=h, w=w, p1=p1, p2=p2, C=channels
    )
    return x


class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == "flash":
            qkv = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, "B H L D -> B L (H D)")
        elif ATTENTION_MODE == "xformers":
            qkv = einops.rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)  # pyright: ignore[reportPossiblyUnboundVariable]
            x = einops.rearrange(x, "B L H D -> B L (H D)", H=self.num_heads)
        elif ATTENTION_MODE == "math":
            qkv = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplementedError

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        skip=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None

    def forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))  # pyright: ignore[reportCallIssue, reportArgumentType]
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size, in_channels=2, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0
        x = self.proj(x)
        x = einops.rearrange(x, "B C H W -> B (H W) C")
        return x


class UViTCVF(ConditionalVectorField):
    def __init__(
        self,
        img_height=40,
        img_width=40,
        patch_size_height=4,
        patch_size_width=4,
        in_channels=2,
        channels_conditioning=62,  # 62 features
        embed_dim=4 * 4 * 2,
        pixel_embed_dim=5,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        conv=True,
        skip=True,
        td_embedding_dim: int | None = None,
    ):
        super().__init__()
        print(f"attention mode is {ATTENTION_MODE}")
        self.in_channels = in_channels

        # Initialize timedelta embedder and adjust channels_conditioning
        self.td_embedding_dim = td_embedding_dim
        if td_embedding_dim is not None and td_embedding_dim > 0:
            self.td_embedder = FourierEncoder(dim=td_embedding_dim)
            channels_conditioning += td_embedding_dim
        elif td_embedding_dim == 0:
            self.td_embedder = Rearrange("... -> ... 1")
            channels_conditioning += 1
        else:
            self.td_embedder = None

        self.channels_conditioning = channels_conditioning + pixel_embed_dim
        self.image_size = (img_height, img_width)
        self.patch_size = (patch_size_height, patch_size_width)

        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = (img_height // patch_size_height) * (img_width // patch_size_width)

        self.time_embed = FourierEncoder(dim=embed_dim)

        self.pixel_embedder = PixelEmbedder(
            num_embeddings=img_height * img_width, embedding_dim=pixel_embed_dim
        )

        # images can be used as conditional tokens
        self.conditioning_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_channels=self.channels_conditioning,
            embed_dim=embed_dim,
        )

        # the conditioning image has the same shape as the input image -> same number of patches
        self.extras = 1 + num_patches

        # this tells the token at which position it is in the sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        # TODO add an embedding to indicate where the current patch is located in the image
        # this might be helpful for the x as well as for the conditioning
        # we also need to remove this from the x later on

        self.in_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                )
                for _ in range(depth // 2)
            ]
        )

        self.mid_block = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            norm_layer=norm_layer,
        )

        self.out_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                    skip=skip,
                )
                for _ in range(depth // 2)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size_height * patch_size_width * in_channels
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = (
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=3,
                padding=1,
            )
            if conv
            else nn.Identity()
        )

        trunc_normal_(tensor=self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(tensor=m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioning: dict[str, torch.Tensor]):
        x = self.patch_embed(x)
        B, L, D = x.shape  # L is number of patches
        # Get the time token to the shape [B], the solver only supplies a single number as t,
        # while in training this is a tensor of size [B]
        t = t.expand(B)
        time_token = self.time_embed(t)
        time_token = time_token.unsqueeze(dim=1)

        # Build conditioning with all components
        conditioning_parts = [
            conditioning["predicted_vars"],
            conditioning["auxiliary_vars"],
            conditioning["meta_vars"],
            self.pixel_embedder(conditioning["pixel_idx"]),
        ]

        # Add timedelta embedding to conditioning if configured
        if self.td_embedder is not None:
            td = conditioning["timedelta"]  # [bs]
            td_embed = self.td_embedder(td)  # [bs, 1 | td_embedding_dim]

            # Expand spatially to match conditioning dimensions
            *_, h, w = conditioning["predicted_vars"].shape
            td_embed = td_embed[..., None, None].expand(-1, -1, h, w)  # [bs, td_dim, h, w]
            conditioning_parts.append(td_embed)

        conditioning_cat = torch.cat(conditioning_parts, dim=1)
        conditioning_token = self.conditioning_embed(conditioning_cat)
        x = torch.cat((time_token, conditioning_token, x), dim=1)
        x = x + self.pos_embed

        skips = []  # skip connections
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras :, :]  # throw away the extra tokens (time + conditioning)
        x = unpatchify(
            x, patch_size=self.patch_size, image_size=self.image_size, channels=self.in_channels
        )
        x = self.final_layer(x)
        return x
