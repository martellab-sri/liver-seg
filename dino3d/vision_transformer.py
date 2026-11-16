import math
import re

import torch
import torch.nn as nn


class PatchEmbed3d(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = (patch_size,) * 3
        self.num_patches = (img_size // patch_size) ** 3
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim, init_values):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, init_values):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class VisionTransformer3d(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, init_values, in_chans=1):
        super().__init__()
        self.num_features = embed_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed3d(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, init_values) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, x, w, h, d):
        previous_dtype = x.dtype
        N = self.pos_embed.shape[1] - 1
        if x.shape[1] - 1 == N and w == h == d:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        # +0.1 avoids floating point errors in the interpolation scale factor
        w0, h0, d0 = w // self.patch_size + 0.1, h // self.patch_size + 0.1, d // self.patch_size + 0.1
        cbrt_N = math.pow(N, 1 / 3)
        n = round(cbrt_N)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, n, n, n, dim).permute(0, 4, 1, 2, 3),
            scale_factor=(w0 / cbrt_N, h0 / cbrt_N, d0 / cbrt_N),
            mode="trilinear",
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens(self, x):
        B, _, w, h, d = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        return x + self.interpolate_pos_encoding(x, w, h, d)

    def get_intermediate_layers(self, x, layers):
        """Returns normalized patch tokens after each block index in `layers`."""
        x = self.prepare_tokens(x)
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in layers:
                outputs.append(self.norm(x)[:, 1:])
        return outputs


def load_pretrained_backbone(model, path):
    state_dict = torch.load(path, map_location="cpu", mmap=True, weights_only=True)["teacher"]
    state_dict = {
        re.sub(r"^blocks\.\d+\.", "blocks.", k.removeprefix("backbone.")): v
        for k, v in state_dict.items()
        if k.startswith("backbone.") and k != "backbone.mask_token"
    }
    model.load_state_dict(state_dict)
