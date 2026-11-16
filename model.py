import re

import torch
import torch.nn as nn

from dino3d.unetr_head import UNETRHead
from dino3d.vision_transformer import VisionTransformer3d, load_pretrained_backbone

MODEL_CONFIG = {
    "vit": {"img_size": 112, "patch_size": 16, "embed_dim": 1024, "depth": 24, "num_heads": 16,
            "mlp_ratio": 4, "init_values": 1e-5},
    "lora": {"r": 64, "alpha": 128},
    "image_size": 160,
    "num_classes": 11,
}


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        self.scaling = alpha / r

    def forward(self, x):
        result = self.base(x)
        lora = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype))) * self.scaling
        return (result + lora).to(result.dtype)


def build_model(dino_weights, checkpoint, device):
    cfg = MODEL_CONFIG
    backbone = VisionTransformer3d(**cfg["vit"])
    load_pretrained_backbone(backbone, dino_weights)
    for block in backbone.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, **cfg["lora"])

    model = UNETRHead(backbone, image_size=cfg["image_size"], num_classes=cfg["num_classes"])
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = {
        re.sub(r"^feature_model\.blocks\.\d+\.", "feature_model.blocks.", k.removeprefix("head.").replace(".default", "")): v
        for k, v in state_dict.items()
    }
    # the checkpoint only holds the LoRA and decoder weights, the rest comes from 3DINO
    unexpected = model.load_state_dict(state_dict, strict=False).unexpected_keys
    if unexpected:
        raise RuntimeError(f"Checkpoint does not match the model: {unexpected[:5]}")
    return model.to(device).eval()
