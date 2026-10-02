"""Online-encoder inference, matching the manuscript's deployment definition."""
from pathlib import Path
import json

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from .models.lgwm import Encoder, ActionEncoder, Predictor


class Verifier(nn.Module):
    """Runtime graph only: no EMA encoder, inverse head, or optimizer."""

    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config["encoder"], False, tuple(config["encoder_hw"]))
        self.action_encoder = ActionEncoder(self.encoder.dim, coord_posenc="shared")
        self.predictor = Predictor(self.encoder.dim, config["pred_depth"], config["pred_heads"])

    @classmethod
    def from_pretrained(cls, directory, device="cpu"):
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        if config["format"] != "lgwm-inference-v1" or config["observed_encoder"] != "online":
            raise ValueError("Expected an online-encoder LGWM inference export")
        model = cls(config["model"])
        model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        return model.to(device).eval()

    @torch.inference_mode()
    def forward(self, before, after, action):
        """Images: uint8 RGB [B,3,H,W]; action: canonical Batch-like attributes."""
        for value in (before, after):
            if value.ndim != 4 or value.shape[1] != 3 or value.dtype != torch.uint8:
                raise ValueError("Images must be uint8 RGB [B,3,H,W]")
        if before.shape[0] != after.shape[0]:
            raise ValueError("Before/after batch sizes differ")
        z = self.encoder(before)
        a, pad = self.action_encoder(action)
        predicted = self.predictor(z, a, pad)
        observed = self.encoder(after)
        p = F.normalize(predicted.mean(1).float(), dim=-1, eps=1e-6)
        o = F.normalize(observed.mean(1).float(), dim=-1, eps=1e-6)
        valid = (~pad).unsqueeze(-1).float()
        pooled_action = (a * valid).sum(1) / valid.sum(1).clamp_min(1)
        return {
            "predicted_tokens": predicted,
            "integrity": 1 - (p * o).sum(-1),
            "residual_features": torch.cat([o - p, pooled_action], dim=-1),
        }
