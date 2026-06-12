"""
Phase 3 v2 pipeline that composes the frozen Phase 2 networks
(FiLMInstanceGenerator, FusionPipeline) with one text-conditioned
TextAdapter module on the instance branch only.

v2 changes vs v1:
- Removed bg_adapter entirely. Background follows Phase 2 baseline
  with no text conditioning, eliminating bg over-saturation drift
  and simplifying the contract.
- 5 injection points on instance branch (conv6_3, conv7_3, conv8_3,
  conv9_3, conv10_2) instead of 3.
- TextAdapter MLP: Linear -> GELU -> Linear (was: single Linear).
- ~2.5M trainable parameters (was ~1.05M).

Critical contract: when the adapter is zero-initialised, the output
must be bit-equal (up to numerical noise) to the Phase 2 fusion-stage
output. This is preserved by zero-init on the FINAL MLP layer only.

Reference: see FusionPipeline.forward in models/networks.py (~lines
570-637) which this class mirrors line by line. Bg path now identical
to FusionPipeline; only the instance feature maps fed into WGs are
text-modulated by inst_adapter.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .networks import FiLMInstanceGenerator, FusionPipeline


# Instance-side injection points. Spans encoder bottleneck (conv6_3/
# conv7_3, both 512ch) + decoder (conv8_3 256ch, conv9_3/conv10_2
# 128ch). Total adapter params ≈ 2.5M.
INSTANCE_LAYERS = {
    'conv6_3':  512,
    'conv7_3':  512,
    'conv8_3':  256,
    'conv9_3':  128,
    'conv10_2': 128,
}


class TextAdapter(nn.Module):
    """
    FiLM-style text-conditioned modulator with 2-layer MLP projection.

        feat'        =  feat * (1 + gamma) + beta
        [gamma, beta] = Linear(GELU(Linear(text_emb)))

    Final Linear is zero-init so gamma = beta = 0 at start of training,
    making an untrained pipeline bit-equal to the unmodulated baseline.
    """

    def __init__(self, clip_dim: int,
                 layer_channels: Dict[str, int],
                 hidden_dim: int = 512):
        super().__init__()
        self.clip_dim = clip_dim
        self.layer_channels = dict(layer_channels)
        self.proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * ch),
            )
            for name, ch in layer_channels.items()
        })
        # zero-init ONLY the final Linear of each MLP so the residual
        # path is identity at start; first Linear keeps its default init
        # so gradients flow once training starts.
        for mlp in self.proj.values():
            final = mlp[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, feat_dict, text_emb):
        out = dict(feat_dict)
        for name in feat_dict:
            if name not in self.proj:
                continue
            feat = feat_dict[name]
            gb = self.proj[name](text_emb)                # (B, 2C)
            gamma, beta = gb.chunk(2, dim=1)              # (B, C) each
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta  = beta.unsqueeze(-1).unsqueeze(-1)
            out[name] = feat * (1.0 + gamma) + beta
        return out


class TextColorPipeline(nn.Module):
    def __init__(self,
                 netInst: FiLMInstanceGenerator,
                 netFusion: FusionPipeline,
                 clip_dim: int = 512,
                 hidden_dim: int = 512):
        super().__init__()
        self.netInst = netInst
        self.netFusion = netFusion

        # freeze every Phase-2 parameter; keep BN in eval to avoid running-
        # stat drift during forward passes through frozen branches.
        for p in self.netInst.parameters():
            p.requires_grad_(False)
        for p in self.netFusion.parameters():
            p.requires_grad_(False)
        self.netInst.eval()
        self.netFusion.eval()

        # v2: instance adapter only — background uses Phase 2 baseline
        # without text conditioning. This eliminates the bg-drift /
        # over-saturation issues observed in v1.
        self.inst_adapter = TextAdapter(clip_dim, INSTANCE_LAYERS,
                                        hidden_dim=hidden_dim)

    # ── trainable param helpers ────────────────────────────────────────

    def get_trainable_params(self):
        return list(self.inst_adapter.parameters())

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())

    # ── train/eval override: keep frozen branches always in eval ───────

    def train(self, mode: bool = True):
        self.inst_adapter.train(mode)
        self.netInst.eval()
        self.netFusion.eval()
        return self

    # ── main forward ───────────────────────────────────────────────────

    def forward(self,
                gray_full: torch.Tensor,                 # (1, 1, H, W)
                inst_L: Optional[torch.Tensor],          # (N, 1, H, W) or None
                class_labels: Optional[torch.Tensor],    # (N,) long or None
                box_info_list: Optional[List[torch.Tensor]],  # [bi_H, bi_H2, bi_H4, bi_H8]
                inst_text_embs: Optional[torch.Tensor],  # (N, clip_dim) or None
                bg_text_emb: torch.Tensor,               # (1, clip_dim) — accepted for API compat but UNUSED in v2
                empty_box: bool = False):
        """
        Returns (out_class, out_reg) matching FusionPipeline.forward:
          out_class: (1, 313, H/4, W/4)
          out_reg:   (1, 2,   H,   W)

        Note: bg_text_emb is accepted for backward CLI compatibility but
        is NOT consumed in v2 — background colorization follows Phase 2
        baseline unconditionally.
        """
        del bg_text_emb  # explicitly unused in v2

        # ── 1. instance branch ────────────────────────────────────────
        has_inst = (not empty_box) and inst_L is not None and inst_L.shape[0] > 0
        if has_inst:
            with torch.no_grad():
                _, _, fm_batch = self.netInst(inst_L, class_labels)
            # adapter modulates the 5 injection points; other keys pass through
            fm_batch_text = self.inst_adapter(fm_batch, inst_text_embs)

            N = inst_L.shape[0]
            inst_feats: List[Dict[str, torch.Tensor]] = []
            for i in range(N):
                fmi = {k: fm_batch_text[k][[i]] for k in fm_batch_text}
                inst_feats.append(fmi)
        else:
            inst_feats = []

        # ── 2. background branch (NO bg adapter in v2) ────────────────
        return self._fusion(gray_full, inst_feats, box_info_list, has_inst)

    # ── 3. mirror of FusionPipeline.forward (no bg-adapter hooks) ─────

    def _fusion(self,
                gray_full,
                inst_feats,
                box_info_list,
                has_inst):
        """Reproduces FusionPipeline.forward (networks.py:570-637) verbatim.
        Instance modulation happens upstream in `inst_adapter` — by the time
        we get here the feature_map dicts in `inst_feats` are already
        text-conditioned."""
        F = self.netFusion   # shorthand

        if has_inst:
            bi0, bi1, bi2, bi3 = box_info_list

        def _stack(key):
            return torch.cat([f[key] for f in inst_feats], dim=0)

        def _fuse(wg, feat, key, bi):
            if has_inst:
                return wg(_stack(key), feat, bi)
            return feat

        # ── Encoder ──────────────────────────────────────────────────
        conv1_2 = F.model1(gray_full)
        conv1_2 = _fuse(F.wg_conv1_2,  conv1_2, 'conv1_2',  bi0 if has_inst else None)

        conv2_2 = F.model2(conv1_2[:, :, ::2, ::2])
        conv2_2 = _fuse(F.wg_conv2_2,  conv2_2, 'conv2_2',  bi1 if has_inst else None)

        conv3_3 = F.model3(conv2_2[:, :, ::2, ::2])
        conv3_3 = _fuse(F.wg_conv3_3,  conv3_3, 'conv3_3',  bi2 if has_inst else None)

        conv4_3 = F.model4(conv3_3[:, :, ::2, ::2])
        conv4_3 = _fuse(F.wg_conv4_3,  conv4_3, 'conv4_3',  bi3 if has_inst else None)

        conv5_3 = F.model5(conv4_3)
        conv5_3 = _fuse(F.wg_conv5_3,  conv5_3, 'conv5_3',  bi3 if has_inst else None)

        conv6_3 = F.model6(conv5_3)
        conv6_3 = _fuse(F.wg_conv6_3,  conv6_3, 'conv6_3',  bi3 if has_inst else None)

        conv7_3 = F.model7(conv6_3)
        conv7_3 = _fuse(F.wg_conv7_3,  conv7_3, 'conv7_3',  bi3 if has_inst else None)

        # ── Decoder ──────────────────────────────────────────────────
        conv8_up = F.model8up(conv7_3) + F.model3short8(conv3_3)
        conv8_up = _fuse(F.wg_conv8_up, conv8_up, 'conv8_up', bi2 if has_inst else None)

        conv8_3 = F.model8(conv8_up)
        conv8_3 = _fuse(F.wg_conv8_3,  conv8_3, 'conv8_3',  bi2 if has_inst else None)

        conv9_up = F.model9up(conv8_3) + F.model2short9(conv2_2)
        conv9_up = _fuse(F.wg_conv9_up, conv9_up, 'conv9_up', bi1 if has_inst else None)

        conv9_3 = F.model9(conv9_up)
        conv9_3 = _fuse(F.wg_conv9_3,  conv9_3, 'conv9_3',  bi1 if has_inst else None)

        conv10_up = F.model10up(conv9_3) + F.model1short10(conv1_2)
        conv10_up = _fuse(F.wg_conv10_up, conv10_up, 'conv10_up', bi0 if has_inst else None)

        conv10_2 = F.model10(conv10_up)
        conv10_2 = _fuse(F.wg_conv10_2, conv10_2, 'conv10_2', bi0 if has_inst else None)

        out_class = F.model_class(conv8_3)
        out_reg   = F.output_conv(conv10_2)
        return out_class, out_reg
