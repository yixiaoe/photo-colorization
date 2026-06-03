"""
Phase 3 pipeline that composes the frozen Phase 2 networks
(FiLMInstanceGenerator, FusionPipeline) with two text-conditioned
TextAdapter modules (instance + background).

Critical contract: when both adapters are zero-initialised, the output
must be bit-equal (up to numerical noise) to the Phase 2 fusion-stage
output. The bg_adapter only modulates `conv8_3`, `conv9_3`, `conv10_2`
**after** the corresponding `model_*` block has run, i.e. at the exact
points consumed by the WeightGenerator + heads in FusionPipeline.

Reference: see FusionPipeline.forward in models/networks.py (~lines
570-637) which this class mirrors line by line.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .networks import FiLMInstanceGenerator, FusionPipeline


# Decoder layers where the adapters inject. These match the keys in the
# feature_map dict produced by InstanceGenerator/FiLMInstanceGenerator
# (see networks.py:252-259).
DECODER_LAYERS = {
    'conv8_3':  256,
    'conv9_3':  128,
    'conv10_2': 128,
}


class TextAdapter(nn.Module):
    """
    FiLM-style text-conditioned modulator over a multi-scale feature dict.

        feat'        =  feat * (1 + gamma) + beta
        [gamma, beta] = Linear(text_emb).chunk(2, dim=1)

    Linear layers are zero-init so gamma = beta = 0 at the start of training,
    making an untrained pipeline bit-equal to the unmodulated baseline.
    """

    def __init__(self, clip_dim: int, layer_channels: Dict[str, int]):
        super().__init__()
        self.clip_dim = clip_dim
        self.layer_channels = dict(layer_channels)
        self.proj = nn.ModuleDict({
            name: nn.Linear(clip_dim, 2 * ch)
            for name, ch in layer_channels.items()
        })
        for m in self.proj.values():
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, feat_dict, text_emb):
        out = dict(feat_dict)
        # only modulate keys present in feat_dict – callers may pass a
        # single-layer dict (bg hook) or the full instance feature map
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
                 clip_dim: int = 512):
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

        self.inst_adapter = TextAdapter(clip_dim, DECODER_LAYERS)
        self.bg_adapter   = TextAdapter(clip_dim, DECODER_LAYERS)

    # ── trainable param helpers ────────────────────────────────────────

    def get_trainable_params(self):
        return list(self.inst_adapter.parameters()) + list(self.bg_adapter.parameters())

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())

    # ── train/eval override: keep frozen branches always in eval ───────

    def train(self, mode: bool = True):
        # adapters follow the requested mode
        self.inst_adapter.train(mode)
        self.bg_adapter.train(mode)
        # frozen branches stay in eval to lock BN stats and dropout
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
                bg_text_emb: torch.Tensor,               # (1, clip_dim)
                empty_box: bool = False):
        """
        Returns (out_class, out_reg) matching FusionPipeline.forward:
          out_class: (1, 313, H/4, W/4)
          out_reg:   (1, 2,   H,   W)
        """
        # ── 1. instance branch ────────────────────────────────────────
        has_inst = (not empty_box) and inst_L is not None and inst_L.shape[0] > 0
        if has_inst:
            with torch.no_grad():
                # frozen Phase-2 forward; AMP autocast is taken from caller context
                _, _, fm_batch = self.netInst(inst_L, class_labels)
            # adapter is the *only* trainable bit here, so gradients flow back
            # through fm_batch_text via the adapter's Linear weights.
            fm_batch_text = self.inst_adapter(fm_batch, inst_text_embs)

            N = inst_L.shape[0]
            inst_feats: List[Dict[str, torch.Tensor]] = []
            for i in range(N):
                fmi = {k: fm_batch_text[k][[i]] for k in fm_batch_text}
                inst_feats.append(fmi)
        else:
            inst_feats = []

        # ── 2. background branch with bg adapter ──────────────────────
        return self._fusion_with_bg_adapter(
            gray_full, inst_feats, box_info_list, bg_text_emb, has_inst,
        )

    # ── 3. mirror of FusionPipeline.forward with bg-adapter hooks ─────

    def _fusion_with_bg_adapter(self,
                                gray_full,
                                inst_feats,
                                box_info_list,
                                bg_text_emb,
                                has_inst):
        """Reproduces FusionPipeline.forward (networks.py:570-637) and inserts
        bg_adapter modulation at conv8_3 / conv9_3 / conv10_2."""
        F = self.netFusion   # shorthand

        if has_inst:
            bi0, bi1, bi2, bi3 = box_info_list

        def _stack(key):
            return torch.cat([f[key] for f in inst_feats], dim=0)

        def _fuse(wg, feat, key, bi):
            if has_inst:
                return wg(_stack(key), feat, bi)
            return feat

        def _bg_modulate(name, feat):
            """Apply bg_adapter only at the three decoder hook points."""
            if name not in DECODER_LAYERS:
                return feat
            return self.bg_adapter({name: feat}, bg_text_emb)[name]

        # ── Encoder (no bg-adapter hooks; identical to FusionPipeline) ──
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
        conv8_3 = _bg_modulate('conv8_3', conv8_3)
        conv8_3 = _fuse(F.wg_conv8_3,  conv8_3, 'conv8_3',  bi2 if has_inst else None)

        conv9_up = F.model9up(conv8_3) + F.model2short9(conv2_2)
        conv9_up = _fuse(F.wg_conv9_up, conv9_up, 'conv9_up', bi1 if has_inst else None)

        conv9_3 = F.model9(conv9_up)
        conv9_3 = _bg_modulate('conv9_3', conv9_3)
        conv9_3 = _fuse(F.wg_conv9_3,  conv9_3, 'conv9_3',  bi1 if has_inst else None)

        conv10_up = F.model10up(conv9_3) + F.model1short10(conv1_2)
        conv10_up = _fuse(F.wg_conv10_up, conv10_up, 'conv10_up', bi0 if has_inst else None)

        conv10_2 = F.model10(conv10_up)
        conv10_2 = _bg_modulate('conv10_2', conv10_2)
        conv10_2 = _fuse(F.wg_conv10_2, conv10_2, 'conv10_2', bi0 if has_inst else None)

        out_class = F.model_class(conv8_3)
        out_reg   = F.output_conv(conv10_2)
        return out_class, out_reg
