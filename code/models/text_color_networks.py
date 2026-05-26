"""
Phase 3 — 文本引导上色网络结构定义。

本模块包含两个核心组件：

1. TextCrossAttentionBlock
   - 将 CLIP 文本 token 特征通过 Cross-Attention 注入图像特征图
   - 图像特征作为 Query，文本 token 作为 Key/Value
   - 使用 zero-init output projection 确保训练初始阶段为恒等映射
   - 支持 key_padding_mask 屏蔽 CLIP 的 PAD token

2. ClipTextColorGenerator
   - 使用组合模式（composition）包装 Phase 2 的 InstanceGenerator
   - 冻结整个 backbone，仅在解码器的 conv8_3 和 conv9_3 后
     插入可训练的 TextCrossAttentionBlock
   - 训练时只更新 ~609K 参数（两个 cross-attention block）
   - 通过 load_backbone_weights() 加载 Phase 2 stage-full 预训练权重

设计灵感来源：L-CoDer (Language-based Colorization with Color-object
Decoupling Transformer)，使用 token-level cross-attention 实现
空间级颜色控制。

约束：不修改 models/networks.py，通过导入 InstanceGenerator 实例复用。
"""
import torch
import torch.nn as nn

from models.networks import InstanceGenerator


class TextCrossAttentionBlock(nn.Module):
    """
    Cross-Attention injecting CLIP text tokens into image feature maps.

    Zero-init output projection ensures identity at initialization.
    """

    def __init__(self, feat_dim, clip_dim=512, num_heads=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.norm = nn.LayerNorm(feat_dim)
        self.text_proj = nn.Linear(clip_dim, feat_dim)
        self.cross_attn = nn.MultiheadAttention(
            feat_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, feat, text_tokens, key_padding_mask=None):
        """
        Args:
            feat:             (N, C, H, W) image feature map
            text_tokens:      (N, L, 512) CLIP token features
            key_padding_mask: (N, L) bool, True = ignore
        Returns:
            (N, C, H, W) modulated feature map
        """
        N, C, H, W = feat.shape
        q = feat.flatten(2).permute(0, 2, 1)       # (N, HW, C)
        q_norm = self.norm(q)

        kv = self.text_proj(text_tokens)            # (N, L, C)
        attn_out, _ = self.cross_attn(
            q_norm, kv, kv, key_padding_mask=key_padding_mask)
        out = self.out_proj(attn_out)               # (N, HW, C)
        result = q + out                            # residual
        return result.permute(0, 2, 1).reshape(N, C, H, W)


class ClipTextColorGenerator(nn.Module):
    """
    Phase 3 generator: InstanceGenerator backbone + Cross-Attention at decoder.

    Composition pattern: wraps a frozen InstanceGenerator and injects
    TextCrossAttentionBlock after conv8_3 and conv9_3.
    """

    def __init__(self, num_heads=4):
        super().__init__()
        self.backbone = InstanceGenerator()
        self.xattn8 = TextCrossAttentionBlock(256, 512, num_heads)
        self.xattn9 = TextCrossAttentionBlock(128, 512, num_heads)

    def forward(self, x, text_tokens, key_padding_mask=None):
        """
        Args:
            x:                (N, 1, H, W) L channel
            text_tokens:      (N, 77, 512) CLIP token features
            key_padding_mask: (N, 77) bool
        Returns:
            out_class: (N, 313, H/4, W/4)
            out_reg:   (N, 2, H, W)
        """
        bb = self.backbone
        # encoder
        conv1_2 = bb.model1(x)
        conv2_2 = bb.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = bb.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = bb.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = bb.model5(conv4_3)
        conv6_3 = bb.model6(conv5_3)
        conv7_3 = bb.model7(conv6_3)
        # decoder with cross-attention injection
        conv8_up = bb.model8up(conv7_3) + bb.model3short8(conv3_3)
        conv8_3 = bb.model8(conv8_up)
        conv8_3 = self.xattn8(conv8_3, text_tokens, key_padding_mask)
        out_class = bb.model_class(conv8_3)
        conv9_up = bb.model9up(conv8_3) + bb.model2short9(conv2_2)
        conv9_3 = bb.model9(conv9_up)
        conv9_3 = self.xattn9(conv9_3, text_tokens, key_padding_mask)
        conv10_up = bb.model10up(conv9_3) + bb.model1short10(conv1_2)
        conv10_2 = bb.model10(conv10_up)
        out_reg = bb.model_out(conv10_2)
        return out_class, out_reg

    def load_backbone_weights(self, state_dict):
        self.backbone.load_state_dict(state_dict)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def get_trainable_params(self):
        return list(self.xattn8.parameters()) + list(self.xattn9.parameters())

