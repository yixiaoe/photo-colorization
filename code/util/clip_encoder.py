"""
Phase 3 — CLIP 文本编码器封装模块。

本模块封装 OpenAI CLIP ViT-B/32 的文本 Transformer，用于将用户输入的
自然语言描述（如 "一辆红色的汽车在蓝天下面"）编码为 token 级别的语义特征。

与常规 CLIP 使用方式不同，这里不使用 [EOS] pooled 向量，而是提取
Transformer 每一层输出后的完整 token 序列 (N, 77, 512)，以便后续
Cross-Attention 模块能够对不同文本 token 进行空间级注意力对齐。

设计要点：
  - 模型完全冻结（requires_grad=False），不参与训练
  - 使用 open_clip_torch 库加载预训练权重
  - 返回 padding_mask 用于 Cross-Attention 中屏蔽无效 token

依赖：open_clip_torch >= 2.20.0
"""
import torch
import open_clip


class CLIPTextEncoder:
    """Extract token-level features from CLIP text transformer (frozen)."""

    def __init__(self, arch='ViT-B-32', pretrained='openai', device='cpu'):
        self.device = torch.device(device)
        model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained)
        model.eval()
        self.model = model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(arch)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts):
        """
        Args:
            texts: list of str, length N
        Returns:
            token_features: (N, 77, 512) float32
            padding_mask:   (N, 77) bool — True for PAD positions
        """
        tokens = self.tokenizer(texts).to(self.device)  # (N, 77)
        x = self.model.token_embedding(tokens)
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # (77, N, D)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # (N, 77, D)
        x = self.model.ln_final(x)
        padding_mask = (tokens == 0)
        return x.float(), padding_mask
