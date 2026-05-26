import torch
from models.text_color_networks import TextCrossAttentionBlock


def test_xattn_output_shape():
    block = TextCrossAttentionBlock(feat_dim=256, clip_dim=512, num_heads=4)
    feat = torch.randn(2, 256, 8, 8)
    text = torch.randn(2, 77, 512)
    out = block(feat, text)
    assert out.shape == (2, 256, 8, 8)


def test_xattn_identity_at_init():
    block = TextCrossAttentionBlock(feat_dim=128, clip_dim=512, num_heads=4)
    feat = torch.randn(1, 128, 16, 16)
    text = torch.randn(1, 77, 512)
    out = block(feat, text)
    # zero-init out_proj means output ≈ input (residual only)
    assert torch.allclose(out, feat, atol=1e-5)


def test_xattn_with_padding_mask():
    block = TextCrossAttentionBlock(feat_dim=64, clip_dim=512, num_heads=4)
    feat = torch.randn(2, 64, 4, 4)
    text = torch.randn(2, 77, 512)
    mask = torch.zeros(2, 77, dtype=torch.bool)
    mask[:, 10:] = True  # only first 10 tokens are valid
    out = block(feat, text, key_padding_mask=mask)
    assert out.shape == (2, 64, 4, 4)


def test_xattn_gradient_flows():
    block = TextCrossAttentionBlock(feat_dim=128, clip_dim=512, num_heads=4)
    feat = torch.randn(1, 128, 4, 4, requires_grad=True)
    text = torch.randn(1, 77, 512)
    out = block(feat, text)
    loss = out.sum()
    loss.backward()
    assert block.text_proj.weight.grad is not None
    assert feat.grad is not None
