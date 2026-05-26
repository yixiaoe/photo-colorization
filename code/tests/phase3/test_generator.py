import torch
from models.text_color_networks import ClipTextColorGenerator


def test_generator_output_shape():
    gen = ClipTextColorGenerator(num_heads=4)
    x = torch.randn(2, 1, 256, 256)
    text = torch.randn(2, 77, 512)
    mask = torch.zeros(2, 77, dtype=torch.bool)
    out_class, out_reg = gen(x, text, mask)
    assert out_class.shape == (2, 313, 64, 64)
    assert out_reg.shape == (2, 2, 256, 256)


def test_generator_freeze_backbone():
    gen = ClipTextColorGenerator()
    gen.freeze_backbone()
    # backbone params frozen
    for p in gen.backbone.parameters():
        assert not p.requires_grad
    # xattn params trainable
    for p in gen.xattn8.parameters():
        assert p.requires_grad
    for p in gen.xattn9.parameters():
        assert p.requires_grad


def test_generator_trainable_param_count():
    gen = ClipTextColorGenerator()
    gen.freeze_backbone()
    trainable = gen.get_trainable_params()
    total = sum(p.numel() for p in trainable)
    # xattn8 (256-dim) + xattn9 (128-dim) ≈ 600K
    assert 400_000 < total < 1_500_000


def test_generator_identity_at_init():
    """With zero-init xattn, output should match plain InstanceGenerator."""
    from models.networks import InstanceGenerator
    gen = ClipTextColorGenerator()
    ref = gen.backbone
    ref.eval()
    gen.eval()
    x = torch.randn(1, 1, 64, 64)
    text = torch.randn(1, 77, 512)
    mask = torch.zeros(1, 77, dtype=torch.bool)
    with torch.no_grad():
        out_class, out_reg = gen(x, text, mask)
        ref_class, ref_reg, _ = ref(x)
    assert torch.allclose(out_class, ref_class, atol=1e-5)
    assert torch.allclose(out_reg, ref_reg, atol=1e-5)
