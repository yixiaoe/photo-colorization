import torch
from types import SimpleNamespace
from models.text_color_model import TextColorModel


def _make_opt(is_train=True):
    return SimpleNamespace(
        gpu_ids=[],
        isTrain=is_train,
        checkpoints_dir='./checkpoints',
        name='test_text_color',
        full_ckpt='',
        clip_arch='ViT-B-32',
        num_heads=4,
        rebalance_gamma=0.5,
        lr=1e-4,
        beta1=0.9,
        niter=1,
        niter_decay=0,
        lr_policy='lambda',
        ab_norm=110.,
        ab_max=110.,
        ab_quant=10.,
        l_norm=100.,
        l_cent=50.,
        epoch_count=0,
    )


def test_model_forward_backward():
    opt = _make_opt(is_train=True)
    model = TextColorModel()
    model.initialize(opt)
    model.train()

    data = {
        'rgb_img': torch.rand(2, 3, 64, 64),
        'caption': ['a red car', 'blue sky'],
    }
    model.set_input(data)
    model.optimize_parameters()
    losses = model.get_current_losses()
    assert 'G' in losses
    assert losses['G'] > 0
    assert not torch.isnan(torch.tensor(losses['G']))


def test_model_visuals():
    opt = _make_opt(is_train=True)
    model = TextColorModel()
    model.initialize(opt)
    model.train()

    data = {
        'rgb_img': torch.rand(1, 3, 64, 64),
        'caption': ['a green tree'],
    }
    model.set_input(data)
    model.forward()
    visuals = model.get_current_visuals()
    assert 'fake_rgb' in visuals
    assert visuals['fake_rgb'].shape == (1, 3, 64, 64)
