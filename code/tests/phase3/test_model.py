import torch
from types import SimpleNamespace
from models.text_color_model import TextColorModel
from util.util import load_zhang2016_ab_bins


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
        huber_weight=3.0,
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


def test_model_visuals_include_class_and_regression_outputs():
    opt = _make_opt(is_train=False)
    model = TextColorModel()
    model.opt = opt
    model.real_L = torch.rand(1, 1, 64, 64) * 2 - 1
    model.real_ab = torch.rand(1, 2, 64, 64) * 2 - 1
    model.pred_class = torch.randn(1, 313, 16, 16)
    model.pred_ab = torch.rand(1, 2, 64, 64) * 2 - 1
    model.pts_in_hull = torch.tensor(load_zhang2016_ab_bins()).float()

    visuals = model.get_current_visuals()

    assert visuals['fake_rgb'].shape == (1, 3, 64, 64)
    assert visuals['fake_rgb_reg'].shape == (1, 3, 64, 64)
