import os
import csv

import torch

from models.inst_fusion_model import _ce_huber_loss
from train import (
    _append_csv_row,
    _has_nonfinite_loss,
    _is_better_val_loss,
    _load_model_state,
    _reduce_model_learning_rates,
    _save_model_state,
    _save_monitor_epoch,
    _save_monitor_visuals,
)


def test_ce_huber_loss_uses_configurable_huber_weight():
    logits = torch.zeros(1, 313, 2, 2)
    pred_ab = torch.zeros(1, 2, 8, 8)
    gt_ab_soft = torch.zeros(1, 313, 2, 2)
    gt_ab_soft[:, 0] = 1.0
    gt_ab_hard = torch.zeros(1, 2, 2, dtype=torch.long)
    rebalance_w = torch.ones(313)
    pts_in_hull = torch.zeros(313, 2)
    pts_in_hull[0] = torch.tensor([30.0, 0.0])

    total, parts = _ce_huber_loss(
        logits, pred_ab, gt_ab_soft, gt_ab_hard, rebalance_w,
        ab_norm=110., pts_in_hull=pts_in_hull,
        huber_weight=3.0, return_components=True)

    assert torch.allclose(total, parts['ce'] + 3.0 * parts['huber'])
    assert torch.allclose(parts['huber_weighted'], 3.0 * parts['huber'])


def test_monitor_helpers_write_csv_and_images(tmp_path):
    csv_path = tmp_path / 'loss_history.csv'
    _append_csv_row(csv_path, {'step': 1, 'loss_G': 2.5})
    _append_csv_row(csv_path, {'step': 2, 'loss_G': 2.0})

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['step'] == '1'
    assert rows[1]['loss_G'] == '2.0'

    visuals = {
        'real_gray': torch.zeros(1, 3, 8, 8),
        'fake_rgb': torch.ones(1, 3, 8, 8),
        'fake_rgb_reg': torch.ones(1, 3, 8, 8) * 0.5,
        'real_rgb': torch.ones(1, 3, 8, 8) * 0.25,
    }
    _save_monitor_visuals(tmp_path, epoch=3, sample_id='sample_001',
                          visuals=visuals)

    out_dir = tmp_path / 'epoch_003'
    assert os.path.isfile(out_dir / 'sample_001_real_gray.png')
    assert os.path.isfile(out_dir / 'sample_001_fake_rgb.png')
    assert os.path.isfile(out_dir / 'sample_001_fake_rgb_reg.png')
    assert os.path.isfile(out_dir / 'sample_001_real_rgb.png')


def test_monitor_epoch_returns_psnr_and_ssim(tmp_path):
    class DummyModel:
        def set_input(self, data):
            self.data = data

        def forward(self):
            pass

        def get_current_visuals(self):
            return {
                'real_gray': torch.zeros(1, 3, 12, 12),
                'fake_rgb': self.data['fake_rgb'],
                'real_rgb': self.data['real_rgb'],
            }

    monitor_loader = [
        {
            'file_id': ['same'],
            'fake_rgb': torch.ones(1, 3, 12, 12) * 0.25,
            'real_rgb': torch.ones(1, 3, 12, 12) * 0.25,
        },
        {
            'file_id': ['close'],
            'fake_rgb': torch.ones(1, 3, 12, 12) * 0.4,
            'real_rgb': torch.ones(1, 3, 12, 12) * 0.5,
        },
    ]

    metrics = _save_monitor_epoch(DummyModel(), monitor_loader, tmp_path, 5)

    assert metrics['monitor_num'] == 2
    assert metrics['monitor_psnr'] > 20.0
    assert 0.0 <= metrics['monitor_ssim'] <= 1.0
    assert os.path.isfile(tmp_path / 'epoch_005' / 'same_fake_rgb.png')


def test_validation_loss_not_psnr_controls_best_checkpoint():
    assert _is_better_val_loss(1.9, None, min_delta=0.0)
    assert _is_better_val_loss(1.89, 1.9, min_delta=0.005)
    assert not _is_better_val_loss(1.898, 1.9, min_delta=0.005)
    assert not _is_better_val_loss(float('nan'), 1.9, min_delta=0.0)


def test_nan_recovery_helpers_detect_bad_loss_and_reduce_lr():
    param = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([param], lr=1.0)

    class DummyScheduler:
        base_lrs = [1.0]

    class DummyModel:
        optimizers = [optimizer]
        schedulers = [DummyScheduler()]

    assert _has_nonfinite_loss({'G': float('nan')})
    assert _has_nonfinite_loss({'G': float('inf')})
    assert not _has_nonfinite_loss({'G': 1.0, 'ce': 2.0})

    new_lrs = _reduce_model_learning_rates(DummyModel(), factor=0.1)

    assert new_lrs == [0.1]
    assert optimizer.param_groups[0]['lr'] == 0.1
    assert DummyModel.schedulers[0].base_lrs == [0.1]


def test_recovery_checkpoint_does_not_overwrite_latest(tmp_path):
    class DummyModel:
        model_names = ['G']
        device = torch.device('cpu')
        save_dir = os.fspath(tmp_path)

        def __init__(self):
            self.netG = torch.nn.Linear(1, 1)

    model = DummyModel()
    with torch.no_grad():
        model.netG.weight.fill_(2.0)

    _save_model_state(model, 'recovery_clean')
    with torch.no_grad():
        model.netG.weight.fill_(4.0)
    _load_model_state(model, 'recovery_clean')

    assert torch.allclose(model.netG.weight, torch.full_like(model.netG.weight, 2.0))
    assert os.path.isfile(tmp_path / 'recovery_clean_net_G.pth')
    assert not os.path.exists(tmp_path / 'latest_net_G.pth')
