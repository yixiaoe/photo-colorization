import torch
import argparse

from options.phase3_options import Phase3TrainOptions
from scripts.eval_phase3_prompt_control import (
    compute_prompt_summary,
    edge_aware_total_variation,
)


def test_compute_prompt_summary_prefers_rank_then_locality_then_psnr():
    rows = [
        {
            "checkpoint": "a",
            "rank_success": 1,
            "neg_minus_pos": 0.10,
            "outside_delta": 0.03,
            "psnr": 20.0,
        },
        {
            "checkpoint": "a",
            "rank_success": 0,
            "neg_minus_pos": 0.02,
            "outside_delta": 0.04,
            "psnr": 22.0,
        },
        {
            "checkpoint": "b",
            "rank_success": 1,
            "neg_minus_pos": 0.06,
            "outside_delta": 0.01,
            "psnr": 24.0,
        },
        {
            "checkpoint": "b",
            "rank_success": 1,
            "neg_minus_pos": 0.07,
            "outside_delta": 0.02,
            "psnr": 23.0,
        },
    ]

    summary = compute_prompt_summary(rows)

    assert summary["a"]["rank_success_rate"] == 0.5
    assert summary["b"]["rank_success_rate"] == 1.0
    assert summary["best_prompt_checkpoint"] == "b"
    assert summary["best_psnr_checkpoint"] == "b"


def test_edge_aware_total_variation_downweights_l_channel_edges():
    ab = torch.tensor([[[[0.0, 1.0, 2.0]], [[0.0, 0.0, 0.0]]]])
    flat_l = torch.zeros(1, 1, 1, 3)
    edge_l = torch.tensor([[[[0.0, 1.0, 2.0]]]])

    flat = edge_aware_total_variation(ab, flat_l, edge_k=10.0)
    edge = edge_aware_total_variation(ab, edge_l, edge_k=10.0)

    assert flat["tv_ab"] > 0.0
    assert edge["edge_aware_tv_ab"] < flat["edge_aware_tv_ab"]


def test_phase3_train_options_accept_seed_without_changing_default():
    options = Phase3TrainOptions()
    parser = options.initialize(argparse.ArgumentParser())

    default = parser.parse_args([])
    seeded = parser.parse_args(["--seed", "2026"])

    assert default.seed is None
    assert seeded.seed == 2026
