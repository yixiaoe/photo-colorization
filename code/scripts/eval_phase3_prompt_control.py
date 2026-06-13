"""Evaluate Phase 3 prompt control on fixed color-object records."""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _float(row, key):
    return float(row.get(key, 0.0))


def _mean(rows, key):
    if not rows:
        return 0.0
    return sum(_float(row, key) for row in rows) / len(rows)


def compute_prompt_summary(rows):
    """Summarize per-checkpoint prompt-control metrics."""
    checkpoints = []
    for row in rows:
        ckpt = row["checkpoint"]
        if ckpt not in checkpoints:
            checkpoints.append(ckpt)

    summary = {}
    for ckpt in checkpoints:
        part = [row for row in rows if row["checkpoint"] == ckpt]
        summary[ckpt] = {
            "n": len(part),
            "mean_pos_dist": _mean(part, "pos_dist"),
            "mean_neg_dist": _mean(part, "neg_dist"),
            "mean_other_dist": _mean(part, "other_dist"),
            "mean_neg_minus_pos": _mean(part, "neg_minus_pos"),
            "mean_rank_loss": _mean(part, "rank_loss"),
            "rank_success_rate": _mean(part, "rank_success"),
            "mean_inside_delta": _mean(part, "inside_delta"),
            "mean_outside_delta": _mean(part, "outside_delta"),
            "mean_outside_to_inside": _mean(part, "outside_to_inside"),
            "mean_tv_ab": _mean(part, "tv_ab"),
            "mean_edge_aware_tv_ab": _mean(part, "edge_aware_tv_ab"),
            "mean_tv_inside": _mean(part, "tv_inside"),
            "mean_tv_outside": _mean(part, "tv_outside"),
            "mean_psnr": _mean(part, "psnr"),
            "mean_ssim": _mean(part, "ssim"),
        }

    def best_by(metric):
        if not checkpoints:
            return ""
        return max(checkpoints, key=lambda ckpt: summary[ckpt][metric])

    def prompt_key(ckpt):
        item = summary[ckpt]
        return (
            item["rank_success_rate"],
            item["mean_neg_minus_pos"],
            -item["mean_outside_delta"],
            item["mean_psnr"],
        )

    def balanced_key(ckpt):
        item = summary[ckpt]
        return (
            item["rank_success_rate"],
            item["mean_neg_minus_pos"],
            -item["mean_outside_delta"],
            -item["mean_edge_aware_tv_ab"],
            item["mean_psnr"],
        )

    if checkpoints:
        summary["best_prompt_checkpoint"] = max(checkpoints, key=prompt_key)
        summary["best_balanced_checkpoint"] = max(checkpoints, key=balanced_key)
        summary["best_psnr_checkpoint"] = best_by("mean_psnr")
    return summary


def _weighted_mean(values, weights=None):
    if values.numel() == 0:
        return values.new_tensor(0.0)
    if weights is None:
        return values.mean()
    total = weights.sum()
    if float(total.detach().cpu()) <= 1e-8:
        return values.new_tensor(0.0)
    return (values * weights).sum() / total


def _weighted_penalty_mean(values, weights):
    if values.numel() == 0:
        return values.new_tensor(0.0)
    return (values * weights).mean()


def edge_aware_total_variation(ab, l_channel, mask=None, edge_k=10.0):
    """Compute TV metrics for normalized ab, optionally split by mask."""
    if l_channel.shape[-2:] != ab.shape[-2:]:
        l_channel = F.interpolate(
            l_channel, size=ab.shape[-2:], mode="bilinear",
            align_corners=False)
    if mask is not None and mask.shape[-2:] != ab.shape[-2:]:
        mask = F.interpolate(mask.float(), size=ab.shape[-2:], mode="nearest")

    dx = (ab[:, :, :, 1:] - ab[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    dy = (ab[:, :, 1:, :] - ab[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    l_dx = (l_channel[:, :, :, 1:] - l_channel[:, :, :, :-1]).abs()
    l_dy = (l_channel[:, :, 1:, :] - l_channel[:, :, :-1, :]).abs()
    wx = torch.exp(-float(edge_k) * l_dx)
    wy = torch.exp(-float(edge_k) * l_dy)

    tv = 0.5 * (_weighted_mean(dx) + _weighted_mean(dy))
    edge_tv = 0.5 * (
        _weighted_penalty_mean(dx, wx) + _weighted_penalty_mean(dy, wy))

    result = {
        "tv_ab": float(tv.detach().cpu()),
        "edge_aware_tv_ab": float(edge_tv.detach().cpu()),
        "tv_inside": 0.0,
        "tv_outside": 0.0,
    }
    if mask is None:
        return result

    mask = (mask > 0.5).float()
    inside_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    inside_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    outside = 1.0 - mask
    outside_x = outside[:, :, :, 1:] * outside[:, :, :, :-1]
    outside_y = outside[:, :, 1:, :] * outside[:, :, :-1, :]
    result["tv_inside"] = float(
        (0.5 * (_weighted_mean(dx, inside_x) +
                _weighted_mean(dy, inside_y))).detach().cpu())
    result["tv_outside"] = float(
        (0.5 * (_weighted_mean(dx, outside_x) +
                _weighted_mean(dy, outside_y))).detach().cpu())
    return result


def _make_opt(args, checkpoint_name):
    gpu_ids = []
    if args.gpu_ids != "-1" and torch.cuda.is_available():
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    return SimpleNamespace(
        isTrain=False,
        gpu_ids=gpu_ids,
        checkpoints_dir=args.checkpoints_dir,
        results_dir=args.results_dir,
        name=args.name,
        which_epoch=checkpoint_name,
        full_ckpt=args.full_ckpt,
        clip_arch=args.clip_arch,
        clip_pretrained_path=args.clip_pretrained_path,
        num_heads=args.num_heads,
        ab_norm=args.ab_norm,
        ab_max=110.0,
        ab_quant=10.0,
        l_norm=100.0,
        l_cent=50.0,
        mask_cent=0.5,
    )


def _load_model(args, checkpoint_name):
    from models.text_color_model import TextColorModel

    model = TextColorModel()
    model.initialize(_make_opt(args, checkpoint_name))
    model.load_networks(checkpoint_name)
    model.eval()
    return model


def _select_indices(dataset, limit, min_area_ratio):
    selected = []
    seen = set()
    for idx, rec in enumerate(dataset.records):
        area = float(rec.get("area") or 0.0)
        denom = max(1.0, float(rec.get("width", 1)) * float(rec.get("height", 1)))
        key = (rec.get("object", ""), rec.get("color", ""))
        if area / denom < min_area_ratio or key in seen:
            continue
        selected.append(idx)
        seen.add(key)
        if len(selected) >= limit:
            return selected
    for idx in range(len(dataset)):
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= limit:
            break
    return selected


def _font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _tensor_to_image(tensor):
    from util.util import tensor2im

    return Image.fromarray(tensor2im(tensor))


def _panel(image, title, subtitle="", size=256, header_h=56):
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size + header_h), (245, 245, 245))
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((7, 6), title[:32], fill=(20, 20, 20), font=_font(15))
    if subtitle:
        draw.text((7, 30), subtitle[:44], fill=(70, 70, 70), font=_font(11))
    return canvas


def _save_grid(path, visuals, prompts, row, size):
    panels = [
        _panel(_tensor_to_image(visuals["real_gray"][0:1]), "gray",
               f"image_id={row['image_id']}", size=size),
        _panel(_tensor_to_image(visuals["real_rgb"][0:1]), "gt",
               f"{row['object']} {row['color']}", size=size),
        _panel(_tensor_to_image(visuals["fake_rgb"][0:1]), "pos",
               prompts[0], size=size),
        _panel(_tensor_to_image(visuals["fake_rgb"][1:2]), "neg",
               prompts[1], size=size),
        _panel(_tensor_to_image(visuals["fake_rgb"][2:3]), "other",
               prompts[2], size=size),
    ]
    width = sum(panel.width for panel in panels)
    height = panels[0].height + 34
    canvas = Image.new("RGB", (width, height), (230, 230, 230))
    draw = ImageDraw.Draw(canvas)
    title = (
        f"{row['checkpoint']} idx={row['idx']} "
        f"pos={row['pos_dist']:.4f} neg={row['neg_dist']:.4f} "
        f"out={row['outside_delta']:.4f} psnr={row['psnr']:.2f}"
    )
    draw.text((8, 8), title, fill=(20, 20, 20), font=_font(12))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 34))
        x += panel.width
    canvas.save(path)


def _evaluate_checkpoint(args, dataset, indices, checkpoint_name, out_dir):
    from models.text_color_model import _masked_ab_huber_loss
    from util.metrics import compute_psnr, compute_ssim
    from util.util import decode_zhang2016_annealed_mean

    model = _load_model(args, checkpoint_name)
    ckpt_dir = out_dir / checkpoint_name
    if args.save_images:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    device = model.device
    for idx in indices:
        item = dataset[idx]
        rec = dataset.records[idx]
        rgb = item["rgb_img"].unsqueeze(0).repeat(3, 1, 1, 1)
        obj = rec.get("object", "object")
        prompts = [
            item["caption_pos"],
            item["caption_neg"],
            f"a colorful photo of a {obj}",
        ]
        data = {"rgb_img": rgb, "caption": prompts}
        with torch.no_grad():
            model.set_input(data)
            model.forward()
            pred_ab_4x = decode_zhang2016_annealed_mean(
                model.pred_class, model.pts_in_hull, T=args.decode_temp,
                ab_norm_val=model.opt.ab_norm)
            gt_ab_4x = F.interpolate(
                model.real_ab, size=pred_ab_4x.shape[-2:],
                mode="bilinear", align_corners=False)
            mask4 = item["mask_4x"].unsqueeze(0).to(device).float()
            outside4 = 1.0 - mask4
            pos_dist = _masked_ab_huber_loss(
                pred_ab_4x[0:1], gt_ab_4x[0:1], mask4).item()
            neg_dist = _masked_ab_huber_loss(
                pred_ab_4x[1:2], gt_ab_4x[1:2], mask4).item()
            other_dist = _masked_ab_huber_loss(
                pred_ab_4x[2:3], gt_ab_4x[2:3], mask4).item()
            inside_delta = _masked_ab_huber_loss(
                pred_ab_4x[0:1], pred_ab_4x[1:2], mask4).item()
            outside_delta = _masked_ab_huber_loss(
                pred_ab_4x[0:1], pred_ab_4x[1:2], outside4).item()
            rank_loss = max(0.0, args.rank_margin + pos_dist - neg_dist)
            rank_success = 1 if neg_dist >= pos_dist + args.rank_margin else 0
            tv = edge_aware_total_variation(
                pred_ab_4x[0:1], model.real_L[0:1], mask4,
                edge_k=args.edge_k)
            visuals = model.get_current_visuals()
            psnr = compute_psnr(
                visuals["fake_rgb"][0:1].clamp(0, 1),
                visuals["real_rgb"][0:1].clamp(0, 1))
            ssim = compute_ssim(
                visuals["fake_rgb"][0:1].clamp(0, 1),
                visuals["real_rgb"][0:1].clamp(0, 1))

        row = {
            "checkpoint": checkpoint_name,
            "idx": idx,
            "image_id": rec.get("image_id", -1),
            "file_name": rec.get("file_name", ""),
            "object": obj,
            "color": rec.get("color", ""),
            "neg_color": rec.get("neg_color", ""),
            "pos_dist": pos_dist,
            "neg_dist": neg_dist,
            "other_dist": other_dist,
            "neg_minus_pos": neg_dist - pos_dist,
            "rank_loss": rank_loss,
            "rank_success": rank_success,
            "inside_delta": inside_delta,
            "outside_delta": outside_delta,
            "outside_to_inside": outside_delta / max(inside_delta, 1e-8),
            "tv_ab": tv["tv_ab"],
            "edge_aware_tv_ab": tv["edge_aware_tv_ab"],
            "tv_inside": tv["tv_inside"],
            "tv_outside": tv["tv_outside"],
            "psnr": psnr,
            "ssim": ssim,
            "caption_pos": prompts[0],
            "caption_neg": prompts[1],
            "caption_other": prompts[2],
        }
        rows.append(row)
        if args.save_images:
            obj_name = obj.replace(" ", "-")
            filename = f"{idx:04d}_{obj_name}_{row['color']}.jpg"
            _save_grid(ckpt_dir / filename, visuals, prompts, row, args.image_size)
        print(
            f"[{checkpoint_name}] idx={idx} {obj} {row['color']} "
            f"pos={pos_dist:.4f} neg={neg_dist:.4f} "
            f"rank_ok={rank_success} out={outside_delta:.4f} "
            f"tv={tv['edge_aware_tv_ab']:.4f} psnr={psnr:.2f}",
            flush=True,
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_file",
                        default="data/phase3_color_object_no_person_val.jsonl")
    parser.add_argument("--checkpoints", nargs="+", default=["best", "latest"])
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--name", default="text_color_5090_run")
    parser.add_argument("--full_ckpt", default="checkpoints/inst_full/80_net_G.pth")
    parser.add_argument("--clip_arch", default="ViT-B-32-quickgelu")
    parser.add_argument("--clip_pretrained_path",
                        default="checkpoints/clip/open_clip_model.safetensors")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ab_norm", type=float, default=110.0)
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--fineSize", type=int, default=256)
    parser.add_argument("--min_area_ratio", type=float, default=0.015)
    parser.add_argument("--rank_margin", type=float, default=0.05)
    parser.add_argument("--decode_temp", type=float, default=0.38)
    parser.add_argument("--edge_k", type=float, default=10.0)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--save_images", action="store_true")
    return parser.parse_args()


def main():
    from data_process.color_object_dataset import CocoColorObjectDataset

    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_dir) / (
        f"{args.name}_prompt_control_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = CocoColorObjectDataset(
        args.records_file, fine_size=args.fineSize, split="val",
        random_flip=False)
    indices = _select_indices(dataset, args.limit, args.min_area_ratio)
    print(f"records={len(dataset)} selected_indices={indices}")
    all_rows = []
    for checkpoint_name in args.checkpoints:
        all_rows.extend(
            _evaluate_checkpoint(args, dataset, indices, checkpoint_name, out_dir))

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = compute_prompt_summary(all_rows)
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "records_file": args.records_file,
            "indices": indices,
            "summary": summary,
        }, f, indent=2)
    print("SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"metrics_csv={metrics_path}")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
