"""Create Phase 3 prompt-control visualizations for sampled COCO images.

Each output image contains:
  original | target mask | grayscale | positive prompt | negative prompt | other prompt
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_process.color_object_utils import (
    choose_caption_for_object,
    dominant_hsv_color,
    ensure_color_object_caption,
    negative_color_for,
    resize_binary_mask,
    segmentation_to_mask,
)
from models.text_color_model import TextColorModel
from util.util import tensor2im


def _font(size):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _wrap_text(text, font, max_width):
    words = text.split()
    lines = []
    current = ""
    scratch = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(scratch)
    for word in words:
        trial = word if not current else current + " " + word
        width = draw.textbbox((0, 0), trial, font=font)[2]
        if width <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


def _panel(image, title, subtitle="", size=256, header_h=72):
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size + header_h), (245, 245, 245))
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(16)
    sub_font = _font(12)
    draw.text((8, 8), title, fill=(20, 20, 20), font=title_font)
    y = 30
    if subtitle:
        for line in _wrap_text(subtitle, sub_font, size - 16):
            draw.text((8, y), line, fill=(60, 60, 60), font=sub_font)
            y += 14
    return canvas


def _bbox_mask(bbox, width, height):
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    if not bbox or len(bbox) != 4:
        return mask
    x, y, w, h = bbox
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(int(width), int(round(x + w)))
    y1 = min(int(height), int(round(y + h)))
    mask[y0:y1, x0:x1] = 1
    return mask


def _instance_mask(instance, width, height):
    seg = instance.get("segmentation")
    if isinstance(seg, list) and seg:
        try:
            return segmentation_to_mask(seg, width, height)
        except Exception:
            pass
    return _bbox_mask(instance.get("bbox"), width, height)


def _select_target(record):
    instances = record.get("instances", [])
    valid = [i for i in instances if i.get("bbox") and not i.get("iscrowd")]
    if not valid:
        valid = [i for i in instances if i.get("bbox")]
    non_person = [i for i in valid if i.get("category_name") != "person"]
    pool = non_person or valid
    if not pool:
        return None
    return max(pool, key=lambda i: float(i.get("area") or 0.0))


def _mask_overlay(image, mask):
    base = image.convert("RGB")
    mask = np.asarray(mask).astype(bool)
    color = np.zeros((base.height, base.width, 3), dtype=np.uint8)
    color[..., 0] = 255
    base_arr = np.asarray(base).copy()
    base_arr[mask] = (0.55 * base_arr[mask] + 0.45 * color[mask]).astype(np.uint8)
    out = Image.fromarray(base_arr)
    draw = ImageDraw.Draw(out)
    ys, xs = np.where(mask)
    if xs.size:
        draw.rectangle(
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            outline=(255, 20, 20),
            width=4,
        )
    return out


def _gray_panel_image(image, size):
    return image.convert("L").convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def _make_opt(args, checkpoint_name):
    return SimpleNamespace(
        isTrain=False,
        gpu_ids=[],
        checkpoints_dir=args.checkpoints_dir,
        results_dir=args.results_dir,
        name=args.name,
        which_epoch=checkpoint_name,
        full_ckpt=args.full_ckpt,
        clip_arch=args.clip_arch,
        clip_pretrained_path=args.clip_pretrained_path,
        num_heads=args.num_heads,
        ab_norm=110.0,
        ab_max=110.0,
        ab_quant=10.0,
        l_norm=100.0,
        l_cent=50.0,
        mask_cent=0.5,
    )


def _move_to_device(model, device):
    model.device = device
    model.netG.to(device)
    model.pts_in_hull = model.pts_in_hull.to(device)
    model.clip_encoder.device = device
    model.clip_encoder.model.to(device)
    model.gpu_ids = []


def _load_model(args, checkpoint_name, device):
    opt = _make_opt(args, checkpoint_name)
    model = TextColorModel()
    model.initialize(opt)
    _move_to_device(model, device)
    model.load_networks(checkpoint_name)
    model.eval()
    return model


@torch.no_grad()
def _predict(model, image, prompts, size):
    tfm = T.Compose([
        T.Resize((size, size), interpolation=Image.Resampling.BILINEAR),
        T.ToTensor(),
    ])
    rgb = tfm(image).unsqueeze(0).repeat(len(prompts), 1, 1, 1)
    model.set_input({"rgb_img": rgb, "caption": prompts})
    model.forward()
    visuals = model.get_current_visuals()
    return [Image.fromarray(tensor2im(visuals["fake_rgb"][i:i + 1])) for i in range(len(prompts))]


def _compose(record, original, mask, prompts, outputs, checkpoint_name, size):
    mask_img = _mask_overlay(original, mask)
    gray = _gray_panel_image(original, size)
    target = record["target"]
    panels = [
        _panel(original, "Original", record["file_name"], size=size),
        _panel(mask_img, "Target mask", f"{target['object']} / {target['color']}", size=size),
        _panel(gray, "Gray input", "L channel only", size=size),
        _panel(outputs[0], "Correct prompt", prompts[0], size=size),
        _panel(outputs[1], "Negative prompt", prompts[1], size=size),
        _panel(outputs[2], "Other prompt", prompts[2], size=size),
    ]
    width = sum(p.width for p in panels)
    height = panels[0].height + 36
    canvas = Image.new("RGB", (width, height), (230, 230, 230))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 8),
        f"{checkpoint_name} | image_id={record['image_id']} | target={target['object']}",
        fill=(20, 20, 20),
        font=_font(16),
    )
    x = 0
    for p in panels:
        canvas.paste(p, (x, 36))
        x += p.width
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", default="results/phase3_coco_val30_10660_20260604_142546")
    parser.add_argument("--out_dir", default="results/phase3_prompt_mps_val30")
    parser.add_argument("--checkpoints", nargs="+", default=["best", "latest"])
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--name", default="text_color_5090_run")
    parser.add_argument("--full_ckpt", default="checkpoints/inst_full/80_net_G.pth")
    parser.add_argument("--clip_arch", default="ViT-B-32-quickgelu")
    parser.add_argument("--clip_pretrained_path", default="checkpoints/clip/open_clip_model.safetensors")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")
    device = torch.device("mps")

    sample_dir = Path(args.sample_dir)
    manifest_path = sample_dir / "annotations" / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        records = json.load(f)["records"][:args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for checkpoint_name in args.checkpoints:
        model = _load_model(args, checkpoint_name, device)
        ckpt_out = out_dir / checkpoint_name
        ckpt_out.mkdir(parents=True, exist_ok=True)

        for rec in records:
            image_path = sample_dir / rec["image_path"]
            original = Image.open(image_path).convert("RGB")
            target = _select_target(rec)
            if target is None:
                continue

            mask = _instance_mask(target, rec["width"], rec["height"])
            color_est = dominant_hsv_color(np.asarray(original), mask)
            color = color_est.color
            neg_color = negative_color_for(color)
            obj = target.get("category_name", "object")
            base_caption = choose_caption_for_object(rec["captions"], obj) or rec["captions"][0]
            pos_prompt = ensure_color_object_caption(base_caption, color, obj)
            neg_prompt = ensure_color_object_caption(base_caption, neg_color, obj)
            other_prompt = "a colorful outdoor scene with blue sky and green grass"
            prompts = [pos_prompt, neg_prompt, other_prompt]

            outputs = _predict(model, original, prompts, args.size)
            rec_summary = {
                "checkpoint": checkpoint_name,
                "image_id": rec["image_id"],
                "file_name": rec["file_name"],
                "target_object": obj,
                "target_color": color,
                "negative_color": neg_color,
                "color_confidence": color_est.confidence,
                "positive_prompt": pos_prompt,
                "negative_prompt": neg_prompt,
                "other_prompt": other_prompt,
            }
            rec["target"] = {"object": obj, "color": color}
            composite = _compose(rec, original, mask, prompts, outputs, checkpoint_name, args.size)
            out_path = ckpt_out / f"{rec['index']:02d}_{Path(rec['file_name']).stem}.png"
            composite.save(out_path)
            rec_summary["output_path"] = str(out_path)
            summary.append(rec_summary)
            print(f"[{checkpoint_name}] saved {out_path}")

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Done. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
