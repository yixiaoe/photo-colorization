"""
POST /api/colorize
Input:  { "image_b64": "...", "method": "phase1"|"phase2"|"phase3",
          "prompts": {"0": "a red dog"},
          "detections": [...] }
Output: { "colorized_b64": "...", "gray_b64": "...",
          "instance_crops": [{id, gray_b64, colored_b64}] }

Inference logic mirrors test.py exactly for each phase.
"""
import base64
import io
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from flask import Blueprint, request, jsonify
from PIL import Image

# ensure code/ is on path
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

colorize_bp = Blueprint('colorize', __name__)

_models: dict = {}
DEVICE = torch.device('cpu')


def _base_opt(**kwargs):
    opt = types.SimpleNamespace(
        fineSize=256, ab_norm=110., l_norm=100., l_cent=50.,
        T=0.38, rebalance_gamma=0.5,
        gpu_ids=[], isTrain=False,
        num_classes=91, embed_dim=64,
        checkpoints_dir='./checkpoints',
        which_epoch='latest',
        box_num=8, score_thresh=0.5,
        **kwargs
    )
    return opt


# ── image helpers ──────────────────────────────────────────────────────────────

def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')


def _pil_to_b64(pil_img: Image.Image, fmt='PNG') -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1,3,H,W) or (3,H,W) float [0,1] → PIL RGB."""
    if t.dim() == 4:
        t = t[0]
    arr = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor(pil_img: Image.Image, sz: int, device) -> torch.Tensor:
    """PIL RGB → (1,3,sz,sz) float [0,1]."""
    pil_r = pil_img.resize((sz, sz), resample=Image.BILINEAR)
    arr = np.asarray(pil_r).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _is_color(pil_img: Image.Image) -> bool:
    arr = np.array(pil_img.convert('RGB')).astype(float)
    return float(arr.std(axis=2).mean()) > 5.0


# ── Phase 1 ───────────────────────────────────────────────────────────────────
# Mirrors test.py main() → model.set_input(data) → forward() → get_current_visuals()

def _get_phase1_model():
    if 'phase1' not in _models:
        from models.cnn_color_model import CnnColorModel
        opt = _base_opt(
            model='cnn_color', method='cnn_color', stage='full',
            name='cnn_color_imagenet',
        )
        m = CnnColorModel()
        m.initialize(opt)
        m.load_networks(60)
        m.eval()
        _models['phase1'] = (m, opt)
        print('[colorize] Phase 1 model loaded')
    return _models['phase1']


def _run_phase1(pil_img: Image.Image):
    from util.util import rgb2lab, lab2rgb, decode_zhang2016_annealed_mean
    model, opt = _get_phase1_model()
    sz = opt.fineSize
    orig_W, orig_H = pil_img.size

    rgb_t = _pil_to_tensor(pil_img, sz, DEVICE)  # (1,3,sz,sz)

    with torch.no_grad():
        # Exactly as test.py: set_input → forward → get_current_visuals
        # set_input computes real_L and real_ab from rgb_img
        lab = rgb2lab(rgb_t, opt)
        real_L = lab[:, [0]]   # (1,1,sz,sz)

        # forward: netG(real_L) → logits (1,313,sz/4,sz/4)
        logits = model.netG(real_L)

        # decode: annealed-mean
        pred_ab = decode_zhang2016_annealed_mean(
            logits.float(), model.pts_in_hull, T=opt.T, ab_norm_val=opt.ab_norm)

        # upsample ab to full sz
        pred_ab_up = F.interpolate(pred_ab, size=(sz, sz),
                                   mode='bilinear', align_corners=False)

        fake_lab = torch.cat([real_L, pred_ab_up], dim=1)
        fake_rgb = lab2rgb(fake_lab, opt).clamp(0, 1)  # (1,3,sz,sz)

    # upsample to original resolution
    fake_rgb_full = F.interpolate(fake_rgb, size=(orig_H, orig_W),
                                  mode='bilinear', align_corners=False)
    return _tensor_to_pil(fake_rgb_full)


# ── Phase 2 ───────────────────────────────────────────────────────────────────
# Mirrors test.py main() stage=fusion path:
#   _set_input_fusion_test(data) → forward() → get_current_visuals()

def _get_phase2_model():
    if 'phase2' not in _models:
        from models.inst_fusion_model import InstFusionModel
        opt = _base_opt(
            model='inst_fusion', method='inst_fusion', stage='fusion',
            name='inst_fusion_fusion',
            full_ckpt='checkpoints/inst_fusion_full/80_net_G.pth',
            inst_ckpt='checkpoints/inst_fusion_instance/25_net_G.pth',
            fusion_ckpt='checkpoints/inst_fusion_fusion/25_net_G.pth',
            lr_backbone=1e-5,
        )
        m = InstFusionModel()
        m.initialize(opt)
        m.load_networks(25)
        m.eval()
        _models['phase2'] = (m, opt)
        print('[colorize] Phase 2 model loaded')
    return _models['phase2']


def _label_id(label_str: str) -> int:
    from util.maskrcnn_helper import COCO_CLASSES
    try:
        return COCO_CLASSES.index(label_str)
    except (ValueError, AttributeError):
        return 1


def _run_phase2(pil_img: Image.Image, detections: list):
    from util.util import rgb2lab, decode_zhang2016_annealed_mean, lab2rgb
    from data_process.colorization_dataset import get_box_info

    model, opt = _get_phase2_model()
    sz = opt.fineSize
    orig_W, orig_H = pil_img.size

    rgb_t = _pil_to_tensor(pil_img, sz, DEVICE)  # (1,3,sz,sz)
    empty_box = len(detections) == 0

    # Build data dict exactly as _set_input_fusion_test expects:
    # 'rgb_img' (1,3,H,W), 'empty_box', and if not empty:
    # 'cropped_img' (1,N,3,H,W), 'class_labels' (1,N), box_info* (1,N,6)
    data = {'rgb_img': rgb_t, 'empty_box': empty_box}

    instance_crops_out = []
    inst_crops = []
    class_labels = []

    if not empty_box:
        n = len(detections)
        bi  = np.zeros((1, n, 6), dtype=np.int64)
        bi2 = np.zeros_like(bi)
        bi4 = np.zeros_like(bi)
        bi8 = np.zeros_like(bi)

        for i, d in enumerate(detections):
            x0, y0, x1, y1 = d['bbox']
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(orig_W, x1); y1 = min(orig_H, y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            crop = pil_img.crop((x0, y0, x1, y1)).resize((sz, sz), Image.BILINEAR)
            crop_t = torch.from_numpy(
                np.asarray(crop).astype(np.float32) / 255.0).permute(2, 0, 1)
            inst_crops.append(crop_t)
            lbl = d['label']
            class_labels.append(lbl if isinstance(lbl, int) else _label_id(lbl))
            bi[0, i]  = get_box_info((x0, y0, x1, y1), (orig_W, orig_H), sz)
            bi2[0, i] = get_box_info((x0, y0, x1, y1), (orig_W, orig_H), sz // 2)
            bi4[0, i] = get_box_info((x0, y0, x1, y1), (orig_W, orig_H), sz // 4)
            bi8[0, i] = get_box_info((x0, y0, x1, y1), (orig_W, orig_H), sz // 8)

        if inst_crops:
            cropped_img = torch.stack(inst_crops).unsqueeze(0)  # (1,N,3,H,W)
            data['cropped_img']  = cropped_img
            data['class_labels'] = torch.tensor(class_labels, dtype=torch.long).unsqueeze(0)
            data['box_info']     = torch.from_numpy(bi)
            data['box_info_2x']  = torch.from_numpy(bi2)
            data['box_info_4x']  = torch.from_numpy(bi4)
            data['box_info_8x']  = torch.from_numpy(bi8)
        else:
            data['empty_box'] = True

        # per-instance single-branch coloring for modal (FiLMInstanceGenerator only)
        netInst = getattr(model, 'netInst', None)
        if netInst is None and hasattr(model.netG, 'netInst'):
            netInst = model.netG.netInst
        pts = model.pts_in_hull

        for i, d in enumerate(detections):
            if i >= len(inst_crops):
                break
            crop_t = inst_crops[i].unsqueeze(0).to(DEVICE)
            lab_inst = rgb2lab(crop_t, opt)
            inst_L = lab_inst[:, [0]]
            label_t = torch.tensor([class_labels[i]], dtype=torch.long, device=DEVICE)
            try:
                with torch.no_grad():
                    result = netInst(inst_L, label_t) if netInst else None
                if result is not None:
                    out_c = result[0]  # first return value is out_class
                    pred_ab = decode_zhang2016_annealed_mean(
                        out_c.float(), pts, T=opt.T, ab_norm_val=opt.ab_norm)
                    pred_ab_up = F.interpolate(pred_ab, size=(sz, sz),
                                               mode='bilinear', align_corners=False)
                    fake_lab = torch.cat([inst_L, pred_ab_up], dim=1)
                    inst_colored = lab2rgb(fake_lab, opt).clamp(0, 1)
                    inst_gray = (inst_L * opt.l_norm + opt.l_cent) / 100.0
                    inst_gray = inst_gray.expand(-1, 3, -1, -1).clamp(0, 1)
                    instance_crops_out.append({
                        'id': i,
                        'label': d['label'],
                        'gray_b64':    _pil_to_b64(_tensor_to_pil(inst_gray)),
                        'colored_b64': _pil_to_b64(_tensor_to_pil(inst_colored)),
                    })
            except Exception as e:
                print(f'[colorize] inst {i} single-branch failed: {e}')

    # main fusion inference — mirrors test.py: _set_input_fusion_test → forward → get_current_visuals
    with torch.no_grad():
        model._set_input_fusion_test(data)
        model.forward()
    visuals = model.get_current_visuals()
    fake_rgb = visuals['fake_rgb']  # (1,3,sz,sz)
    fake_rgb_full = F.interpolate(fake_rgb, size=(orig_H, orig_W),
                                  mode='bilinear', align_corners=False)
    return _tensor_to_pil(fake_rgb_full), instance_crops_out


# ── Phase 3 ───────────────────────────────────────────────────────────────────
# Mirrors test.py _test_text_color() exactly

def _get_phase3_model():
    if 'phase3' not in _models:
        from models.text_color_model import TextColorModel
        opt = _base_opt(
            model='text_color', method='text_color',
            name='phase3_text_color',
            full_ckpt='checkpoints/inst_fusion_full/80_net_G.pth',
            inst_ckpt='checkpoints/inst_fusion_instance/25_net_G.pth',
            fusion_ckpt='checkpoints/inst_fusion_fusion/25_net_G.pth',
            clip_cache='datasets/phase3/clip_text_cache.pt',
            adapter_hidden=512, max_inst=8,
            use_amp=False, ab_cap=0.45,
        )
        m = TextColorModel()
        m.initialize(opt)
        m.load_networks(10)
        m.eval()
        _models['phase3'] = (m, opt)
        print('[colorize] Phase 3 model loaded')
    return _models['phase3']


def _run_phase3(pil_img: Image.Image, detections: list, prompts: dict):
    from util.util import rgb2lab, decode_zhang2016_annealed_mean, lab2rgb
    from data_process.colorization_dataset import get_box_info
    from util.maskrcnn_helper import class_name

    model, opt = _get_phase3_model()
    sz = opt.fineSize
    orig_W, orig_H = pil_img.size

    # Exactly as _test_text_color(): resize → rgb2lab → real_L_full
    pil_resized = pil_img.resize((sz, sz), resample=Image.BILINEAR)
    full_rgb = torch.from_numpy(
        np.asarray(pil_resized)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    full_rgb = full_rgb.to(DEVICE)
    lab_full = rgb2lab(full_rgb, opt)
    real_L_full = lab_full[:, [0]]

    empty_box = len(detections) == 0
    inst_L = None
    class_labels_t = None
    box_info_list = None
    text_inst = None

    if not empty_box:
        inst_crops, class_labels, captions = [], [], []
        n = len(detections)
        box_info    = np.zeros((n, 6), dtype=np.int64)
        box_info_2x = np.zeros_like(box_info)
        box_info_4x = np.zeros_like(box_info)
        box_info_8x = np.zeros_like(box_info)

        for i, d in enumerate(detections):
            x0, y0, x1, y1 = d['bbox']
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(orig_W, x1); y1 = min(orig_H, y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            crop = pil_img.crop((x0, y0, x1, y1)).resize((sz, sz), resample=Image.BILINEAR)
            inst_crops.append(
                torch.from_numpy(np.asarray(crop)).permute(2, 0, 1).float() / 255.0)
            lbl = d['label']
            class_labels.append(lbl if isinstance(lbl, int) else _label_id(lbl))
            box_info[i]    = get_box_info((x0, y0, x1, y1), pil_img.size, sz)
            box_info_2x[i] = get_box_info((x0, y0, x1, y1), pil_img.size, sz // 2)
            box_info_4x[i] = get_box_info((x0, y0, x1, y1), pil_img.size, sz // 4)
            box_info_8x[i] = get_box_info((x0, y0, x1, y1), pil_img.size, sz // 8)
            lbl_name = lbl if isinstance(lbl, str) else class_name(lbl)
            captions.append(prompts.get(str(i), f'a {lbl_name}'))

        if inst_crops:
            cropped_rgb = torch.stack(inst_crops).to(DEVICE)
            lab_inst = rgb2lab(cropped_rgb, opt)
            inst_L = lab_inst[:, [0]]
            class_labels_t = torch.tensor(class_labels, dtype=torch.long, device=DEVICE)
            box_info_list = [
                torch.from_numpy(box_info).to(DEVICE),
                torch.from_numpy(box_info_2x).to(DEVICE),
                torch.from_numpy(box_info_4x).to(DEVICE),
                torch.from_numpy(box_info_8x).to(DEVICE),
            ]
            text_inst = model.clip.encode(captions)
        else:
            empty_box = True

    text_bg = model.clip.encode([''])

    with torch.no_grad():
        out_class, _ = model.netT(
            real_L_full, inst_L, class_labels_t, box_info_list,
            text_inst, text_bg, empty_box=empty_box)

    pred_ab = decode_zhang2016_annealed_mean(
        out_class.float(), model.pts_in_hull, T=opt.T, ab_norm_val=opt.ab_norm)

    # ab cap — mirrors test.py
    ab_cap = getattr(opt, 'ab_cap', 0.45)
    if ab_cap > 0:
        mag = pred_ab.norm(dim=1, keepdim=True).clamp(min=1e-6)
        pred_ab = pred_ab * (ab_cap / mag).clamp(max=1.0)

    pred_ab_up = F.interpolate(pred_ab, size=(sz, sz),
                               mode='bilinear', align_corners=False)
    fake_lab = torch.cat([real_L_full, pred_ab_up], dim=1)
    fake_rgb = lab2rgb(fake_lab, opt).clamp(0, 1)
    fake_rgb_full = F.interpolate(fake_rgb, size=(orig_H, orig_W),
                                  mode='bilinear', align_corners=False)
    return _tensor_to_pil(fake_rgb_full)


# ── main route ────────────────────────────────────────────────────────────────

@colorize_bp.route('/api/colorize', methods=['POST'])
def colorize():
    data = request.get_json(force=True)
    if not data or 'image_b64' not in data:
        return jsonify({'error': 'image_b64 required'}), 400

    method     = data.get('method', 'phase2')
    prompts    = data.get('prompts', {})
    detections = data.get('detections', [])

    try:
        pil_img = _b64_to_pil(data['image_b64'])
    except Exception as e:
        return jsonify({'error': f'image decode failed: {e}'}), 400

    orig_W, orig_H = pil_img.size
    is_color = _is_color(pil_img)

    # gray_b64 for color input images
    gray_b64 = None
    if is_color:
        from util.util import rgb2lab
        opt_tmp = _base_opt()
        rgb_t = _pil_to_tensor(pil_img, 256, DEVICE)
        lab = rgb2lab(rgb_t, opt_tmp)
        real_L = lab[:, [0]]
        gray_t = (real_L * opt_tmp.l_norm + opt_tmp.l_cent) / 100.0
        gray_t = gray_t.expand(-1, 3, -1, -1).clamp(0, 1)
        gray_full = F.interpolate(gray_t, size=(orig_H, orig_W),
                                  mode='bilinear', align_corners=False)
        gray_b64 = _pil_to_b64(_tensor_to_pil(gray_full))

    instance_crops_out = []
    try:
        if method == 'phase1':
            colored = _run_phase1(pil_img)
        elif method == 'phase2':
            colored, instance_crops_out = _run_phase2(pil_img, detections)
        elif method == 'phase3':
            colored = _run_phase3(pil_img, detections, prompts)
        else:
            return jsonify({'error': f'unknown method: {method}'}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    resp = {'colorized_b64': _pil_to_b64(colored)}
    if gray_b64:
        resp['gray_b64'] = gray_b64
    if instance_crops_out:
        resp['instance_crops'] = instance_crops_out
    return jsonify(resp)
