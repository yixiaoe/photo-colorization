"""
Phase 3 developer verification — runs without GPU / open_clip / pycocotools.

Two checks (both required before training):
  1. EQUIVALENCE: TextColorPipeline with zero-init adapters produces output
     bit-equal to FusionPipeline.forward on the same input. Without ckpts
     it uses random init; with --full_ckpt/--inst_ckpt/--fusion_ckpt it
     uses the real Phase 2 weights for a stronger check.
  2. SMOKE: TextColorModel.set_input → forward → backward → optimize
     completes with real gradients on a synthetic batch, and frozen
     Phase 2 weights accumulate no gradient.

Run from code/:
  python scripts/verify_phase3.py           # offline, random init
  python scripts/verify_phase3.py --with_ckpts   # uses Phase 2 ckpts
"""
import argparse
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── make the module imports work without open_clip / pycocotools ───────────

def _stub_optional_deps():
    """Replace optional deps with no-op stubs so the test runs anywhere."""
    fake_pycocotools = types.ModuleType('pycocotools')
    fake_pycocotools_coco = types.ModuleType('pycocotools.coco')
    fake_pycocotools_coco.COCO = object
    sys.modules.setdefault('pycocotools', fake_pycocotools)
    sys.modules.setdefault('pycocotools.coco', fake_pycocotools_coco)

    # stub CLIPTextEncoder so we don't need open_clip
    import util.clip_encoder as ce

    class _FakeCLIP:
        embed_dim = 512
        def __init__(self, device='cpu', cache_path=None):
            self.device = torch.device(device)
        def encode(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            return torch.zeros(len(texts), self.embed_dim, device=self.device)

    ce.CLIPTextEncoder = _FakeCLIP


# ── 1. equivalence ─────────────────────────────────────────────────────────

def _load(net, path, label):
    if not path or not os.path.isfile(path):
        print(f'  [skip] {label}: {path!r}')
        return False
    state = torch.load(path, map_location='cpu')
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f'  [load] {label}: missing={len(missing)} unexpected={len(unexpected)}')
    return True


def test_equivalence(args) -> bool:
    print('\n== Test 1: TextColorPipeline ≡ Phase 2 fusion (zero-init adapters) ==')
    from models.networks import FiLMInstanceGenerator, FusionPipeline
    from models.text_color_networks import TextColorPipeline

    torch.manual_seed(args.seed)
    netInst   = FiLMInstanceGenerator(num_classes=91, embed_dim=64).eval()
    netFusion = FusionPipeline().eval()

    if args.with_ckpts:
        _load(netInst,   args.inst_ckpt,   'instance')
        _load(netFusion, args.fusion_ckpt, 'fusion')
        _load(netFusion, args.full_ckpt,   'full -> fusion backbone')

    tcp = TextColorPipeline(netInst, netFusion, clip_dim=512).eval()
    print(f'  adapter params: {tcp.num_trainable_parameters():,}')
    for n, p in tcp.named_parameters():
        if p.requires_grad and p.abs().sum().item() != 0.0:
            print(f'  [FAIL] adapter {n} NOT zero-init')
            return False
    print('  ✓ all adapter params zero-init')

    sz, N = 256, 3
    gray_full = torch.randn(1, 1, sz, sz)
    inst_L = torch.randn(N, 1, sz, sz)
    class_labels = torch.tensor([1, 18, 3])
    box_info_list = []
    for scale in (1, 2, 4, 8):
        s = sz // scale
        box_info_list.append(torch.tensor([
            [0, s - 100, 0, s - 100, 100, 100],
            [10, s - 110, 10, s - 110, 100, 100],
            [20, s - 120, 20, s - 120, 100, 100],
        ], dtype=torch.long))

    text_inst = torch.randn(N, 512)
    text_bg   = torch.randn(1, 512)

    with torch.no_grad():
        _, _, fm = netInst(inst_L, class_labels)
        ref_feats = [{k: fm[k][[i]] for k in fm} for i in range(N)]
        ref_cls, ref_reg = netFusion(gray_full, ref_feats, box_info_list)
        new_cls, new_reg = tcp(gray_full, inst_L, class_labels,
                               box_info_list, text_inst, text_bg,
                               empty_box=False)

    dc = (ref_cls - new_cls).abs().max().item()
    dr = (ref_reg - new_reg).abs().max().item()
    print(f'  max |Δ out_class| = {dc:.2e}')
    print(f'  max |Δ out_reg|   = {dr:.2e}')

    with torch.no_grad():
        ref_cls_e, ref_reg_e = netFusion(gray_full, [], None)
        new_cls_e, new_reg_e = tcp(gray_full, None, None, None,
                                    None, text_bg, empty_box=True)
    dc_e = (ref_cls_e - new_cls_e).abs().max().item()
    dr_e = (ref_reg_e - new_reg_e).abs().max().item()
    print(f'  empty_box max Δ: cls={dc_e:.2e}  reg={dr_e:.2e}')

    ok = max(dc, dr, dc_e, dr_e) < 1e-4
    print('  ✓ PASS' if ok else '  ✗ FAIL')
    return ok


# ── 2. smoke ───────────────────────────────────────────────────────────────

def _build_smoke_opt():
    class _O: pass
    o = _O()
    o.gpu_ids = []
    o.isTrain = True
    o.checkpoints_dir = '/tmp/phase3_verify_ckpt'
    o.results_dir = '/tmp/phase3_verify_results'
    o.name = 'verify'
    o.full_ckpt = o.inst_ckpt = o.fusion_ckpt = ''
    o.clip_cache = ''
    o.ab_norm = 110.;  o.ab_max = 110.;  o.ab_quant = 10.
    o.l_norm  = 100.;  o.l_cent = 50.;   o.mask_cent = 0.5
    o.A = int(2 * o.ab_max / o.ab_quant + 1);  o.B = o.A
    o.T = 0.38;  o.rebalance_gamma = 0.5
    o.lr = 1e-4;  o.beta1 = 0.5
    o.lr_policy = 'lambda'
    o.niter = 1;  o.niter_decay = 1;  o.epoch_count = 0
    o.huber_weight = 3.0
    o.lambda_inst = 1.0
    o.lambda_rank = 0.1
    o.lambda_outside = 0.2
    o.rank_margin = 0.05
    o.rank_warmup_epoch = 0
    o.rank_warmup_len = 1
    return o


def _fake_batch(sz=64, N=3, device='cpu'):
    full_rgb = torch.rand(1, 3, sz, sz, device=device)
    cropped  = torch.rand(N, 3, sz, sz, device=device)
    labels   = torch.tensor([1, 18, 3], dtype=torch.long, device=device)

    def bi(scale):
        s = sz // scale
        return torch.tensor([
            [0, s - 10, 0, s - 10, 10, 10],
            [5, s - 15, 5, s - 15, 10, 10],
            [3, s - 13, 3, s - 13, 10, 10],
        ], dtype=torch.long, device=device)

    m_full = torch.zeros(N, 1, sz, sz, device=device)
    for i in range(N):
        x0, y0 = 5 + i * 5, 5 + i * 5
        m_full[i, 0, y0:y0 + 30, x0:x0 + 30] = 1.0
    m_4x = torch.nn.functional.interpolate(m_full, size=(sz // 4, sz // 4),
                                           mode='nearest')

    return {
        'full_rgb':       full_rgb,
        'cropped_rgb':    cropped,
        'class_labels':   labels,
        'class_names':    ['person', 'dog', 'car'],
        'box_info':       bi(1),  'box_info_2x': bi(2),
        'box_info_4x':    bi(4),  'box_info_8x': bi(8),
        'masks_full':     m_full, 'masks_4x':    m_4x,
        'caption_pos':    ['a red person', 'a yellow dog', 'a blue car'],
        'caption_neg':    ['a cyan person', 'a purple dog', 'an orange car'],
        'caption_bg_pos': 'outdoor scene', 'caption_bg_neg': 'indoor room',
        'empty_box':      False, 'file_id':     'verify',
    }


def test_smoke() -> bool:
    print('\n== Test 2: TextColorModel optimize_parameters (synthetic batch) ==')
    from models.text_color_model import TextColorModel
    opt = _build_smoke_opt()
    model = TextColorModel()
    model.initialize(opt)
    model.set_epoch(1)
    model.train()

    for step in range(3):
        model.set_input(_fake_batch(sz=64, N=3))
        model.optimize_parameters()
        losses = model.get_current_losses()
        print(f'  step {step}: ' +
              '  '.join(f'{k}={v:.4f}' for k, v in losses.items()))

    # gradient sanity (manual forward/backward; no optimizer.step)
    model.set_input(_fake_batch(sz=64, N=3))
    model.optimizer.zero_grad()
    model.forward()
    model.backward()
    model.loss_G.backward()

    g_inst = sum(p.grad.abs().sum().item()
                 for p in model.netT.inst_adapter.parameters() if p.grad is not None)
    g_bg   = sum(p.grad.abs().sum().item()
                 for p in model.netT.bg_adapter.parameters() if p.grad is not None)
    print(f'  Σ|grad inst_adapter| = {g_inst:.3e}')
    print(f'  Σ|grad bg_adapter|   = {g_bg:.3e}')
    if g_inst <= 0 or g_bg <= 0:
        print('  ✗ FAIL: at least one adapter received no gradient')
        return False

    for n, p in model.netInst.named_parameters():
        if p.grad is not None:
            print(f'  ✗ FAIL: frozen netInst.{n} has grad')
            return False
    for n, p in model.netFusion.named_parameters():
        if p.grad is not None:
            print(f'  ✗ FAIL: frozen netFusion.{n} has grad')
            return False
    print('  ✓ frozen Phase-2 networks: no gradients')

    # empty_box path
    eb = _fake_batch(sz=64, N=1)
    eb['empty_box'] = True
    eb.pop('cropped_rgb');  eb.pop('class_labels')
    model.set_input(eb)
    model.optimize_parameters()
    print('  ✓ empty_box path OK')

    # save round-trip
    model.save_networks('latest')
    print('  ✓ save_networks OK')
    return True


# ── entry ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--with_ckpts', action='store_true',
                    help='load real Phase 2 ckpts (otherwise random init)')
    ap.add_argument('--full_ckpt',   default='checkpoints/inst_fusion_full/80_net_G.pth')
    ap.add_argument('--inst_ckpt',   default='checkpoints/inst_fusion_instance/25_net_G.pth')
    ap.add_argument('--fusion_ckpt', default='checkpoints/inst_fusion_fusion/25_net_G.pth')
    ap.add_argument('--seed',        type=int, default=0)
    args = ap.parse_args()

    _stub_optional_deps()

    ok = True
    ok &= test_equivalence(args)
    ok &= test_smoke()

    print('\n[RESULT]', 'ALL PASS ✓' if ok else 'FAIL ✗')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
