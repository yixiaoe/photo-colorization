"""
Download CLIP ViT-B/32 weights to a local cache so training does not depend
on HuggingFace connectivity at runtime.

Saves the model state_dict to:
  code/checkpoints/clip/ViT-B-32-quickgelu.pt

Usage (run from code/):
  python scripts/download_clip.py
"""
import argparse
import os

import torch

try:
    import open_clip
except ImportError:
    raise SystemExit(
        '[download_clip] open_clip is not installed. '
        'Run: pip install open_clip_torch ftfy regex'
    )


_DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'checkpoints', 'clip', 'ViT-B-32-quickgelu.pt',
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', default='ViT-B-32-quickgelu')
    ap.add_argument('--pretrained', default='laion400m_e32',
                    help='open_clip pretrained tag')
    ap.add_argument('--output', default=_DEFAULT_OUT)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if os.path.isfile(args.output):
        print(f'[download_clip] already present: {args.output}')
        return

    print(f'[download_clip] fetching {args.model_name} / {args.pretrained} ...')
    model, _, _ = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained
    )
    state = model.state_dict()
    torch.save(state, args.output)
    print(f'[download_clip] saved -> {args.output}  ({len(state)} tensors)')


if __name__ == '__main__':
    main()
