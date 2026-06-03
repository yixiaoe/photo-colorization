"""
Precompute CLIP text embeddings for every caption template used in Phase 3
training and store them as a single .pt file. Saves ~30-60h over the full
30+30-epoch schedule because the dataset/model can skip the CLIP forward
pass and just look up the (512,)-vector by string key.

Caption templates covered:
  - Instance:  "a <color> <class_name>" + 4 alt templates per pair
  - Background: ~10 hand-written scene strings + colour variants
  - Empty:     "" (encoded with the CLIP <pad> embedding)

Usage (run from code/, after download_clip.py succeeded):
  python scripts/cache_clip_embeddings.py \
      --out_file data/clip_text_cache.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.clip_encoder import CLIPTextEncoder
from scripts.build_phase3_jsonl import ALL_COLORS, CLASS_NAMES


# Instance caption templates (apply per (color, class) pair). "{c}" = color,
# "{o}" = object name. Keep this list small; the dataset randomly picks one
# per sample at training time but we pre-encode every possibility.
INST_TEMPLATES = [
    'a {c} {o}',
    'a photo of a {c} {o}',
    'the {o} is {c}',
    '{c} {o}',
    'a {c} coloured {o}',
]

# Background caption pool (used for the bg_adapter). Cover a few axes:
# indoor/outdoor, time-of-day, weather.
BG_CAPTIONS = [
    '',                       # empty fallback
    'outdoor scene',
    'indoor room',
    'a sunny day',
    'a cloudy day',
    'at sunset',
    'at night',
    'at noon',
    'in the morning',
    'a dark scene',
    'a bright scene',
    'a colourful background',
    'a grey background',
    'a green grass field',
    'a blue sky',
    'a snowy landscape',
    'a sandy beach',
    'a wooden floor',
    'a white wall',
    'a city street',
    'a forest',
]


def collect_captions():
    captions = set()

    # instance combinations
    classes = sorted(set(CLASS_NAMES.values()))
    for o in classes:
        captions.add(f'a {o}')           # class-name fallback
        for c in ALL_COLORS:
            for tpl in INST_TEMPLATES:
                captions.add(tpl.format(c=c, o=o))

    for s in BG_CAPTIONS:
        captions.add(s)

    return sorted(captions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_file', default='data/clip_text_cache.pt')
    ap.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch_size', type=int, default=256)
    args = ap.parse_args()

    captions = collect_captions()
    print(f'[cache] {len(captions)} unique captions to encode')

    enc = CLIPTextEncoder(device=args.device)

    cache: dict = {}
    for i in range(0, len(captions), args.batch_size):
        chunk = captions[i:i + args.batch_size]
        embs = enc.encode(chunk)              # (B, 512), on device
        for s, e in zip(chunk, embs):
            cache[s] = e.detach().cpu().float()
        if (i // args.batch_size) % 10 == 0:
            print(f'  encoded {i + len(chunk)}/{len(captions)}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)) or '.', exist_ok=True)
    torch.save(cache, args.out_file)
    print(f'[cache] saved -> {args.out_file}  ({len(cache)} entries, '
          f'~{os.path.getsize(args.out_file)/1024/1024:.1f} MB)')


if __name__ == '__main__':
    main()
