"""
CLIP text-tower wrapper for Phase 3.

  - Loads open_clip ViT-B/32 (quickgelu) weights, preferring local checkpoint
    at code/checkpoints/clip/ViT-B-32-quickgelu.pt; falls back to the open_clip
    hub if absent.
  - Freezes everything; only the text path is used.
  - Optional in-memory + on-disk embedding cache.

Usage:
    enc = CLIPTextEncoder(device='cuda')
    embs = enc.encode(['a red car', 'a blue car'])    # (2, 512)
"""
import os
from typing import List, Optional

import torch

try:
    import open_clip
except ImportError:
    open_clip = None


_MODEL_NAME = 'ViT-B-32-quickgelu'
_LOCAL_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'checkpoints', 'clip', f'{_MODEL_NAME}.pt',
)


class CLIPTextEncoder:
    """
    Frozen CLIP text encoder. Lazy-loaded singleton (one model per device).
    """
    _instances = {}

    def __new__(cls, device='cuda', cache_path: Optional[str] = None):
        key = str(device)
        if key not in cls._instances:
            inst = super().__new__(cls)
            inst._init(device, cache_path)
            cls._instances[key] = inst
        return cls._instances[key]

    def _init(self, device, cache_path):
        if open_clip is None:
            raise ImportError(
                'open_clip_torch is required for CLIPTextEncoder. '
                'pip install open_clip_torch ftfy regex'
            )

        self.device = torch.device(device)

        # ── model load: prefer local ckpt ──────────────────────────────
        if os.path.isfile(_LOCAL_CKPT):
            model, _, _ = open_clip.create_model_and_transforms(
                _MODEL_NAME, pretrained=None
            )
            state = torch.load(_LOCAL_CKPT, map_location='cpu')
            model.load_state_dict(state)
            print(f'[CLIP] loaded local ckpt: {_LOCAL_CKPT}')
        else:
            model, _, _ = open_clip.create_model_and_transforms(
                _MODEL_NAME, pretrained='laion400m_e32'
            )
            print(f'[CLIP] downloaded {_MODEL_NAME} from open_clip hub')

        # drop the visual tower to save memory
        if hasattr(model, 'visual'):
            del model.visual

        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
        self.embed_dim = 512   # ViT-B/32 text projection

        # ── optional embedding cache ───────────────────────────────────
        self._cache: dict = {}
        self.cache_path = cache_path
        if cache_path and os.path.isfile(cache_path):
            blob = torch.load(cache_path, map_location='cpu')
            for k, v in blob.items():
                self._cache[k] = v.float()
            print(f'[CLIP] embedding cache loaded: {len(self._cache)} entries '
                  f'from {cache_path}')

    # ── public API ──────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        texts: list of strings (or a single string)
        returns: Tensor of shape (B, embed_dim) on self.device, float32
        """
        if isinstance(texts, str):
            texts = [texts]

        outs = [None] * len(texts)
        missing_idx, missing_txt = [], []
        for i, t in enumerate(texts):
            t = t or ''
            cached = self._cache.get(t)
            if cached is not None:
                outs[i] = cached.to(self.device)
            else:
                missing_idx.append(i)
                missing_txt.append(t)

        if missing_txt:
            tokens = self.tokenizer(missing_txt).to(self.device)
            feats = self.model.encode_text(tokens).float()
            for i, t, f in zip(missing_idx, missing_txt, feats):
                self._cache[t] = f.detach().cpu()
                outs[i] = f

        return torch.stack(outs, dim=0)

    def save_cache(self, path: Optional[str] = None):
        target = path or self.cache_path
        if target is None:
            raise ValueError('save_cache: no path given')
        os.makedirs(os.path.dirname(os.path.abspath(target)) or '.', exist_ok=True)
        torch.save({k: v.cpu() for k, v in self._cache.items()}, target)
        print(f'[CLIP] embedding cache saved: {len(self._cache)} entries -> {target}')
