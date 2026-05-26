# Phase 3: CLIP 文本引导上色 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 Phase 2 全图骨干（InstanceGenerator），在解码器注入 CLIP Cross-Attention，实现通过自然语言文本精确控制灰度图上色结果。

**Architecture:** 冻结 InstanceGenerator 全部权重（从 Phase 2 stage-full 加载），在 conv8_3 和 conv9_3 后插入可训练的 TextCrossAttentionBlock。CLIP ViT-B/32 text encoder 冻结提取 token-level features，经 Linear 投影后作为 Cross-Attention 的 K/V。训练仅更新 ~2.5M 新增参数。

**Tech Stack:** PyTorch 2.0+, open_clip_torch (ViT-B/32), COCO2017 + captions_train2017.json, pytest

**约束:** 不修改任何 Phase 1/Phase 2 已有文件。所有代码为新增文件。

---

## 文件结构（全部新增）

```
code/
├── models/
│   ├── text_color_networks.py       # TextCrossAttentionBlock, ClipTextColorGenerator
│   └── text_color_model.py          # TextColorModel (训练/推理)
├── options/
│   └── phase3_options.py            # Phase3TrainOptions, Phase3TestOptions
├── data_process/
│   └── text_color_dataset.py        # CocoCaptionDataset
├── util/
│   └── clip_encoder.py              # CLIPTextEncoder wrapper
├── tests/
│   └── phase3/
│       ├── conftest.py              # pytest fixtures
│       ├── test_clip_encoder.py
│       ├── test_cross_attention.py
│       ├── test_generator.py
│       ├── test_dataset.py
│       └── test_model.py
├── train_phase3.py                  # 训练入口
├── test_phase3.py                   # 推理入口（文本控制可视化）
└── scripts/
    ├── train_phase3.sh
    └── test_phase3.sh
```

---

## 依赖

```
open_clip_torch>=2.20.0
pytest>=7.0
```

---

### Task 1: CLIP Text Encoder Wrapper

**Files:**
- Create: `code/util/clip_encoder.py`
- Test: `code/tests/phase3/test_clip_encoder.py`
- Create: `code/tests/__init__.py`
- Create: `code/tests/phase3/__init__.py`
- Create: `code/tests/phase3/conftest.py`

- [ ] **Step 1: Write conftest.py with shared fixtures**

```python
# code/tests/phase3/conftest.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import torch

@pytest.fixture
def device():
    return torch.device('cpu')

@pytest.fixture
def dummy_text():
    return ["a red car under blue sky"]

@pytest.fixture
def batch_texts():
    return ["a red car under blue sky", "a golden dog on green grass"]
```

- [ ] **Step 2: Write failing test for CLIPTextEncoder**

```python
# code/tests/phase3/test_clip_encoder.py
import torch
from util.clip_encoder import CLIPTextEncoder

def test_clip_encoder_output_shape(device, dummy_text):
    enc = CLIPTextEncoder(device=device)
    tokens, mask = enc.encode(dummy_text)
    assert tokens.shape == (1, 77, 512)
    assert mask.shape == (1, 77)
    assert mask.dtype == torch.bool

def test_clip_encoder_batch(device, batch_texts):
    enc = CLIPTextEncoder(device=device)
    tokens, mask = enc.encode(batch_texts)
    assert tokens.shape == (2, 77, 512)

def test_clip_encoder_frozen(device, dummy_text):
    enc = CLIPTextEncoder(device=device)
    tokens, mask = enc.encode(dummy_text)
    assert not tokens.requires_grad
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd code && python -m pytest tests/phase3/test_clip_encoder.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'util.clip_encoder'"

- [ ] **Step 4: Implement CLIPTextEncoder**

```python
# code/util/clip_encoder.py
import torch
import open_clip


class CLIPTextEncoder:
    """Frozen CLIP ViT-B/32 text encoder. Extracts token-level features."""

    def __init__(self, arch='ViT-B-32', pretrained='openai', device='cpu'):
        self.device = torch.device(device)
        model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained)
        model.eval()
        self.model = model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(arch)
        # freeze
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts):
        """
        Args:
            texts: list of str, length N
        Returns:
            token_features: (N, 77, 512) float32
            padding_mask:   (N, 77) bool — True for PAD positions
        """
        tokens = self.tokenizer(texts).to(self.device)  # (N, 77)
        # extract token-level features from CLIP text transformer
        x = self.model.token_embedding(tokens)
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # (77, N, 512)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # (N, 77, 512)
        x = self.model.ln_final(x)
        # padding mask: CLIP pads with 0, EOS is 49407
        padding_mask = (tokens == 0)
        return x.float(), padding_mask
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd code && python -m pytest tests/phase3/test_clip_encoder.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add code/util/clip_encoder.py code/tests/ 
git commit -m "feat(phase3): add CLIPTextEncoder wrapper with token-level extraction"
```

---

<!-- TASK2_CONTINUE -->

