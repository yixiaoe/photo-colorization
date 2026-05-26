"""
Phase 3 pytest 公共 fixtures。

提供测试中复用的基础设施：
  - device: CPU torch.device（单测不依赖 GPU）
  - dummy_text: 单条文本输入
  - batch_texts: 多条文本输入（测试 batch 处理）
"""
import sys
import os
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
