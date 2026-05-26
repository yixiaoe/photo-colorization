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
    assert mask.shape == (2, 77)


def test_clip_encoder_frozen(device, dummy_text):
    enc = CLIPTextEncoder(device=device)
    tokens, mask = enc.encode(dummy_text)
    assert not tokens.requires_grad


def test_clip_encoder_padding_mask(device):
    enc = CLIPTextEncoder(device=device)
    _, mask = enc.encode(["hi"])
    # "hi" is short, most positions should be PAD (True)
    assert mask[0, -1].item() is True
    # first token (SOS) should not be PAD
    assert mask[0, 0].item() is False
