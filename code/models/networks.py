"""Network architecture definitions."""
import torch
import torch.nn as nn


# ── helpers ───────────────────────────────────────────────────────────────────

def _conv_block(in_ch, out_ch, kernel=3, stride=1, pad=1, dilation=1, bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                        padding=pad * dilation, dilation=dilation, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return layers


# ── Phase 1 ───────────────────────────────────────────────────────────────────

class CnnColorGenerator(nn.Module):
    """
    Zhang et al. 2016 colorization network (PyTorch re-implementation).
    Input:  (N, 1, H, W)  – L channel, normalised to [-1, 1]
    Output: (N, 313, H, W) – raw logits over ab colour bins (1/4 spatial res)

    Architecture mirrors colorization_train_val_v2.prototxt:
      conv1 (64)  stride-2
      conv2 (128) stride-2
      conv3 (256) stride-2
      conv4 (512) dilation-1
      conv5 (512) dilation-2
      conv6 (512) dilation-2
      conv7 (512) dilation-1
      conv8 (256) deconv-2 → 1/4 of input res
      pred  (313) 1×1 conv
    """

    def __init__(self):
        super().__init__()

        # conv1  H → H/2
        self.model1 = nn.Sequential(
            *_conv_block(1,   64,  stride=1),
            *_conv_block(64,  64,  stride=2),
        )
        # conv2  H/2 → H/4
        self.model2 = nn.Sequential(
            *_conv_block(64,  128, stride=1),
            *_conv_block(128, 128, stride=2),
        )
        # conv3  H/4 → H/8
        self.model3 = nn.Sequential(
            *_conv_block(128, 256, stride=1),
            *_conv_block(256, 256, stride=1),
            *_conv_block(256, 256, stride=2),
        )
        # conv4  H/8, dilation 1
        self.model4 = nn.Sequential(
            *_conv_block(256, 512, stride=1, dilation=1),
            *_conv_block(512, 512, stride=1, dilation=1),
            *_conv_block(512, 512, stride=1, dilation=1),
        )
        # conv5  H/8, dilation 2
        self.model5 = nn.Sequential(
            *_conv_block(512, 512, stride=1, dilation=2),
            *_conv_block(512, 512, stride=1, dilation=2),
            *_conv_block(512, 512, stride=1, dilation=2),
        )
        # conv6  H/8, dilation 2
        self.model6 = nn.Sequential(
            *_conv_block(512, 512, stride=1, dilation=2),
            *_conv_block(512, 512, stride=1, dilation=2),
            *_conv_block(512, 512, stride=1, dilation=2),
        )
        # conv7  H/8, dilation 1
        self.model7 = nn.Sequential(
            *_conv_block(512, 512, stride=1, dilation=1),
            *_conv_block(512, 512, stride=1, dilation=1),
            *_conv_block(512, 512, stride=1, dilation=1),
        )
        # conv8  H/8 → H/4 (deconv)
        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            *_conv_block(256, 256, stride=1),
            *_conv_block(256, 256, stride=1),
        )
        # prediction head  → (N, 313, H/4, W/4)
        self.model_out = nn.Conv2d(256, 313, kernel_size=1)

    def forward(self, x):
        # x: (N, 1, H, W)
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        x = self.model4(x)
        x = self.model5(x)
        x = self.model6(x)
        x = self.model7(x)
        x = self.model8(x)
        return self.model_out(x)   # (N, 313, H/4, W/4)

    def get_features(self, x):
        """Return deep features before the prediction head (for Phase 3 ExemplarAttention)."""
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        x = self.model4(x)
        x = self.model5(x)
        x = self.model6(x)
        x = self.model7(x)
        feat = self.model8(x)
        return feat                # (N, 256, H/4, W/4)

    def predict_from_features(self, feat):
        return self.model_out(feat)


# ── Phase 2 placeholders (Task-08) ───────────────────────────────────────────

class InstFusionGenerator(nn.Module):
    """Dual-branch backbone — implemented in Task-08."""
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Task-08")


class FusionGenerator(nn.Module):
    """3-conv fusion weight predictor — implemented in Task-08."""
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Task-08")


# ── Phase 3 placeholders (Task-11) ───────────────────────────────────────────

class ExemplarAttention(nn.Module):
    """Cross-Attention colour transfer — implemented in Task-11."""
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Task-11")


class StyleHarmonizer(nn.Module):
    """Inter-branch Cross-Attention (--harmonize) — implemented in Task-11."""
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Task-11")


# ── factory ───────────────────────────────────────────────────────────────────

def define_G(opt):
    """Return the generator network for the chosen method."""
    if opt.method == 'cnn_color':
        return CnnColorGenerator()
    raise NotImplementedError(f"define_G: unknown method '{opt.method}' (Task-08 for inst_fusion)")
