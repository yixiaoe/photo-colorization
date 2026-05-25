"""Network architecture definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ── Phase 2 ───────────────────────────────────────────────────────────────────

class InstanceGenerator(nn.Module):
    """
    Su et al. CVPR 2020 colorization backbone.
    Shared architecture for full-image branch and instance branch.

    Input:  (N, 1, H, W) – L channel only
    Output: (out_class, out_reg, feature_map)
      out_class:   (N, 313, H/4, W/4) – raw classification logits
      out_reg:     (N, 2,   H,   W)   – Tanh ab regression
      feature_map: dict of intermediate tensors keyed by layer name

    Downsampling is done via strided-slicing ([:,:,::2,::2]) to match the
    reference implementation; upsampling uses ConvTranspose2d.
    """

    def __init__(self):
        super().__init__()
        B = True  # bias

        # ── Encoder ──────────────────────────────────────────────────────
        # conv1: (N,1,H,W) → (N,64,H,W), then ::2 → H/2
        self.model1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(64),
        )
        # conv2: (N,64,H/2,W/2) → (N,128,H/2,W/2), then ::2 → H/4
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(128),
        )
        # conv3: (N,128,H/4,W/4) → (N,256,H/4,W/4), then ::2 → H/8
        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        # conv4: (N,256,H/8,W/8) → (N,512,H/8,W/8), dilation=1
        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        # conv5: dilation=2
        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        # conv6: dilation=2
        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        # conv7: dilation=1
        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(512),
        )

        # ── Decoder ──────────────────────────────────────────────────────
        # conv8: H/8 → H/4 via deconv
        self.model8up = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=B)
        self.model3short8 = nn.Conv2d(256, 256, 3, 1, 1, bias=B)
        self.model8 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(256),
        )

        # conv9: H/4 → H/2
        self.model9up = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=B)
        self.model2short9 = nn.Conv2d(128, 128, 3, 1, 1, bias=B)
        self.model9 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(128),
        )

        # conv10: H/2 → H
        self.model10up = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=B)
        self.model1short10 = nn.Conv2d(64, 128, 3, 1, 1, bias=B)
        self.model10 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=B),
            nn.LeakyReLU(0.2),
        )

        # ── Output heads ─────────────────────────────────────────────────
        self.model_class = nn.Conv2d(256, 313, kernel_size=1, bias=B)  # from conv8_3, H/4
        self.model_out = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1, bias=B),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        x: (N, 1, H, W)
        Returns (out_class, out_reg, feature_map).
        """
        # Encoder
        conv1_2 = self.model1(x)                           # (N, 64,  H,   W)
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])     # (N, 128, H/2, W/2)
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])     # (N, 256, H/4, W/4)
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])     # (N, 512, H/8, W/8)
        conv5_3 = self.model5(conv4_3)                     # (N, 512, H/8, W/8)
        conv6_3 = self.model6(conv5_3)                     # (N, 512, H/8, W/8)
        conv7_3 = self.model7(conv6_3)                     # (N, 512, H/8, W/8)

        # Decoder with skip connections
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)  # (N, 256, H/4, W/4)
        conv8_3  = self.model8(conv8_up)                                 # (N, 256, H/4, W/4)

        out_class = self.model_class(conv8_3)                            # (N, 313, H/4, W/4)

        conv9_up  = self.model9up(conv8_3) + self.model2short9(conv2_2)  # (N, 128, H/2, W/2)
        conv9_3   = self.model9(conv9_up)                                 # (N, 128, H/2, W/2)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2) # (N, 128, H,   W)
        conv10_2  = self.model10(conv10_up)                               # (N, 128, H,   W)

        out_reg = self.model_out(conv10_2)                                # (N, 2,   H,   W)

        feature_map = {
            'conv1_2':  conv1_2,  'conv2_2':  conv2_2,  'conv3_3':  conv3_3,
            'conv4_3':  conv4_3,  'conv5_3':  conv5_3,  'conv6_3':  conv6_3,
            'conv7_3':  conv7_3,  'conv8_up': conv8_up, 'conv8_3':  conv8_3,
            'conv9_up': conv9_up, 'conv9_3':  conv9_3,
            'conv10_up': conv10_up, 'conv10_2': conv10_2,
        }
        return out_class, out_reg, feature_map


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) conditioned on a class embedding.

    feat * (1 + γ) + β  — residual form ensures identity initialisation.

    Args:
        embed_dim:    dimension of the incoming class embedding vector
        num_channels: C of the feature map to be modulated
    """

    def __init__(self, embed_dim: int, num_channels: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 2 * num_channels)
        # zero-init so at start γ=0, β=0 → identity
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, feat, emb):
        """
        feat: (N, C, H, W)
        emb:  (N, embed_dim)
        """
        params = self.fc(emb)                   # (N, 2C)
        gamma, beta = params.chunk(2, dim=1)    # (N, C) each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + gamma) + beta


class FiLMInstanceGenerator(InstanceGenerator):
    """
    Instance-branch backbone: InstanceGenerator + FiLM conditioning at conv4–conv7.

    The class label is embedded via label_embedding and injected after each of
    model4, model5, model6, model7 using residual FiLM (1+γ form).

    Args:
        num_classes: size of the label vocabulary (91 covers COCO 1–90)
        embed_dim:   dimensionality of the class embedding
    """

    def __init__(self, num_classes: int = 91, embed_dim: int = 64):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.film4 = FiLMLayer(embed_dim, 512)
        self.film5 = FiLMLayer(embed_dim, 512)
        self.film6 = FiLMLayer(embed_dim, 512)
        self.film7 = FiLMLayer(embed_dim, 512)

    def forward(self, x, class_label):
        """
        x:           (N, 1, H, W)
        class_label: (N,) int64
        Returns (out_class, out_reg, feature_map) – same contract as InstanceGenerator.
        """
        emb = self.label_embedding(class_label)  # (N, embed_dim)

        # Encoder with FiLM injection at conv4–conv7
        conv1_2 = self.model1(x)
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.film4(self.model4(conv3_3[:, :, ::2, ::2]), emb)
        conv5_3 = self.film5(self.model5(conv4_3), emb)
        conv6_3 = self.film6(self.model6(conv5_3), emb)
        conv7_3 = self.film7(self.model7(conv6_3), emb)

        # Decoder
        conv8_up  = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3   = self.model8(conv8_up)
        out_class = self.model_class(conv8_3)

        conv9_up  = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3   = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2  = self.model10(conv10_up)
        out_reg   = self.model_out(conv10_2)

        feature_map = {
            'conv1_2':  conv1_2,  'conv2_2':  conv2_2,  'conv3_3':  conv3_3,
            'conv4_3':  conv4_3,  'conv5_3':  conv5_3,  'conv6_3':  conv6_3,
            'conv7_3':  conv7_3,  'conv8_up': conv8_up, 'conv8_3':  conv8_3,
            'conv9_up': conv9_up, 'conv9_3':  conv9_3,
            'conv10_up': conv10_up, 'conv10_2': conv10_2,
        }
        return out_class, out_reg, feature_map

    def get_backbone_params(self):
        """Parameters of the InstanceGenerator backbone (excluding FiLM)."""
        film_names = {'label_embedding', 'film4', 'film5', 'film6', 'film7'}
        return [p for n, p in self.named_parameters()
                if n.split('.')[0] not in film_names]

    def get_film_params(self):
        """Parameters of the FiLM layers + embedding."""
        film_names = {'label_embedding', 'film4', 'film5', 'film6', 'film7'}
        return [p for n, p in self.named_parameters()
                if n.split('.')[0] in film_names]


class WeightGenerator(nn.Module):
    """
    Soft-attention fusion module (Su et al. CVPR 2020).

    Predicts per-pixel soft weights for each instance feature map and the
    background (full-image) feature map, then returns the weighted sum.

    Args:
        input_ch: number of channels at this feature level
        inner_ch: hidden channels in the small weight-prediction CNNs
    """

    def __init__(self, input_ch: int, inner_ch: int = 16):
        super().__init__()
        self.instance_conv = nn.Sequential(
            nn.Conv2d(input_ch, inner_ch, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(inner_ch, inner_ch, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(inner_ch, 1, 3, 1, 1), nn.ReLU(True),
        )
        self.bg_conv = nn.Sequential(
            nn.Conv2d(input_ch, inner_ch, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(inner_ch, inner_ch, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(inner_ch, 1, 3, 1, 1), nn.ReLU(True),
        )
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def _resize_and_pad(feat, info):
        """
        Resize instance feature to the bbox region, then pad to full map size.

        info: [left_pad, right_pad, top_pad, bot_pad, box_w, box_h]
              coordinates are already scaled to the current feature resolution.
        """
        feat = F.interpolate(feat, size=(info[5], info[4]), mode='bilinear',
                             align_corners=False)
        feat = F.pad(feat, (info[0], info[1], info[2], info[3]), value=0.)
        return feat

    def forward(self, instance_feat, bg_feat, box_info):
        """
        instance_feat: (N, C, h, w) – N instance features stacked
        bg_feat:       (1, C, h, w) – full-image feature
        box_info:      (N, 6)       – bbox info at this feature scale

        Returns fused feature (1, C, h, w).
        """
        N = instance_feat.shape[0]
        covered = torch.zeros_like(bg_feat)[:1, :1]  # (1,1,h,w)

        # batch all instance_conv calls into one GPU kernel launch (N, 1, h, w)
        all_inst_weights = self.instance_conv(instance_feat)   # (N, 1, h, w)

        mask_weights = []
        feat_maps    = []
        for i in range(N):
            info = box_info[i]
            weight_i = self._resize_and_pad(all_inst_weights[[i]], info)  # (1,1,h,w)
            feat_i   = self._resize_and_pad(instance_feat[[i]], info)     # (1,C,h,w)

            mask = torch.zeros_like(bg_feat)[:1, :1]
            r0, c0 = int(info[2]), int(info[0])
            rh, rw = int(info[5]), int(info[4])
            mask[0, 0, r0:r0+rh, c0:c0+rw] = 1.0
            covered = torch.clamp(covered + mask, 0., 1.)

            mask_weights.append(weight_i)
            feat_maps.append(feat_i)

        bg_weight = self.bg_conv(bg_feat)                      # (1,1,h,w)
        bg_weight = bg_weight + (1.0 - covered) * 1e5
        mask_weights.append(bg_weight)
        feat_maps.append(bg_feat)

        weights = self.softmax(torch.cat(mask_weights, dim=1))  # (1, N+1, h, w)
        feat_stack = torch.cat(feat_maps, dim=0)                # (N+1, C, h, w)
        weights_t  = weights.permute(1, 0, 2, 3)               # (N+1, 1, h, w)
        fused = (feat_stack * weights_t).sum(dim=0, keepdim=True)  # (1, C, h, w)
        return fused


class FusionPipeline(nn.Module):
    """
    Stage-3 full pipeline: frozen backbone (from stage-1) + trainable
    WeightGenerators (×13) + trainable output_conv.

    The backbone runs on the full gray image; the WeightGenerators fuse its
    per-layer features with instance features from FiLMInstanceGenerator.
    When no instances are present the backbone output flows unchanged to
    output_conv.

    Call freeze_backbone() after loading stage-1 weights to lock the backbone.
    """

    def __init__(self):
        super().__init__()
        B = True

        # ── Backbone (identical to InstanceGenerator, no heads) ──────────
        self.model1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(64),
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(128),
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.model8up     = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=B)
        self.model3short8 = nn.Conv2d(256, 256, 3, 1, 1, bias=B)
        self.model8 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        self.model9up     = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=B)
        self.model2short9 = nn.Conv2d(128, 128, 3, 1, 1, bias=B)
        self.model9 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=B), nn.ReLU(True),
            nn.BatchNorm2d(128),
        )
        self.model10up     = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=B)
        self.model1short10 = nn.Conv2d(64, 128, 3, 1, 1, bias=B)
        self.model10 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=B),
            nn.LeakyReLU(0.2),
        )

        # ── WeightGenerators – one per feature-level ─────────────────────
        # Scale 0 (H×W): conv1_2, conv10_up, conv10_2
        self.wg_conv1_2  = WeightGenerator(64)
        self.wg_conv10_up = WeightGenerator(128)
        self.wg_conv10_2 = WeightGenerator(128)
        # Scale 1 (H/2×W/2): conv2_2, conv9_up, conv9_3
        self.wg_conv2_2  = WeightGenerator(128)
        self.wg_conv9_up = WeightGenerator(128)
        self.wg_conv9_3  = WeightGenerator(128)
        # Scale 2 (H/4×W/4): conv3_3, conv8_up, conv8_3
        self.wg_conv3_3  = WeightGenerator(256)
        self.wg_conv8_up = WeightGenerator(256)
        self.wg_conv8_3  = WeightGenerator(256)
        # Scale 3 (H/8×W/8): conv4_3 … conv7_3
        self.wg_conv4_3  = WeightGenerator(512)
        self.wg_conv5_3  = WeightGenerator(512)
        self.wg_conv6_3  = WeightGenerator(512)
        self.wg_conv7_3  = WeightGenerator(512)

        # ── Trainable output head ─────────────────────────────────────────
        self.output_conv = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1, bias=B),
            nn.Tanh(),
        )

    # ── weight helpers ────────────────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze all backbone parameters (keep WeightGenerators + output_conv trainable)."""
        trainable = {'wg_conv1_2', 'wg_conv2_2', 'wg_conv3_3',
                     'wg_conv4_3', 'wg_conv5_3', 'wg_conv6_3', 'wg_conv7_3',
                     'wg_conv8_up', 'wg_conv8_3', 'wg_conv9_up', 'wg_conv9_3',
                     'wg_conv10_up', 'wg_conv10_2', 'output_conv'}
        for name, param in self.named_parameters():
            param.requires_grad_(name.split('.')[0] in trainable)

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def load_backbone_weights(self, state_dict):
        """Load InstanceGenerator weights into the backbone (strict=False)."""
        self.load_state_dict(state_dict, strict=False)

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, gray_full, inst_feats_list, box_info_list):
        """
        gray_full:       (1, 1, H, W)
        inst_feats_list: list of N feature_map dicts from FiLMInstanceGenerator
        box_info_list:   [bi_H(N,6), bi_H2(N,6), bi_H4(N,6), bi_H8(N,6)]
                         or None/empty list when no instances.

        Returns fused_ab (1, 2, H, W) with Tanh.
        """
        N = len(inst_feats_list)
        has_inst = N > 0

        if has_inst:
            # Stack per-layer features across all instances: (N, C, h, w)
            def _stack(key):
                return torch.cat([f[key] for f in inst_feats_list], dim=0)

            bi0, bi1, bi2, bi3 = box_info_list  # (N,6) at H, H/2, H/4, H/8

        def _fuse(wg, feat, inst_key, bi):
            if has_inst:
                return wg(_stack(inst_key), feat, bi)
            return feat

        # ── Encoder ──────────────────────────────────────────────────────
        conv1_2 = self.model1(gray_full)
        conv1_2 = _fuse(self.wg_conv1_2, conv1_2, 'conv1_2', bi0 if has_inst else None)

        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv2_2 = _fuse(self.wg_conv2_2, conv2_2, 'conv2_2', bi1 if has_inst else None)

        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv3_3 = _fuse(self.wg_conv3_3, conv3_3, 'conv3_3', bi2 if has_inst else None)

        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv4_3 = _fuse(self.wg_conv4_3, conv4_3, 'conv4_3', bi3 if has_inst else None)

        conv5_3 = self.model5(conv4_3)
        conv5_3 = _fuse(self.wg_conv5_3, conv5_3, 'conv5_3', bi3 if has_inst else None)

        conv6_3 = self.model6(conv5_3)
        conv6_3 = _fuse(self.wg_conv6_3, conv6_3, 'conv6_3', bi3 if has_inst else None)

        conv7_3 = self.model7(conv6_3)
        conv7_3 = _fuse(self.wg_conv7_3, conv7_3, 'conv7_3', bi3 if has_inst else None)

        # ── Decoder ──────────────────────────────────────────────────────
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_up = _fuse(self.wg_conv8_up, conv8_up, 'conv8_up', bi2 if has_inst else None)

        conv8_3 = self.model8(conv8_up)
        conv8_3 = _fuse(self.wg_conv8_3, conv8_3, 'conv8_3', bi2 if has_inst else None)

        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_up = _fuse(self.wg_conv9_up, conv9_up, 'conv9_up', bi1 if has_inst else None)

        conv9_3 = self.model9(conv9_up)
        conv9_3 = _fuse(self.wg_conv9_3, conv9_3, 'conv9_3', bi1 if has_inst else None)

        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_up = _fuse(self.wg_conv10_up, conv10_up, 'conv10_up', bi0 if has_inst else None)

        conv10_2 = self.model10(conv10_up)
        conv10_2 = _fuse(self.wg_conv10_2, conv10_2, 'conv10_2', bi0 if has_inst else None)

        return self.output_conv(conv10_2)   # (1, 2, H, W)


# ── Phase 3 placeholders ────────────────────────────────────────────

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
    """Return the generator network for the chosen method and stage."""
    if opt.method == 'cnn_color':
        return CnnColorGenerator()
    if opt.method == 'inst_fusion':
        stage = getattr(opt, 'stage', 'full')
        if stage == 'full':
            return InstanceGenerator()
        if stage == 'instance':
            num_classes = getattr(opt, 'num_classes', 91)
            embed_dim   = getattr(opt, 'embed_dim', 64)
            return FiLMInstanceGenerator(num_classes=num_classes, embed_dim=embed_dim)
        if stage == 'fusion':
            return FusionPipeline()
    raise NotImplementedError(f"define_G: unknown method/stage '{opt.method}/{getattr(opt,'stage','?')}'")
