"""Network architecture definitions for the colorization models."""
import functools

import torch
import torch.nn as nn


def get_norm_layer(norm_type='batch'):
    """Return a 2D normalisation layer factory."""
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == 'none':
        return None
    raise NotImplementedError(f'normalization layer [{norm_type}] is not found')


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialise convolution and normalisation weights."""
    def init_func(module):
        classname = module.__class__.__name__
        if hasattr(module, 'weight') and ('Conv' in classname or 'Linear' in classname):
            if init_type == 'normal':
                nn.init.normal_(module.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(module.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(module.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if getattr(module, 'bias', None) is not None:
                nn.init.constant_(module.bias.data, 0.0)
        elif 'BatchNorm2d' in classname and hasattr(module, 'weight'):
            nn.init.normal_(module.weight.data, 1.0, init_gain)
            nn.init.constant_(module.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=None):
    """Move a network to the requested device and initialise weights."""
    gpu_ids = gpu_ids or []
    if gpu_ids:
        assert torch.cuda.is_available(), 'CUDA was requested but is not available'
        net.to(torch.device(f'cuda:{gpu_ids[0]}'))
    init_weights(net, init_type)
    return net


class ConvNormReLU(nn.Module):
    """Conv2d + optional norm + ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        use_bias = norm_layer is None
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=use_bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        layers.append(nn.ReLU(True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Zhang2016Generator(nn.Module):
    """Zhang et al. 2016 global colorization network.

    Input:
        Normalised Lab L channel, shaped N x 1 x H x W.
    Output:
        Per-pixel color-bin logits, shaped N x 313 x H/4 x W/4.

    The topology follows the official Caffe deploy network: three early
    downsampling blocks, dilated semantic blocks, and one learned upsampling
    block before the final 313-way color classifier.
    """
    def __init__(self, input_nc=1, output_nc=313, ngf=64, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.model1 = nn.Sequential(
            ConvNormReLU(input_nc, ngf, 3, 1, 1, norm_layer=norm_layer),
            ConvNormReLU(ngf, ngf, 3, 2, 1, norm_layer=norm_layer),
        )
        self.model2 = nn.Sequential(
            ConvNormReLU(ngf, ngf * 2, 3, 1, 1, norm_layer=norm_layer),
            ConvNormReLU(ngf * 2, ngf * 2, 3, 2, 1, norm_layer=norm_layer),
        )
        self.model3 = nn.Sequential(
            ConvNormReLU(ngf * 2, ngf * 4, 3, 1, 1, norm_layer=norm_layer),
            ConvNormReLU(ngf * 4, ngf * 4, 3, 1, 1, norm_layer=norm_layer),
            ConvNormReLU(ngf * 4, ngf * 4, 3, 2, 1, norm_layer=norm_layer),
        )
        self.model4 = nn.Sequential(
            ConvNormReLU(ngf * 4, ngf * 8, 3, 1, 1, norm_layer=norm_layer),
            ConvNormReLU(ngf * 8, ngf * 8, 3, 1, 1, norm_layer=norm_layer),
            ConvNormReLU(ngf * 8, ngf * 8, 3, 1, 1, norm_layer=norm_layer),
        )
        self.model5 = nn.Sequential(
            ConvNormReLU(ngf * 8, ngf * 8, 3, 1, 2, dilation=2, norm_layer=norm_layer),
            ConvNormReLU(ngf * 8, ngf * 8, 3, 1, 2, dilation=2, norm_layer=norm_layer),
            ConvNormReLU(ngf * 8, ngf * 8, 3, 1, 2, dilation=2, norm_layer=norm_layer),
        )
        self.model6 = nn.Sequential(
            ConvNormReLU(ngf * 8, ngf * 8, 3, 1, 2, dilation=2, norm_layer=norm_layer),
            ConvNormReLU(ngf * 8, ngf * 8, 3, 1, 2, dilation=2, norm_layer=norm_layer),
            ConvNormReLU(ngf * 8, ngf * 8, 3, 1, 2, dilation=2, norm_layer=norm_layer),
        )
        self.model7 = nn.Sequential(
            ConvNormReLU(ngf * 8, ngf * 8, 3, 1, 1, norm_layer=norm_layer),
            ConvNormReLU(ngf * 8, ngf * 8, 3, 1, 1, norm_layer=norm_layer),
            ConvNormReLU(ngf * 8, ngf * 8, 3, 1, 1, norm_layer=norm_layer),
        )
        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ngf * 4) if norm_layer is not None else nn.Identity(),
            nn.ReLU(True),
            ConvNormReLU(ngf * 4, ngf * 4, 3, 1, 1, norm_layer=norm_layer),
            ConvNormReLU(ngf * 4, ngf * 4, 3, 1, 1, norm_layer=norm_layer),
        )
        self.classifier = nn.Conv2d(ngf * 4, output_nc, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        x = self.model4(x)
        x = self.model5(x)
        x = self.model6(x)
        x = self.model7(x)
        x = self.model8(x)
        return self.classifier(x)


def define_G(opt):
    """Return the generator network for the chosen method."""
    norm_layer = get_norm_layer(getattr(opt, 'norm', 'batch'))
    method = getattr(opt, 'method', 'zhang2016')
    if method == 'zhang2016':
        net = Zhang2016Generator(
            input_nc=1,
            output_nc=313,
            ngf=getattr(opt, 'ngf', 64),
            norm_layer=norm_layer,
        )
        return init_net(net, getattr(opt, 'init_type', 'normal'), getattr(opt, 'gpu_ids', []))
    if method == 'inst2020':
        raise NotImplementedError('inst2020 networks are scheduled for Task-08')
    raise NotImplementedError(f'unknown colorization method [{method}]')
