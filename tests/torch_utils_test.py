from face2anime.layers import ConvX2UpsamplingOp2d
from face2anime.networks import SkipGenerator
from face2anime.torch_utils import *
import torch.nn as nn
from torch.nn.utils import spectral_norm


def test_is_conv():
    conv2d = nn.Conv2d(1, 3, 3)

    assert is_conv(nn.Conv1d(4, 4, 2))
    assert is_conv(nn.ConvTranspose1d(2, 2, 4))
    assert is_conv(conv2d)
    assert is_conv(nn.ConvTranspose2d(3, 1, 3))
    assert is_conv(nn.Conv3d(1, 2, 1))
    assert is_conv(nn.ConvTranspose3d(1, 2, 1))
    assert not is_conv(nn.Sequential(conv2d))
    assert not is_conv(nn.ModuleList([conv2d]))


def test_has_sn_hook():
    assert has_sn_hook(spectral_norm(nn.Conv2d(2, 2, 1)))
    assert has_sn_hook(spectral_norm(nn.Linear(2, 4)))
    assert not has_sn_hook(nn.Conv2d(2, 2, 1))
    assert not has_sn_hook(nn.Linear(2, 4))
    assert not has_sn_hook(nn.Sequential(spectral_norm(nn.Conv2d(2, 2, 1))))


def test_every_conv_has_sn():
    net_one_hasnt = nn.Sequential(
        spectral_norm(nn.Conv2d(2, 2, 1)),
        nn.BatchNorm2d(2),
        nn.Sequential(
            spectral_norm(nn.Conv2d(2, 2, 1)),
            nn.Conv2d(2, 2, 1)
        )
    )
    net_all_have = nn.Sequential(
        spectral_norm(nn.Conv2d(2, 2, 1)),
        nn.BatchNorm2d(2),
        nn.Sequential(
            spectral_norm(nn.Conv2d(2, 2, 1)),
            nn.BatchNorm2d(2),
            spectral_norm(nn.Conv2d(2, 2, 1))
        )
    )

    class TmpNet(nn.Module):
        def __init__(self, sn_conv2=False, extra_layers=None):
            super().__init__()
            self.conv1 = spectral_norm(nn.Conv2d(2, 2, 1))
            self.bn = nn.BatchNorm2d(2)
            self.conv2 = nn.Conv2d(2, 2, 1)
            if sn_conv2: self.conv2 = spectral_norm(self.conv2)
            self.conv2 = nn.ModuleList([nn.Sequential(self.conv2)])
            self.extra_layers = extra_layers or nn.Identity()

        def forward(self, x):
            return self.extra_layers(self.conv2[0](self.bn(self.conv1(x))))

    assert every_conv_has_sn(spectral_norm(nn.Conv2d(2, 2, 1)))
    assert every_conv_has_sn(nn.Linear(2, 2))
    assert every_conv_has_sn(net_all_have)
    assert every_conv_has_sn(TmpNet(sn_conv2=True, extra_layers=net_all_have))
    assert not every_conv_has_sn(nn.Conv2d(2, 2, 1))
    assert not every_conv_has_sn(net_one_hasnt)
    assert not every_conv_has_sn(TmpNet(sn_conv2=False, extra_layers=net_all_have))
    assert not every_conv_has_sn(TmpNet(sn_conv2=True, extra_layers=net_one_hasnt))


def test_get_mean_weights():
    conv2d_w1 = nn.Conv2d(1, 2, 3)
    nn.init.constant_(conv2d_w1.weight, 1.)
    nn.init.constant_(conv2d_w1.bias, 2.)
    lin_w3 = nn.Linear(2, 2, bias=False)
    nn.init.constant_(lin_w3.weight, 3.)
    net = nn.Sequential(
        conv2d_w1,
        nn.Sequential(lin_w3)
    )

    expected_full = {
        '0.weight': 1.,
        '0.bias': 2.,
        '1.0.weight': 3.
    }
    expected_only_conv = {
        '0.weight': 1.,
        '0.bias': 2.,
    }

    assert get_mean_weights(net, nn.Conv2d) == expected_only_conv
    assert get_mean_weights(net, (nn.Conv2d, nn.Linear)) == expected_full


def test_add_sn():
    net = nn.Sequential(
        nn.Conv2d(1, 1, 1),
        nn.BatchNorm2d(1),
        nn.ReLU(),
        nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1))
    )
    add_sn(net)

    assert every_conv_has_sn(net)


def test_set_sn_to_every_conv():
    net = SkipGenerator(32, 3, ConvX2UpsamplingOp2d(), ConvX2UpsamplingOp2d(),
                        sn=False)
    set_sn_to_every_conv(net)

    assert every_conv_has_sn(net)
