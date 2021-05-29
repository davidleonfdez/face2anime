from face2anime.misc import FeaturesCalculator
from face2anime.transforms import normalize_imagenet_tf
from fastai.vision.all import Lambda, Normalize, TensorImage
import torch
import torch.nn as nn


class FakeVgg(nn.Module):
    def __init__(self, layers): 
        super().__init__()
        self.features = nn.Sequential(*layers)
    def forward(self, x): return self.features(x)


def test_features_calculator():
    l = Lambda(lambda t: t * 2)

    fake_vgg = FakeVgg([Lambda(lambda t: t * 2) for _ in range(4)])
    style_layers = [1, 2]
    content_layers = [2, 3]
    input_norm_tf = Normalize.from_stats(torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5]))
    ftrs_calc = FeaturesCalculator(style_layers, content_layers, fake_vgg, 
                                   input_norm_tf=input_norm_tf)
    in_t = TensorImage(torch.Tensor([[[[0., 0.5]]*2]*3]*2))
    norm_in_t = input_norm_tf(in_t)
    norm_imagenet_in_t = normalize_imagenet_tf(in_t)

    style_ftrs = ftrs_calc.calc_style(norm_in_t)
    content_ftrs = ftrs_calc.calc_content(norm_in_t)
    style_ftrs2, content_ftrs2 = ftrs_calc.calc_style_and_content(norm_in_t)

    expected_style_ftrs = [norm_imagenet_in_t * 4, norm_imagenet_in_t * 8]
    expected_content_ftrs = [norm_imagenet_in_t * 8, norm_imagenet_in_t * 16]

    for sf in (style_ftrs, style_ftrs2):
        assert len(sf) == len(expected_style_ftrs)
        assert [torch.allclose(act, exp) for act, exp in zip(sf, expected_style_ftrs)] == [True]*len(sf)
    for cf in (content_ftrs, content_ftrs2):
        assert len(cf) == len(expected_content_ftrs)
        assert [torch.allclose(act, exp) for act, exp in zip(cf, expected_content_ftrs)] == [True]*len(cf)
