from face2anime.layers import (ConcatPoolHalfDownsamplingOp2d, CondConvX2UpsamplingOp2d, 
                               CondInterpConvUpsamplingOp2d, ConvHalfDownsamplingOp2d, 
                               ConvX2UpsamplingOp2d, InterpConvUpsamplingOp2d)
from face2anime.networks import (CondResGenerator, custom_generator, img2img_generator,
                                 NoiseSplitDontSplitStrategy, NoiseSplitEqualLeave1stOutStrategy, 
                                 res_critic, res_generator, SkipGenerator)
from face2anime.torch_utils import every_conv_has_sn
import pytest
from random import randint
import torch


def _g_out_sz_is_ok(g, n_ch, in_sz, out_sz):
    # Pick a random batch size to avoid using a fixed one
    bs = randint(1, 5)
    in_t = torch.rand(bs, in_sz)
    out = g(in_t)
    return out.size() == torch.Size([bs, n_ch, out_sz, out_sz])


@pytest.mark.parametrize("out_sz", [32, 64])
@pytest.mark.parametrize("n_ch", [1, 3])
@pytest.mark.parametrize("in_sz", [50, 100])
@pytest.mark.parametrize("n_ftrs", [16, 32])
def test_custom_generator(out_sz, n_ch, in_sz, n_ftrs):
    up_op = ConvX2UpsamplingOp2d()
    g = custom_generator(out_sz, n_ch, up_op, in_sz=in_sz, n_features=n_ftrs)

    assert every_conv_has_sn(g)
    assert _g_out_sz_is_ok(g, n_ch, in_sz, out_sz)


@pytest.mark.parametrize("out_sz", [32, 64])
@pytest.mark.parametrize("n_ch", [1, 3])
@pytest.mark.parametrize("in_sz", [50, 100])
@pytest.mark.parametrize("n_ftrs", [16, 32])
def test_res_generator(out_sz, n_ch, in_sz, n_ftrs):
    up_op = ConvX2UpsamplingOp2d()
    id_up_op = InterpConvUpsamplingOp2d()
    g = res_generator(out_sz, n_ch, up_op, id_up_op, in_sz=in_sz, n_features=n_ftrs,
                      n_extra_res_blocks=2, n_extra_convs_by_res_block=2)

    assert every_conv_has_sn(g)
    assert _g_out_sz_is_ok(g, n_ch, in_sz, out_sz)


@pytest.mark.parametrize("out_sz", [32, 64])
@pytest.mark.parametrize("n_ch", [1, 3])
@pytest.mark.parametrize("in_sz", [50, 100])
@pytest.mark.parametrize("n_ftrs", [16, 32])
def test_cond_res_generator(out_sz, n_ch, in_sz, n_ftrs):
    common_args = [out_sz, n_ch]
    kwargs = dict(in_sz=in_sz, n_features=n_ftrs)
    n_noise_splits = CondResGenerator.calc_n_upblocks(out_sz)    

    noise_split_strat = NoiseSplitEqualLeave1stOutStrategy()
    cond_sz = noise_split_strat.calc_cond_sz(in_sz, n_noise_splits)
    g_split_equal_z = CondResGenerator(*common_args,
                                       CondConvX2UpsamplingOp2d(cond_sz),
                                       CondInterpConvUpsamplingOp2d(cond_sz),
                                       noise_split_strat,
                                       **kwargs)

    noise_split_strat = NoiseSplitDontSplitStrategy()
    cond_sz = noise_split_strat.calc_cond_sz(in_sz, n_noise_splits)
    g_not_split_z = CondResGenerator(*common_args,
                                     CondConvX2UpsamplingOp2d(cond_sz),
                                     CondInterpConvUpsamplingOp2d(cond_sz),
                                     noise_split_strat,
                                     **kwargs)

    assert every_conv_has_sn(g_split_equal_z)
    assert _g_out_sz_is_ok(g_split_equal_z, n_ch, in_sz, out_sz)
    assert _g_out_sz_is_ok(g_not_split_z, n_ch, in_sz, out_sz)


@pytest.mark.parametrize("out_sz", [32, 64])
@pytest.mark.parametrize("n_ch", [1, 3])
@pytest.mark.parametrize("in_sz", [50, 100])
@pytest.mark.parametrize("n_ftrs", [16, 32])
def test_skip_generator(out_sz, n_ch, in_sz, n_ftrs):
    up_op = ConvX2UpsamplingOp2d()
    id_up_op = InterpConvUpsamplingOp2d()
    g = SkipGenerator(out_sz, n_ch, up_op, id_up_op, in_sz=in_sz, n_features=n_ftrs)

    assert every_conv_has_sn(g)
    assert _g_out_sz_is_ok(g, n_ch, in_sz, out_sz)


@pytest.mark.parametrize("in_sz", [32, 64])
@pytest.mark.parametrize("n_ch", [1, 3])
@pytest.mark.parametrize("n_ftrs", [16, 32])
def test_res_critic(in_sz, n_ch, n_ftrs):
    down_op = ConvHalfDownsamplingOp2d()
    id_down_op = ConcatPoolHalfDownsamplingOp2d()
    crit = res_critic(in_sz, n_ch, down_op, id_down_op, n_features=n_ftrs, flatten_full=False)

    bs = randint(1, 5)
    in_t = torch.rand(bs, n_ch, in_sz, in_sz)
    out = crit(in_t)

    assert every_conv_has_sn(crit)
    assert out.size() == torch.Size([bs, 1])


@pytest.mark.parametrize("in_sz", [32, 64])
@pytest.mark.parametrize("n_ch", [1, 3])
@pytest.mark.parametrize("latent_sz", [50, 100])
@pytest.mark.parametrize("mid_mlp_depth", [0, 2])
def test_img2img_generator(in_sz, n_ch, latent_sz, mid_mlp_depth):
    g = img2img_generator(in_sz, n_ch, latent_sz=latent_sz, mid_mlp_depth=mid_mlp_depth)

    bs = randint(1, 5)
    # Needed because BN expects more than 1 value per channel when training
    if bs == 1: g.eval()
    in_t = torch.rand(bs, n_ch, in_sz, in_sz)
    out = g(in_t)

    assert every_conv_has_sn(g)
    assert out.size() == in_t.size()
