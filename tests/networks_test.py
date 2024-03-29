from fastai.layers import ResBlock
from face2anime.layers import (ConcatPoolHalfDownsamplingOp2d, CondConvX2UpsamplingOp2d, 
                               CondInterpConvUpsamplingOp2d, ConvHalfDownsamplingOp2d, 
                               ConvX2UpsamplingOp2d, FeatureStatType, InterpConvUpsamplingOp2d, 
                               ParentNetSource, ResBlockDown)
from face2anime.networks import (basic_encoder, CondResGenerator, custom_generator, 
                                 Img2ImgGenerator, NoiseSplitDontSplitStrategy, 
                                 NoiseSplitEqualLeave1stOutStrategy, patch_res_critic,
                                 PatchResCritic, res_critic, res_generator, SkipGenerator)
from face2anime.torch_utils import every_conv_has_sn
from fastai.vision.all import NormType
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
                                       CondConvX2UpsamplingOp2d(cond_sz, 
                                                                norm_type=NormType.Instance),
                                       CondInterpConvUpsamplingOp2d(cond_sz,
                                                                    norm_type=NormType.Instance),
                                       noise_split_strat,
                                       norm_type=NormType.Instance,
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
@pytest.mark.parametrize("n_ftrs", [16, 32])
@pytest.mark.parametrize("out_sz", [4, 8])
def test_old_patch_res_critic(in_sz, n_ch, n_ftrs, out_sz):
    down_op = ConvHalfDownsamplingOp2d()
    id_down_op = ConcatPoolHalfDownsamplingOp2d()
    crit_args = [in_sz, n_ch, out_sz, down_op, id_down_op]
    crit = patch_res_critic(*crit_args, n_features=n_ftrs, flatten_full=False)
    crit_flat_out = patch_res_critic(*crit_args, n_features=n_ftrs, flatten_full=True)

    bs = randint(1, 5)
    in_t = torch.rand(bs, n_ch, in_sz, in_sz)
    out = crit(in_t)
    flat_out = crit_flat_out(in_t)

    assert every_conv_has_sn(crit)
    assert out.size() == torch.Size([bs, out_sz**2])
    assert flat_out.size() == torch.Size([bs * out_sz**2])


@pytest.mark.parametrize("in_sz", [32, 64])
@pytest.mark.parametrize("n_ftrs", [16, 32])
@pytest.mark.parametrize("out_sz", [4, 8])
@pytest.mark.parametrize("ftrs_stats", 
                         [FeatureStatType.NONE, 
                          FeatureStatType.MEAN, 
                          FeatureStatType.STD | FeatureStatType.CORRELATIONS])
def test_patch_res_critic(in_sz, n_ftrs, out_sz, ftrs_stats):
    n_ch = 3
    down_op = ConvHalfDownsamplingOp2d()
    id_down_op = ConcatPoolHalfDownsamplingOp2d()
    crit_args = [in_sz, n_ch, out_sz, down_op, id_down_op]
    ftrs_stats_source = ParentNetSource(layer_types=(ResBlock, ResBlockDown))
    crit = PatchResCritic(*crit_args, n_features=n_ftrs, flatten_full=False,
                          ftrs_stats=ftrs_stats, ftrs_stats_source=ftrs_stats_source)
    crit_flat_out = PatchResCritic(*crit_args, n_features=n_ftrs, flatten_full=True,
                                   ftrs_stats=ftrs_stats, ftrs_stats_source=ftrs_stats_source)

    bs = randint(1, 5)
    in_t = torch.rand(bs, n_ch, in_sz, in_sz)
    out = crit(in_t)
    flat_out = crit_flat_out(in_t)
    stats_mult = 2 if ftrs_stats != FeatureStatType.NONE else 1

    assert every_conv_has_sn(crit)
    assert out.size() == torch.Size([bs, out_sz**2 * stats_mult])
    assert flat_out.size() == torch.Size([bs * out_sz**2 * stats_mult])


@pytest.mark.parametrize("in_sz", [32, 64])
@pytest.mark.parametrize("n_ch", [1, 3])
@pytest.mark.parametrize("out_sz", [50, 100])
def test_basic_encoder(in_sz, n_ch, out_sz):
    enc = basic_encoder(in_sz, n_ch, out_sz)

    bs = randint(1, 5)
    out = enc(torch.rand(bs, n_ch, in_sz, in_sz))

    assert out.size() == torch.Size([bs, out_sz])


@pytest.mark.parametrize("in_sz", [32, 64])
@pytest.mark.parametrize("n_ch", [1, 3])
@pytest.mark.parametrize("latent_sz", [50, 100])
@pytest.mark.parametrize("mid_mlp_depth", [0, 2])
@pytest.mark.parametrize("skip_connect", [False, True])
def test_img2img_generator(in_sz, n_ch, latent_sz, mid_mlp_depth, skip_connect):
    g = Img2ImgGenerator(in_sz, n_ch, latent_sz=latent_sz, mid_mlp_depth=mid_mlp_depth,
                         skip_connect=skip_connect)

    bs = randint(1, 5)
    # Needed because BN expects more than 1 value per channel when training
    if bs == 1: g.eval()
    in_t = torch.rand(bs, n_ch, in_sz, in_sz)
    out = g(in_t)

    assert every_conv_has_sn(g)
    assert out.size() == in_t.size()
