from face2anime.layers import MiniBatchStdDev
import torch


def test_minibatch_std():
    l = MiniBatchStdDev(group_sz=2, unbiased_std=False)
    t = torch.Tensor([   
        # Group 1
        [[[0, 1], [1, 2]], [[3, 0], [3, 2]], [[0, 1], [2, -17]]],
        [[[1, 1], [3, 4]], [[3, 3], [3, 8]], [[0, 1], [2, 17]]],
        # Group 2
        [[[0, 1], [2, 3]], [[0, 1], [2, 3]], [[0, 1], [2, 3]]],
        [[[2, 3], [4, 5]], [[2, 3], [4, 5]], [[2, 3], [4, 5]]],
    ])
    
    #expected_std = torch.Tensor([
    #    [ [[0.5, 0], [1, 1]], [[0, 1.5], [0, 3]], [[0, 0], [0, 17]] ],
    #    [[[1]*2]*2]*3
    #])
    #expected_mean_std = torch.Tensor([2, 1]) unsqueezed 3 times
    expected = torch.Tensor([   
        # Group 1
        [[[0, 1], [1, 2]], [[3, 0], [3, 2]], [[0, 1], [2, -17]], [[2, 2], [2, 2]]],
        [[[1, 1], [3, 4]], [[3, 3], [3, 8]], [[0, 1], [2, 17]], [[2, 2], [2, 2]]],
        # Group 2
        [[[0, 1], [2, 3]], [[0, 1], [2, 3]], [[0, 1], [2, 3]], [[1, 1], [1, 1]]],
        [[[2, 3], [4, 5]], [[2, 3], [4, 5]], [[2, 3], [4, 5]], [[1, 1], [1, 1]]],
    ])
    
    actual = l(t)
    assert torch.allclose(actual, expected)
