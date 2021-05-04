from face2anime.gen_utils import is_iterable


def test_is_iterable():
    assert is_iterable([])
    assert is_iterable((0,))
    assert is_iterable({'a': 1})
    assert not is_iterable(0)
    assert not is_iterable(0.)
