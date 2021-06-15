from face2anime.gen_utils import coalesce, is_iterable


def test_is_iterable():
    assert is_iterable([])
    assert is_iterable((0,))
    assert is_iterable({'a': 1})
    assert not is_iterable(0)
    assert not is_iterable(0.)


def test_coalesce():
    assert coalesce() is None
    assert coalesce("a") == "a"
    assert coalesce(1, None) == 1
    assert coalesce(None, 2) == 2
    assert coalesce(None, 1, None, 2) == 1
    assert coalesce(None, None, 2) == 2
