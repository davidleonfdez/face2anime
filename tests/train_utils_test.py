from face2anime.train_utils import *
from tests.helpers.testing_fakes import DummyObject


def test_epoch_filter_all():
    ep_filter = EpochFilterAll()

    assert ep_filter.contains(1)
    assert ep_filter.contains(2)
    assert ep_filter.contains(3)
    assert ep_filter.contains(10)
    assert ep_filter.contains(999)


def test_epoch_filter_multiple_of_n():
    ep_filter_3 = EpochFilterMultipleOfN(3)
    ep_filter_5 = EpochFilterMultipleOfN(5)

    assert all(ep_filter_3.contains(i) for i in [3, 6, 9, 12, 99, 300])
    assert all(ep_filter_5.contains(i) for i in [5, 10, 15, 20, 25, 55, 100])
    assert not any(ep_filter_3.contains(i) for i in [1, 2, 4, 5, 7, 8, 10, 11, 100])
    assert not any(ep_filter_5.contains(i) for i in [1, 2, 3, 4, 6, 7, 8, 9, 11, 99, 101])


def test_epoch_filter_after_n():
    ep_filter = EpochFilterAfterN(6)

    assert not any(ep_filter.contains(i) for i in range(7))
    assert ep_filter.contains(7)
    assert ep_filter.contains(8)
    assert ep_filter.contains(9)
    assert ep_filter.contains(99)


def test_composed_epoch_filter():
    ep_filter = ComposedEpochFilter([
        EpochFilterAfterN(6),
        EpochFilterMultipleOfN(3)
    ])

    assert all(ep_filter.contains(i) for i in [9, 12, 15, 18, 99, 102])
    assert not any(ep_filter.contains(i) for i in [1, 3, 4, 5, 6, 7, 8, 10, 11, 100])


class FakeEMACallback(EMACallback):
    def __init__(self): 
        self.n_update_bn_calls = 0
        
    def update_bn(self):
        self.n_update_bn_calls += 1


def test_update_ema_pre_save_action():
    fake_learn = DummyObject()
    fake_ema_cb = FakeEMACallback()
    fake_learn.cbs = [fake_ema_cb]
    action = UpdateEMAPreSaveAction(fake_learn, EpochFilterAfterN(3))
    update_bn_calls_progression = []

    for i in range(6):
        action.before_save(i+1)
        update_bn_calls_progression.append(fake_ema_cb.n_update_bn_calls)

    assert update_bn_calls_progression == [0, 0, 0, 1, 2, 3]
