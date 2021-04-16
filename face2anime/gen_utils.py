__all__ = ['is_iterable']


def is_iterable(x):
    try:
        iterator = iter(x)
    except TypeError:
        return False
    else:
        return True
