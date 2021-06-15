__all__ = ['is_iterable', 'coalesce']


def is_iterable(x):
    try:
        iterator = iter(x)
    except TypeError:
        return False
    else:
        return True


def coalesce(*args):
    "Returns the first not None value."
    return next((a for a in args if a is not None), None)
