import functools
import logging
import time


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)


def timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        get_logger(func.__module__).info("%s took %.4fs", func.__name__, elapsed)
        return result

    return wrapper
