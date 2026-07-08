def build_generator(*args, **kwargs):
    from .factory import build_generator as _build_generator

    return _build_generator(*args, **kwargs)


__all__ = ["build_generator"]
