REGISTRY = {}


def register(logger, value):
    def decorator(func):
        REGISTRY[logger, value] = func
        return func

    return decorator


def log_hyperparam(logger, name, value):
    for (logger_cls, value_cls), func in REGISTRY.items():
        if isinstance(logger, logger_cls) and isinstance(value, value_cls):
            func(logger, name, value)
            return
