"""
Decorators for wrappers
"""

import warnings


def filter_empty_tensor_warning(func):
    """
    Filter "Empty tensor" PyTorch warnings

    :param func: Function for which we want to suppress warnings
    """

    def wrapper(*args, **kwargs):
        """
        For LGATr, when setting `in_s_channels` or `out_s_channels` to 0 the following
        warning appears because tensors with *no elements* are created
        """
        action = (
            "ignore"
            if not kwargs.get("in_s_channels", 0) or not kwargs.get("out_s_channels", 0)
            else "default"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action,
                "Initializing zero-element tensors is a no-op",
                category=UserWarning,
            )
            return func(*args, **kwargs)

    return wrapper
