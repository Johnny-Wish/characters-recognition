import warnings
import functools


def deprecated(fcn):
    """This is a decorator which can be used to mark functions
   as deprecated. It will result in a warning being emitted
   when the function is used."""

    @functools.wraps(fcn)
    def new_fcn(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(fcn.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return fcn(*args, **kwargs)

    return new_fcn


class DeprecatedError(DeprecationWarning):
    pass
