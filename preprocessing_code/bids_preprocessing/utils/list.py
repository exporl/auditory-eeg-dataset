"""List utilities."""


def flatten(lst):
    """Flatten a list.

    Parameters
    ----------
    lst: Union[list, tuple]
        A list to be flattened.

    Returns
    -------
    list
        A flattened list.
    """
    result = []
    if isinstance(lst, (list, tuple)):
        for x in lst:
            result.extend(flatten(x))
    else:
        result.append(lst)
    return result
