"""Operations that can be applied when splitting the data."""
import abc


class SplitterOperation(abc.ABC):
    """Base class for operations that can be applied when splitting data."""

    @abc.abstractmethod
    def __call__(self, data):
        """Apply the operation to the data.

        Parameters
        ----------
        data: Any

        Returns
        -------
        Any
            Data after the operation has been applied.
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """Reset the operation to its initial state."""
        pass
