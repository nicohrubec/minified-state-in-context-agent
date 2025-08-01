import os
import sys  # system module


# This is a class example
class ExampleClass(Base1, Base2):
    """This class demonstrates method signatures and docstrings."""

    class_var = 42

    def __init__(self, name: str, value: int = 10):
        self.name = name
        self.value = value

    def compute(self, factor: float) -> float:
        """Compute a scaled value."""
        result = (self.value + 3.14) * factor
        return result


def utility_function(x, y):
    # Returns the max of two values
    return max(x, y)
