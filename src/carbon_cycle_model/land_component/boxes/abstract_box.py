"""
Abstract class implementing the general box template all other land boxes should follow.
"""

from abc import ABC


class AbstractLandBox(ABC):
    """Abstract Creator class to define a land carbon box."""

    def get_carbon(self):
        """Return this box's carbon stock."""
        return self.stock

    def modify_stock(self, modification):
        """Modify the box's carbon stock by the supplied quantity."""
        self.stock += modification

    def __init__(self, carbon0=0):
        self.stock0 = carbon0
        self.stock = carbon0
