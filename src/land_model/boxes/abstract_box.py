from abc import ABC, abstractmethod


class AbstractLandBox(ABC):
    """Abstract Creator class to define a land carbon box."""

    def get_carbon(self):
        return self.pars["carbon"]

    def __init__(self, carbon0: "int" = 0):
        self.pars = {"carbon0": carbon0, "carbon": carbon0}
