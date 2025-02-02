from abstract_box import AbstractLandBox


class VegetationBox(AbstractLandBox):
    """Vegetation box class."""

    cveg0 = 100

    def __init__(self, **kwargs):
        super().__init__(kwargs.get("carbon", self.cveg0))
