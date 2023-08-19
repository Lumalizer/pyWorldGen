import numpy as np
from PIL import Image
from dataclasses import dataclass


@dataclass
class NoiseTexture:
    noise: np.ndarray
    scale: int
    distribution: np.ndarray = None

    def __post_init__(self):
        x = np.random.choice(self.noise.shape[0], self.scale, replace=False)
        y = np.random.choice(self.noise.shape[1], self.scale, replace=False)
        self.distribution = np.sort(self.noise[x, y])

    def to_image(self) -> Image.Image:
        return Image.fromarray((self.noise * 255).astype(np.uint8))

    def lowerthan(self, num: int):
        assert 0 <= num <= 100
        index = int(num*(len(self.distribution) / 100))
        return np.nonzero(self.noise <= self.distribution[index])