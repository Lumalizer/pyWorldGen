import numpy as np
from PIL import Image
from dataclasses import dataclass
from noises.generate_noise import generate_noise
from noises.NoiseTexture import NoiseTexture
from world.biomes import *


@dataclass
class WorldGenerator:
    scale: int = 4800
    zoomlevel: float = 0.052
    octaves: int = 8
    lacunarity: float = 2.0

    sea_level: int = 12 # percentage of lowest altitude pixels

    altitude: NoiseTexture = None
    moisture: NoiseTexture = None
    texture: Image.Image = None

    def generate_moisture(self, altitude: NoiseTexture):
        noise = generate_noise(self.scale, self.zoomlevel, self.octaves, self.lacunarity, power=2)
        # reduce moisture for higher altitudes
        factor = np.power(1.0 - altitude.noise, 12)
        factor = factor / np.max(factor)
        moisture = noise + 0.5 * factor
        moisture = moisture / np.max(moisture)
        moisture = NoiseTexture(moisture, self.scale)
        moisture.noise[altitude.lowerthan(self.sea_level)] = 1
        return moisture

    @staticmethod
    def select_areas_conditional(maps: list[NoiseTexture], values: list, modes: list):
        # selects areas of the world based on the conditions given
        # [altitude, moisture], [30, 83], ['h', 'h'] means higher altitude than 30%, higher moisture than 83%
        conditions = []
        for map, value, mode in zip(maps, values, modes):
            assert 0 <= value <= 100
            target = map.distribution[int(value*(len(map.distribution) / 100))]
            condition = map.noise >= target if mode == 'h' else map.noise <= target
            conditions.append(condition)

        index = np.stack(conditions)
        while len(index.shape) > 2:
            index = np.bitwise_and.reduce(index)
        return np.nonzero(index)

    def texture_from_height_moisture(self, altitude: NoiseTexture, moisture: NoiseTexture) -> Image.Image:
        select = self.select_areas_conditional

        pixels = np.full((*altitude.noise.shape, 3), OCEAN)
        # pixels[select([altitude], [self.sea_level], ['l'])] = OCEAN
        pixels[select([altitude, altitude], [self.sea_level, 14], ['h', 'l'])] = BEACH

        pixels[select([altitude, altitude, moisture], [14, 30, 66], ['h', 'l', 'h'])] = TROPICAL_RAIN_FOREST
        pixels[select([altitude, altitude, moisture], [14, 30, 66], ['h', 'l', 'l'])] = TROPICAL_SEASONAL_FOREST
        pixels[select([altitude, altitude, moisture], [14, 30, 33], ['h', 'l', 'l'])] = GRASSLAND
        pixels[select([altitude, altitude, moisture], [14, 30, 16], ['h', 'l', 'l'])] = SUBTROPICAL_DESERT

        pixels[select([altitude, moisture], [30, 83], ['h', 'h'])] = TEMPERATE_RAIN_FOREST
        pixels[select([altitude, moisture], [30, 83], ['h', 'l'])] = TEMPERATE_DECIDUOS_FOREST
        pixels[select([altitude, moisture], [30, 50], ['h', 'l'])] = GRASSLAND
        pixels[select([altitude, moisture], [30, 16], ['h', 'l'])] = TEMPERATE_DESERT

        pixels[select([altitude, moisture], [60, 66], ['h', 'h'])] = TAIGA
        pixels[select([altitude, moisture], [60, 66], ['h', 'l'])] = SHRUBLAND
        pixels[select([altitude, moisture], [60, 33], ['h', 'l'])] = TEMPERATE_DESERT

        pixels[select([altitude, moisture], [80, 50], ['h', 'h'])] = SNOW
        pixels[select([altitude, moisture], [80, 50], ['h', 'l'])] = TUNDRA
        pixels[select([altitude, moisture], [80, 20], ['h', 'l'])] = BARE
        pixels[select([altitude, moisture], [80, 10], ['h', 'l'])] = SCORCHED

        r = generate_noise(self.scale, self.zoomlevel, self.octaves, self.lacunarity).reshape((self.scale, self.scale, 1))
        r = r/4 + 0.8
        return Image.fromarray((pixels*r).astype(np.uint8))

    def generate_procedural_world(self):
        self.altitude = NoiseTexture(generate_noise(self.scale, self.zoomlevel, self.octaves, self.lacunarity, power=4), self.scale)
        self.moisture = self.generate_moisture(self.altitude)
        self.texture = self.texture_from_height_moisture(self.altitude, self.moisture)
        return self.texture, self.altitude.to_image(), self.moisture.to_image()