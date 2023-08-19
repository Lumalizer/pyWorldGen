import numpy as np
import pyfastnoisesimd


def generate_noise(scale=4800, zoom=0.042, octaves=3, lacunarity=2.0, power=1) -> np.ndarray:
    simplex = pyfastnoisesimd.Noise(numWorkers=8)
    simplex.fractal.octaves = octaves
    simplex.fractal.lacunarity = lacunarity
    simplex.axesScales = (zoom, zoom, zoom)

    simplex_grid = simplex.genAsGrid((scale, scale))
    grid = simplex_grid + abs(np.min(simplex_grid))
    grid = grid / np.max(grid)  # rebalanced to a range between 0 and 1
    return np.power(grid, power)