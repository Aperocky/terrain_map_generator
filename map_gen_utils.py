import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class Map_generator:

    def __init__(self, size):
        self.size = size

    def meshgrid_normal(self):
            # Create a meshgrid of size
            x = np.linspace(0, self.size-1, self.size)
            y = x.copy()
            xx, yy = np.meshgrid(x, y)
            pos = np.empty(xx.shape + (2,))
            pos[:, :, 0] = xx
            pos[:, :, 1] = yy

            # Create the multi_variate distribution randomly
            mu = np.random.randint(self.size, size=2)
            sigma_size_pre = np.random.uniform(self.size*0.2, self.size*0.8)
            sigma_size_top, sigma_size_bot = np.random.uniform(0.1, 1.9, size=2)*sigma_size_pre
            cov_max = np.sqrt(sigma_size_top * sigma_size_bot) * 0.8
            sigma_cov = np.random.uniform(-cov_max, cov_max)
            sigma = np.array([[sigma_size_top, sigma_cov],[sigma_cov, sigma_size_bot]])
            dist = multivariate_normal(mu, sigma)

            # Map the multivariate distribution to 2d array
            rel_heights = dist.pdf(pos)
            rel_heights /= rel_heights.max()
            return rel_heights

    def meshgrid_combine(self, reps = 100):
        master = np.zeros((self.size, self.size))
        for _ in range(reps):
            grid = self.meshgrid_normal()
            grid *= np.random.uniform(-0.8,1)
            master += grid
        master -= 2
        return master

    # Generate a colormap to display mesh result.
    def colormap(self, grid):

        # Color Dictionary
        cdict= {
            "red": (
                (0, 0, 0),
                (0.6, 0, 0.2),
                (0.75, 0.5, 0.7),
                (1, 0.2, 0.2)
            ),
            "green": (
                (0, 0, 0),
                (0.5, 0, 0.3),
                (0.75, 0.5, 0.7),
                (1.0, 0.2, 0.2)
            ),
            "blue": (
                (0, 0, 0.2),
                (0.5, 0.6, 0),
                (0.75, 0, 0.7),
                (1, 0.7, 0.7)
            )
        }
        custmap = colors.LinearSegmentedColormap("mapcol", cdict)
        cmap = colors.ListedColormap(['navy', 'blue', 'green', 'grey', 'brown'])
        # bounds = [-10, -2, 0, 2.5, 4.5, 10]
        norm = colors.Normalize(-10, 10, custmap.N)
        # img = plt.imshow(grid, cmap=cmap, norm=norm)
        img = plt.imshow(grid, interpolation="hamming", cmap=custmap, norm=norm)
        plt.show()

if __name__ == "__main__":
    my_map = Map_generator(100)
    grid = my_map.meshgrid_combine(reps=1000)
    print(grid.shape)
    print(grid.max())
    print(grid.min())
    # print(grid)
    my_map.colormap(grid)
    