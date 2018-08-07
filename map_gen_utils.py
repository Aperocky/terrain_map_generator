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
        img = plt.imshow(grid, interpolation="nearest", cmap=custmap, norm=norm)
        plt.show()

class Rivers:

    def __init__(self, altmap):
        self.altmap = altmap

    def shift_map(self):
        master = self.altmap
        # Create edge slides
        top = np.asarray([master[0]])
        bottom = np.asarray([master[-1]])
        left = np.asarray([master[:, 0]]).T
        right = np.asarray([master[:, -1]]).T
        # Create shift maps
        north = np.concatenate((top, master[:-1, :]), axis=0)
        south = np.concatenate((master[1:, :], bottom), axis=0)
        west = np.concatenate((left, master[:, :-1]), axis=1)
        east = np.concatenate((master[:, 1:], right), axis=1)
        self.north = north
        self.south = south
        self.west = west
        self.east = east

    # Establish flow patterns
    def flow(self):
        # Empty river map for plotting later:
        riverpos = np.zeros(self.altmap.shape)
        self.riverpos = riverpos
        flowmap = np.empty(self.altmap.shape, dtype=tuple)
        for i in range(self.altmap.shape[0]):
            for j in range(self.altmap.shape[1]):
                if self.altmap[i, j] < 0:
                    myres = (0,0)
                    flowmap[i, j] = myres
                    continue
                dn = self.altmap[i, j] - self.north[i, j]
                de = self.altmap[i, j] - self.east[i, j]
                ds = self.altmap[i, j] - self.south[i, j]
                dw = self.altmap[i, j] - self.west[i, j]
                mylist = [dn, de, ds, dw]
                if max(mylist) < 0:
                    myres = (0, 0)
                else:
                    myres = [(-1, 0), (0, 1), (1, 0), (0, -1)][np.argmax(mylist)]
                flowmap[i, j] = myres
        self.flowmap = flowmap
        return flowmap

    def run(self):
        self.shift_map()
        self.flow()
        self.point_flow((50,50))
        self.point_flow((25,25))
        self.point_flow((75,75))

    # River flow from source
    def point_flow(self, source):
        # source: tuple (i, j)
        direction = self.flowmap[source]
        print(source)
        while not direction == (0,0):
            source = tuple(map(sum, zip(source, direction)))
            print(source)
            try:
                me = self.riverpos[source]
            except IndexError:
                return
            direction = self.flowmap[source]
            self.riverpos[source] = 1
        # This appends everything to riverflow

if __name__ == "__main__":
    my_map = Map_generator(100)
    grid = my_map.meshgrid_combine(reps=1000)
    print(grid.shape)
    print(grid.max())
    print(grid.min())
    # print(grid)
    my_map.colormap(grid)

    river = Rivers(grid)
    river.run()
    print(river.flowmap)
    print(river.riverpos.sum())
    plt.imshow(river.riverpos)
    plt.show()
    