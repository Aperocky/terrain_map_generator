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
            sigma_size_top, sigma_size_bot = np.random.uniform(0.2, 1.9, size=2)*sigma_size_pre
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
        self.grid = master
        return master

    def sources(self):
        # Todo, get source of points
        poss = list(zip(*np.where(self.grid > 5)))
        print(len(poss))
        choicenum = 30
        if len(poss) < 300:
            choicenum = int(len(poss)/10) + 1
        if len(poss) > 0:
            idxes = np.random.choice(len(poss), choicenum)
            sources = [poss[idx] for idx in idxes]
        return sources

    def get_rivers(self):
        self.river = Rivers(self.grid)
        # sources = self.sources()
        self.riverpos = self.river.run()
        

    # Generate a colormap to display mesh result.
    def colormap(self):

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

        rdict = {
            "red": (
                (0,0,0),
                (1,0,0)
            ),
            "green": (
                (0,0,0),
                (0.1,0.1,0),
                (1,0,0)
            ),
            "blue": (
                (0,0,0),
                (0.1,0,0.6),
                (0.5,0.8,0.8),
                (1,1,1)
            )
        }

        rivercolors = colors.LinearSegmentedColormap("mapcol", rdict)
        custmap = colors.LinearSegmentedColormap("mapcol", cdict)
        cmap = colors.ListedColormap(['navy', 'blue', 'green', 'grey', 'brown'])
        # bounds = [-10, -2, 0, 2.5, 4.5, 10]
        norm = colors.Normalize(-10, 10, custmap.N)
        rivernorm = colors.Normalize(0,200, rivercolors)
        # img = plt.imshow(grid, cmap=cmap, norm=norm)
        plt.figure(figsize=(8,8))
        plt.box(on=None)
        plt.axis('off')
        plt.imshow(self.riverpos, alpha=1.0, interpolation="nearest", cmap=rivercolors, norm=rivernorm)
        plt.imshow(self.grid, interpolation="hamming", cmap=custmap, norm=norm, alpha=0.6)
        plt.show()

class Rivers:

    def __init__(self, altmap):
        self.altmap = altmap
        self.shape = self.altmap.shape

    def shift_map(self):
        master = self.altmap
        # Create edge slides
        top = np.asarray([master[0]])
        bottom = np.asarray([master[-1]])
        left = np.asarray([master[:, 0]]).T
        right = np.asarray([master[:, -1]]).T
        # Create shift maps
        # north = np.concatenate((top, master[:-1, :]), axis=0)
        # south = np.concatenate((master[1:, :], bottom), axis=0)
        # west = np.concatenate((left, master[:, :-1]), axis=1)
        # east = np.concatenate((master[:, 1:], right), axis=1)
        n = self.altmap[:-2, 1:-1]
        s = self.altmap[2:, 1:-1]
        e = self.altmap[1:-1, 2:]
        w = self.altmap[1:-1, :-2]
        ne = self.altmap[:-2, 2:]
        se = self.altmap[2:, 2:]
        sw = self.altmap[2:, :-2]
        nw = self.altmap[:-2, :-2]
        n = np.pad(n, 1, "constant")
        s = np.pad(s, 1, "constant")
        e = np.pad(e, 1, "constant")
        w = np.pad(w, 1, "constant")
        ne = np.pad(ne, 1, "constant")
        se = np.pad(se, 1, "constant")
        sw = np.pad(sw, 1, "constant")
        nw = np.pad(nw, 1, "constant")
        self.n = n
        self.s = s
        self.e = e
        self.w = w
        self.ne = ne
        self.se = se
        self.sw = sw
        self.nw = nw
        # self.north = north
        # self.south = south
        # self.west = west
        # self.east = east

    # Establish flow patterns
    def flow(self):
        # Empty river map for plotting later:
        riverpos = np.zeros(self.altmap.shape)
        self.altmapa = np.pad(self.altmap[1:-1, 1:-1], 1, "constant")
        self.riverpos = riverpos
        flowmap = np.empty(self.altmap.shape, dtype=tuple)
        for i in range(self.altmap.shape[0]):
            for j in range(self.altmap.shape[1]):
                if self.altmap[i, j] < 0:
                    myres = (0,0)
                    flowmap[i, j] = myres
                    continue
                dn = self.altmapa[i, j] - self.n[i, j]
                de = self.altmapa[i, j] - self.e[i, j]
                ds = self.altmapa[i, j] - self.s[i, j]
                dw = self.altmapa[i, j] - self.w[i, j]
                dne = self.altmapa[i, j] - self.ne[i, j]
                dse = self.altmapa[i, j] - self.se[i, j]
                dsw = self.altmapa[i, j] - self.sw[i, j]
                dnw = self.altmapa[i, j] - self.nw[i, j]
                mylist = [dn, de, ds, dw, dne, dse, dsw, dnw]
                if max(mylist) <= 0:
                    myres = (0, 0)
                else:
                    myres = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)][np.argmax(mylist)]
                flowmap[i, j] = myres
        self.flowmap = flowmap
        return flowmap
    
    def in_range(self, position):
        for i in range(len(position)):
            if position[i] < 0 or position[i] >= self.shape[i]:
                return False
        return True

    # Use flowmap recursively to find flow in each place in the array. 
    def volume_map(self):
        volume = np.zeros(self.altmap.shape, dtype=float)
        self.volume = volume
        flowmap_to = np.empty(self.altmap.shape, dtype=tuple)
        flowmap_from = np.empty(self.altmap.shape, dtype=object)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                flowmap_to[i, j] = tuple(map(sum, zip(self.flowmap[i, j], (i,j))))
                flowmap_from[i, j] = list()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                flowmap_from[flowmap_to[i, j]].append((i,j))
        self.flowmap_from = flowmap_from

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.recursive_flow((i,j))
        # print(self.volume)
                
    # Achieve recursive flow for each loc
    def recursive_flow(self, position):
        if not self.volume[position] == 0:
            # print("already taken %s", str(position))
            return self.volume[position]
        if len(self.flowmap_from[position]) == 0:
            # print("ridge %s", str(position))
            self.volume[position] = 1
            return 1
        # print("summation %s", str(position))
        sum_flow = 1
        for each in self.flowmap_from[position]:
            if each == position:
                continue
            sum_flow += self.recursive_flow(each)
        self.volume[position] = min(sum_flow, 200)
        return sum_flow

    def run(self, sources = None):
        self.shift_map()
        self.flow()
        if not sources is None:
            self.flows(sources)
            return self.riverpos
        self.volume_map()
        return self.volume

    # River flow from source
    def point_flow(self, source):
        # source: tuple (i, j)
        direction = self.flowmap[source]
        while not direction == (0,0):
            # print(source, direction)
            # print(self.altmap[source])
            source = tuple(map(sum, zip(source, direction)))
            if not self.in_range(source):
                return
            # try:
            #     me = self.riverpos[source]
            # except IndexError:
            #     return
            direction = self.flowmap[source]
            self.riverpos[source] = 1
        # This appends everything to riverflow

    def flows(self, sources):
        for source in sources:
            self.point_flow(source)
        return self.riverpos

if __name__ == "__main__":
    my_map = Map_generator(120)
    grid = my_map.meshgrid_combine(reps=1200)
    print(grid.shape)
    print(grid.max())
    print(grid.min())
    # print(grid)
    my_map.get_rivers()
    # my_map.river.volume_map()
    my_map.colormap()
    # plt.imshow(my_map.river.volume)
    # plt.show()

    # river = Rivers(grid)
    # river.run()
    # print(river.flowmap)
    # print(river.riverpos.sum())
    # plt.imshow(river.riverpos)
    # plt.show()
      