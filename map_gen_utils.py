import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, sys

class Map_generator:

    def __init__(self, size, name = "Default"):
        self.size = size
        self.name = name
        if os.path.exists("maps/%s" % self.name):
            # Check number if exist
            for i in range(1000):
                if not os.path.exists("maps/%s%s" % (self.name, str(i).zfill(3))):
                    os.mkdir("maps/%s%s" % (self.name, str(i).zfill(3)))
                    directory = "maps/%s%s" % (self.name, str(i).zfill(3))
                    break
        else:
            os.mkdir("maps/%s" % self.name)
            directory = "maps/%s" % self.name
        self.path = directory

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
            sigma_size_pre = np.random.uniform(self.size*0.3, self.size)
            sigma_size_top, sigma_size_bot = np.random.uniform(0.2, 1.9, size=2)*sigma_size_pre
            cov_max = np.sqrt(sigma_size_top * sigma_size_bot) * 0.8
            sigma_cov = np.random.uniform(-cov_max, cov_max)
            sigma = np.array([[sigma_size_top, sigma_cov],[sigma_cov, sigma_size_bot]])
            dist = multivariate_normal(mu, sigma)

            # Map the multivariate distribution to 2d array
            rel_heights = dist.pdf(pos)
            rel_heights /= rel_heights.max()
            return rel_heights

    def meshgrid_combine(self, reps = 100, adjust = -1):
        master = np.zeros((self.size, self.size))
        for _ in range(reps):
            grid = self.meshgrid_normal()
            grid *= np.random.uniform(-0.8,1)
            master += grid
        master += adjust
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

    def get_rivers(self, dryness= 1):
        self.river = Hydrology(self.grid)
        # sources = self.sources()
        self.volume = self.river.run(dryness=dryness)
        

    # Generate a colormap to display mesh result.
    def colormap(self, save = False):

        # Color Dictionary
        cdict= {
            "red": (
                (0, 0, 0),
                (0.5, 0, 0.6),
                (0.6, 0.6, 0.6),
                (0.75, 0.4, 0.7),
                (1, 0.2, 0.2)
            ),
            "green": (
                (0, 0, 0),
                (0.5, 0, 0.6),
                (0.6, 0.5, 0.5),
                (0.75, 0.3, 0.7),
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
                # (0.0001, 0, 0.8),
                (0.001, 0, 0.4),
                (0.015, 0.4, 0.2),
                (0.05, 0.2, 0),
                (1,0,0)
            ),
            "green": (
                (0,0,0),
                # (0.0001, 0, 0.8),
                (0.001, 0, 1),
                (0.015, 1, 0.7),
                (0.05, 0.7, 0.8),
                (0.10, 0.8, 0),
                (0.11,0.0,0),
                (1,0,0)
            ),
            "blue": (
                (0,0,0),
                (0.05, 0, 0.8),
                (0.10, 0.8, 0),
                (0.1,0,0.9),
                (0.5,0.9,0.7),
                (1,0.7,1)
            ),
            "alpha": (
                (0,0,0),
                (0.001, 0, 0.5),
                (0.015, 0.5, 0.8),
                (0.1, 0.8, 0.9),
                (0.5,0.9,1),
                (1,1,1)
            )
        }

        rivercolors = colors.LinearSegmentedColormap("mapcol", rdict)
        custmap = colors.LinearSegmentedColormap("mapcol", cdict)
        cmap = colors.ListedColormap(['navy', 'blue', 'green', 'grey', 'brown'])
        # bounds = [-10, -2, 0, 2.5, 4.5, 10]
        norm = colors.Normalize(-10, 10, custmap.N)
        rivernorm = colors.Normalize(0, 200, rivercolors)
        # img = plt.imshow(grid, cmap=cmap, norm=norm)
        fig, ax = plt.subplots()
        # plt.box(on=None)
        # plt.axis('off')
        ax.imshow(self.grid, interpolation="hamming", cmap=custmap, norm=norm)
        ax.imshow(self.volume, interpolation="nearest", cmap=rivercolors, norm=rivernorm)
        ax.grid(True)
        plt.show()
        if save:
            fig.savefig("%s/map.png" % self.path, dpi=120, frameon=False, bbox_inches='tight', pad_inches=0)

    # Save these maps!
    def export(self):
        # List export values
        # ALTITUDE MAP. self.grid
        np.save("%s/altitude" % self.path, self.grid)
        # VOLUME MAP. self.volume
        np.save("%s/volume" % self.path, self.volume)
        # PLT map
        self.colormap(save=True)
        #
        # TERRAIN MAP (coming)

        


class Hydrology:

    def __init__(self, altmap):
        self.altmap = altmap
        self.shape = self.altmap.shape

    # ---------------- GENERATE FLOWMAP ------------------- #

    def shift_map(self):
        master = self.altmap
        # Create edge slides
        top = np.asarray([master[0]])
        bottom = np.asarray([master[-1]])
        left = np.asarray([master[:, 0]]).T
        right = np.asarray([master[:, -1]]).T
        # Create shift maps
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

    # ------------------- IN RANGE UTIL FUNCTION ----------- #
    
    def in_range(self, position):
        for i in range(len(position)):
            if position[i] < 0 or position[i] >= self.shape[i]:
                return False
        return True

    # ----------------- FLOW VOLUME (RIVER) GENERATION -------- #

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
        print(np.amax(self.volume))
                
    # Achieve recursive flow for each loc
    def recursive_flow(self, position, cutoff=6):
        if not self.volume[position] == 0:
            # print("already taken %s", str(position))
            return self.volume[position]
        if len(self.flowmap_from[position]) == 0:
            # print("ridge %s", str(position))
            if self.altmap[position] < 0:
                self.volume[position] = 0
                return 0
            self.volume[position] = self.precipitate_map[position]
            return self.precipitate_map[position]
        # print("summation %s", str(position))
        # Use precipitation data
        if self.altmap[position] < 0 or self.altmap[position] > cutoff:
            self.volume[position] = 0
            return 0
        sum_flow = self.precipitate_map[position]
        for each in self.flowmap_from[position]:
            if each == position:
                continue
            sum_flow += self.recursive_flow(each)
        sum_flow -= 0.2
        if sum_flow < 0:
            sum_flow = 0
        self.volume[position] = sum_flow
        return sum_flow

    # ---------------- RUN ALL -----------------------------

    def run(self, sources=None, dryness=1.0):
        self.shift_map()
        self.flow()
        self.precipitate(reps = dryness, factor = dryness)

        # sources is deprecated
        if not sources is None:
            self.flows(sources)
            return self.riverpos

        self.volume_map()
        return self.volume

    # ----------------- PRECIPITATION ----------------------

    def meshgrid_normal(self, mult_factor):
        # Create a meshgrid of size
        x = np.linspace(0, self.shape[0]-1, self.shape[0])
        y = x.copy()
        xx, yy = np.meshgrid(x, y)
        pos = np.empty(xx.shape + (2,))
        pos[:, :, 0] = xx
        pos[:, :, 1] = yy

        # Create the multi_variate distribution randomly
        mu = np.random.randint(-self.shape[0]*0.3, self.shape[0]*1.3, size=2)
        sigma_size_pre = np.random.uniform(self.shape[0]*0.8, self.shape[0])
        sigma_size_top, sigma_size_bot = np.random.uniform(0.2, 2, size=2)*sigma_size_pre
        cov_max = np.sqrt(sigma_size_top * sigma_size_bot) * 0.8
        sigma_cov = np.random.uniform(-cov_max, cov_max)
        sigma = np.array([[sigma_size_top, sigma_cov],[sigma_cov, sigma_size_bot]])
        dist = multivariate_normal(mu, sigma)

        # Map the multivariate distribution to 2d array
        rel_heights = dist.pdf(pos)
        rel_heights *= mult_factor
        return rel_heights

    def precipitate(self, reps = 1, factor = 1):
        reps = int(self.shape[0] * reps)
        factor = int(self.shape[0] * factor)
        precipitate_map = np.zeros(self.shape)
        for i in range(reps):
            grid = self.meshgrid_normal(factor)
            precipitate_map += grid
        self.precipitate_map = precipitate_map
        # print(precipitate_map)
        # print(np.max(precipitate_map))
        # print(np.sum(precipitate_map))
        plt.imshow(precipitate_map)
        plt.show()

    # ----------------- DEPRECATED FUNCTIONS --------------

    # River flow from source (deprecated)
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

    # (deprecated)
    def flows(self, sources):
        for source in sources:
            self.point_flow(source)
        return self.riverpos

if __name__ == "__main__":
    my_name = "default"
    size = 150
    dry = 1
    if len(sys.argv) > 1:
        my_name = sys.argv[1]
    if len(sys.argv) > 2:
        size = int(sys.argv[2])
    if len(sys.argv) > 3:
        dry = float(sys.argv[3])
    my_map = Map_generator(size, name=my_name)
    grid = my_map.meshgrid_combine(reps=1000, adjust=0)
    print(grid.shape)
    print(grid.max())
    print(grid.min())
    # print(grid)
    my_map.get_rivers(dryness = dry)
    my_map.export()
    # my_map.colormap()
    # plt.imshow(my_map.river.volume)
    # plt.show()

    # river = Rivers(grid)
    # river.run()
    # print(river.flowmap)
    # print(river.riverpos.sum())
    # plt.imshow(river.riverpos)
    # plt.show()
      