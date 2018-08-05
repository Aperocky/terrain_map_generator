import numpy as np
import os, sys
sys.path.append(os.getcwd())
from map_gen_utils import Map_generator

class Location:

    # Store function in dictionary
    def add_size(self, num):
        self.size += num

    def change_side(self, side):
        self.side = side

    action_dict = {
        "add" : add_size,
        "change" : change_side
    }

    def __init__(self):
        self.side = "neutral"
        self.size = 0

    def action(self, action):
        if not action[0] in self.action_dict:
            return
        self.action_dict[action[0]](action[1])
        
    def assign_terrain(self, terrain):
        self.terrain = terrain

class Map:

    """
    Terrain Map:
    "DW" (deep water), "SW" (shallow water)
    "P" (plain), "F" (forest), "M" (mountain), "D" (desert)
    """

    def __init__(self, size):
        self.size = size
        self.genmap()

    def action(self, loc, action):
        self.map[loc].action(action)

    def genmap(self):
        my_map = list()
        for i in range(self.size):
            currlist = []
            for j in range(self.size):
                currlist.append(Location())
            my_map.append(currlist)
        self.map = np.asarray(my_map)

    # 2D random bessel function generating maps
    # 2 Filters (ALT, TYPE)
    def terrain_gen(self):
        generator = Map_generator(self.size)
        my_terrain = generator.meshgrid_combine(reps=1000)
        generator.colormap(my_terrain)
        return my_terrain

if __name__ == "__main__":
    # Test map
    my_map = Map(100)
    grid = my_map.terrain_gen()