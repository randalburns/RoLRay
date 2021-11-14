import ray
import numpy as np

@ray.remote
class RoLSubGrid(object):
    """
    class RoLSubGrid: Rules of life implemented in Ray.

        This class updates local paritions and then pulls data from
        adjacent partitions. It implements the ghost cell patter with
        one cell of overlap.

        The class should be invoked with two barriers per iteration.
            * update all local partitions
            * barrier
            * swap data between partitions
            * barrier
    """
    # Create an actor each with its own data
    def __init__(self, dim, has_glider=False):
        self.local_grid = np.zeros(shape=[dim + 2, dim + 2], dtype=np.uint8)
        if has_glider == True:
            glider = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.uint8)
            self.local_grid[1:glider.shape[0] + 1, 1:glider.shape[1] + 1] = glider

    # neighbors for 2-d GoL gride
    def set_neighbors(self, top, bottom, left, right, tl, tr, bl, br):
        self.tl = tl
        self.bl = bl
        self.tr = tr
        self.br = br
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def get_neighbors(self):
        return self.top, self.bottom, self.left, self.right, self.tl, self.bl, self.tr, self.br

    def get_grid(self):
        return self.local_grid

    def step(self):
        """Evaluate the rules of life on a 2-d subarray.
        The array should have an overlap of 1 cell in all dimension
        and on the corner.

        Args:

        Returns:
            outgrid (ndarray): Array updated by rules of life
        """
        # we will receive an array with 1 dimension of padding
        xdim, ydim = self.local_grid.shape

        # output array to keep updates
        outgrid = np.zeros(shape=self.local_grid.shape, dtype=self.local_grid.dtype)

        # update only in center (non-overlapping) regaion
        for x in range(1, xdim - 1):
            for y in range(1, ydim - 1):
                sum = self.local_grid[x - 1, y - 1] + self.local_grid[x, y - 1] + self.local_grid[x + 1, y - 1] + \
                      self.local_grid[x - 1, y] + self.local_grid[x + 1, y] + \
                      self.local_grid[x - 1, y + 1] + self.local_grid[x, y + 1] + self.local_grid[x + 1, y + 1]
                # three neighbors birth
                if (sum == 3):
                    outgrid[x, y] = 1
                # two neighbors no change
                elif (sum == 2):
                    outgrid[x, y] = self.local_grid[x, y]
                    # <2 or >3 death
                else:
                    outgrid[x, y] = 0

        self.local_grid = outgrid

    def cut(self, xl, xh, yl, yh):
        """Return a slice of the local array"""
        return self.local_grid[xl+1:xh+1, yl+1:yh+1]

    def exchange(self):
        """Transfer data from neighbors. Corners and sides."""
        tloid = self.tl.cut.remote(7, 8, 7, 8)  # tl cuts br
        troid = self.tr.cut.remote(7, 8, 0, 1)  # tr cuts bl
        bloid = self.bl.cut.remote(0, 1, 7, 8)  # bl cuts tr
        broid = self.br.cut.remote(0, 1, 0, 1)  # br cuts tl
        toid = self.top.cut.remote(7, 8, 0, 8)  # top
        boid = self.bottom.cut.remote(0, 1, 0, 8)  # bottom
        loid = self.left.cut.remote(0, 8, 7, 8)  # left
        roid = self.right.cut.remote(0, 8, 0, 1)  # right

        self.local_grid[0, 0] = ray.get(tloid)  # tl cuts br
        self.local_grid[0, 9] = ray.get(troid)  # tr cuts bl
        self.local_grid[9, 0] = ray.get(bloid)  # bl cuts tr
        self.local_grid[9, 9] = ray.get(broid)  # br cuts tl
        self.local_grid[0:1, 1:9] = ray.get(toid)  # top
        self.local_grid[9:10, 1:9] = ray.get(boid)  # bottom
        self.local_grid[1:9, 0:1] = ray.get(loid)  # left
        self.local_grid[1:9, 9:10] = ray.get(roid)  # right

# script to drive parallel program
ray.init(num_cpus=4, ignore_reinit_error=True)

# 2x2 array of partitions
dim = 2

# ray objects for actors
oids = np.empty([dim,dim], dtype=object)

# this list will be used to implement barriers, i.e. wait for the completion of a set of jobs
roids = []

# create grids
for ix in range(dim):
    for iy in range(dim):
        if ix == 0 and iy == 0:
            oids[ix, iy] = RoLSubGrid.remote(8, True)
        else:
            oids[ix, iy] = RoLSubGrid.remote(8, False)
        print(f"({ix},{iy}) {oids[ix,iy]}")

# set neighbors T B L R TL TR BL BR
for ix in range(dim):
    for iy in range(dim):
        roids.append(oids[ix, iy].set_neighbors.remote(
            oids[(ix - 1) % dim, iy], oids[(ix + 1) % dim, iy], oids[ix, (iy - 1) % dim], oids[ix, (iy + 1) % dim],
            oids[(ix - 1) % dim, (iy - 1) % dim], oids[(ix - 1) % dim, (iy + 1) % dim],
            oids[(ix + 1) % dim, (iy - 1) % dim], oids[(ix + 1) % dim, (iy + 1) % dim]))

# await the initialization of all neighbors
ray.wait(roids)
# let's look at the neighbor list
for ix in range(dim):
    for iy in range(dim):
        print(ray.get(oids[ix,iy].get_neighbors.remote()))

# 64 iterations will return the glider home
for it in range(68):
    if (it % 4 == 0):
        # Glider only lives in top left and bottom right really
        print("Step")
        print(ray.get(oids[0, 0].get_grid.remote()))
        #print(ray.get(oids[0, 1].get_grid.remote()))
        #print(ray.get(oids[1, 0].get_grid.remote()))
        print(ray.get(oids[1, 1].get_grid.remote()))

    # asychronously update the local grids
    roids = []
    for ix in range(dim):
        for iy in range(dim):
            roids.append(oids[ix, iy].step.remote())
    # first barrier awaiting local updates
    ray.wait(roids)

    # exchange data among sides and corners.
    roids = []
    for ix in range(dim):
        for iy in range(dim):
            #print(f"Exchange {ix,iy}")
            ray.get(oids[ix,iy].exchange.remote())
    # second barrier awaiting exchanges
    ray.wait(roids)

ray.shutdown()
