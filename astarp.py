import ctypes
import numpy as np

import inspect
from os.path import abspath, dirname, join


#fuckall gay shit to play nice with cpp

fname = abspath(inspect.getfile(inspect.currentframe()))

lib = ctypes.cdll.LoadLibrary(join(dirname(fname), 'astar.so'))

astar = lib.astar
ndmat_f_type = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
ndmat_i_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
astar.restype = ctypes.c_bool
astar.argtypes = [ndmat_f_type, ctypes.c_int, ctypes.c_int,
                  ctypes.c_int, ctypes.c_int, ctypes.c_bool,
                  ndmat_i_type]


def astar_path(weights, start, goal, allow_diagonal=True):

    height, width = weights.shape
    start_idx = np.ravel_multi_index(start, (height, width))
    goal_idx = np.ravel_multi_index(goal, (height, width))

    # cpp code
    paths = np.full(height * width, -1, dtype=np.int32)
    success = astar(
        weights.flatten(), height, width, start_idx, goal_idx, allow_diagonal,
        paths  # output parameter
    )
    if not success:
        return np.array([])

    coordinates = []
    path_idx = goal_idx
    while path_idx != start_idx:
        pi, pj = np.unravel_index(path_idx, (height, width))
        coordinates.append((pi, pj))

        path_idx = paths[path_idx]

    if coordinates:
        # splits path into tuples
        coordinates.append(np.unravel_index(start_idx, (height, width)))
        return np.vstack(coordinates[::-1])
    else:
        return np.array([])
