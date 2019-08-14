import cv2
import numpy as np

import astarp

from time import time
from os.path import basename, join, splitext

# maze file
MAZE_FPATH =  'maze_small.png'

OUTP_FPATH = 'maze_small_soln.png'


def main():

    maze = cv2.imread(MAZE_FPATH)
    print('loaded maze of shape %r' % (maze.shape[0:2],))

    grid = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grid[grid == 0] = np.inf
    grid[grid == 255] = 1

    # top left white
    start_j, = np.where(grid[0, :] == 1)
    start = np.array([0, start_j[0]])

    # bottom right white
    end_i, = np.where(grid[:, -1] == 1)
    end = np.array([end_i[0], grid.shape[0] - 1])

    t0 = time()
    
    #list of indices where path is
    path =astarp.astar_path(grid, start, end)
    dur = time() - t0

    if path.shape[0] > 0:
        print('found path of length ',path.shape[0],'in', dur)
        #red path
        maze[path[:, 0], path[:, 1]] = (0, 0, 255)
        print(path)
        print('plotting path to ',OUTP_FPATH)
        cv2.imwrite(OUTP_FPATH, maze)
    else:
        print('no path found')

    print('done')


if __name__ == '__main__':
    main()
