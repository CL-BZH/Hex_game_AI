
import matplotlib.pyplot as plt
import numpy as np
import math

def bell(board_size, min_runs):

    total_cells = board_size*board_size
    max_runs = total_cells * 25
    center = 7 * total_cells / 8
    available_cells = np.array([i for i in range(total_cells)])
    sigma_square = total_cells
    delta = available_cells - center
    delta_sqr = delta * delta
    runs = max_runs* np.exp(-0.5 * delta_sqr/ sigma_square)
    runs[runs < min_runs] = min_runs
    plt.plot(available_cells, runs)
    plt.xlabel("Number of available cells on the board")
    plt.ylabel("Number of runs")
    #plt.ylim(min_runs, np.max(runs)+10)
    ticks = [runs for runs in range(min_runs, int(np.max(runs))) if (runs%100 == 0)]

    plt.yticks(ticks)
    plt.show()

if __name__ == "__main__":
    bell(9, 100)
