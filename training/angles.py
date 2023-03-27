import math
import numpy as np


def calc_angles(i, npy_data):
    # angles between tip and root of ears, distance between ear roots
    angles = [math.degrees(math.atan2(npy_data[i][0][1] - npy_data[i][1][1], npy_data[i][0][0] - npy_data[i][1][0])),
              math.degrees(math.atan2(npy_data[i][4][1] - npy_data[i][5][1], npy_data[i][4][0] - npy_data[i][5][0])),
              math.dist(npy_data[i][1], npy_data[i][5])]

    return np.array(angles)
