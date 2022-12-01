import math
import numpy as np

def calc_angles(i, npy_data):

    angles = []
    angles.append(math.degrees(math.atan2(npy_data[i][1][0][1] - npy_data[i][1][1][1], npy_data[i][1][0][0] - npy_data[i][1][1][0])))
    angles.append(math.degrees(math.atan2(npy_data[i][1][4][1] - npy_data[i][1][5][1], npy_data[i][1][4][0] - npy_data[i][1][5][0])))
    angles.append(math.degrees(math.atan2(npy_data[i][1][1][1] - npy_data[i][1][4][1], npy_data[i][1][1][0] - npy_data[i][1][4][0])))

    return np.array(angles)