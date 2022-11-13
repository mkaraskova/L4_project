import math

def calc_angles(images, npy_data):

    angles = {}
    for i in range(len(images)):
        angles[i] = {}
        angles[i]["left_ear"] = math.degrees(math.atan2(npy_data[i][1][0][1] - npy_data[i][1][1][1], npy_data[i][1][0][0] - npy_data[i][1][1][0]))
        angles[i]["right_ear"] = math.degrees(math.atan2(npy_data[i][1][4][1] - npy_data[i][1][5][1], npy_data[i][1][4][0] - npy_data[i][1][5][0]))
        angles[i]["ear_roots"] = math.degrees(math.atan2(npy_data[i][1][1][1] - npy_data[i][1][4][1], npy_data[i][1][1][0] - npy_data[i][1][4][0]))

    return angles