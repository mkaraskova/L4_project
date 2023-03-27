import cv2
from matplotlib import pyplot as plt
from skimage.feature import hog

# python files imports
from angles import *


def crop_out(images, npy_data, poses, rotate):
    samples = []
    angles = []
    for i in range(len(images)):

        # note of important landmarks for each feature
        landmarks = {
            "left_ear": [npy_data[i][0], npy_data[i][21], npy_data[i][22], npy_data[i][1]],
            "right_ear": [npy_data[i][5], npy_data[i][23], npy_data[i][24], npy_data[i][4]],
            "left_eye": [npy_data[i][1], npy_data[i][2]],
            "right_eye": [npy_data[i][4], npy_data[i][3]],
            "nose": [npy_data[i][13], npy_data[i][19], npy_data[i][6], npy_data[i][20], npy_data[i][17],
                     npy_data[i][7], npy_data[i][18], npy_data[i][15], npy_data[i][16]]
        }

        yaw = poses[i][2]
        regions = []
        # creating bounding box for each facial region and cropping it out
        for key, coord in landmarks.items():
            fd_zeros = np.zeros(900)
            if 10 < yaw < 60 and key == 'left_eye':
                regions.append(fd_zeros)
            elif -60 < yaw < -10 and key == 'right_eye':
                regions.append(fd_zeros)
            elif 60 < yaw < 90 and key == 'left_ear':
                regions.append(fd_zeros)
            elif -90 < yaw < -60 and key == 'right_ear':
                regions.append(fd_zeros)
            else:
                x, y, w, h = cv2.boundingRect((np.asarray(coord)).astype(int))
                # if coordinates negative, change to 0
                expansion = 5
                x = max(x - expansion, 0)
                y = max(y - expansion, 0)
                w = min(w + 2 * expansion, images[i].shape[1] - x)
                h = min(h + 2 * expansion, images[i].shape[0] - y)
                if x < 0:
                    x = 0
                elif y < 0:
                    y = 0
                # Crop the image to the expanded bounding rectangle
                crop_img = images[i][y:y + h, x:x + w]
                crop_img = cv2.resize(crop_img, (100, 100))
                if rotate:
                    crop_img = cv2.flip(crop_img, 1)

                fd, hog_img = hog(crop_img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                                  visualize=True, channel_axis=-1)
                regions.append(fd)

        angles.append(calc_angles(i, npy_data))
        samples.append(np.array(regions).flatten())

    return samples, angles
