import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage.feature import hog

# python files imports
from angles import *

def crop_out(images, npy_data, names):
    samples = []
    labels = []
    angles = []
    if os.path.isdir("try"):
        pass
    else:
        #os.mkdir("try")
        for i in range(len(images)):
            # note of important landmarks for each feature
            landmarks = {
            "left_ear" :[npy_data[i][1][0], npy_data[i][1][21], npy_data[i][1][22], npy_data[i][1][1]],
            "right_ear" : [npy_data[i][1][5], npy_data[i][1][23], npy_data[i][1][24], npy_data[i][1][4]],
            "left_eye" : [npy_data[i][1][1],npy_data[i][1][2]],
            "right_eye" : [npy_data[i][1][4], npy_data[i][1][3]],
            "nose" : [npy_data[i][1][13], npy_data[i][1][19], npy_data[i][1][6], npy_data[i][1][20], npy_data[i][1][17], npy_data[i][1][7], npy_data[i][1][18], npy_data[i][1][15], npy_data[i][1][16]]
            }

            regions = []
            # creating bounding box for each facial region and cropping it out
            for key, coord in landmarks.items():
                x,y,w,h = cv2.boundingRect(np.asarray(coord))
                #cv2.rectangle(images[i],(x-10,y-10),(x+w+15,y+h+15),(255,0,0),1)
                try:
                    crop_img = images[i][y-10:y+h+10, x-10:x+w+10]
                    crop_img = cv2.resize(crop_img, (100,100))
                    fd,hog_img = hog(crop_img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
                except:
                    crop_img = images[i][y:y+h, x:x+w]
                    crop_img = cv2.resize(crop_img, (100,100))
                    fd,hog_img = hog(crop_img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
                regions.append(fd)
                #plt.imshow(hog_im,cmap=plt.cm.gray)
                #plt.savefig("regions/" + cur_name + "_" + key)
                #np.save("regions/" + cur_name + "_" + key,fd,allow_pickle=True) 
            angles.append(calc_angles(i, npy_data))
            labels.append(names[i])
            samples.append(np.array(regions).flatten())
            
    return labels, samples, angles