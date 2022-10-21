import matplotlib.pyplot as plt
import glob
import os
import cv2
from skimage.io import imread, imshow
from skimage import color
from skimage.feature import hog
import numpy as np

def hog_train_img(path):
    images = []
    names = []
    for image in (glob.glob(path+"/*.png")):
        images.append(cv2.imread(image))
        names.append(os.path.basename(image))

    os.mkdir("hog_features")
    n = 0
    for i in images:
        fd, hog_i = hog(color.rgb2gray(i), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        plt.imshow(hog_i)
        plt.savefig("hog_features/"+str(names[n])+".png")
        n += 1

hog_train_img("train_hog")
    
