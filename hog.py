import matplotlib.pyplot as plt
import glob
import os
import cv2
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score
from skimage import color
from skimage.feature import hog
import numpy as np

def hog_train_img(path):
    
    images = []
    names = []
    npy_data = []
    npy_names = []
    hog_features = []

    #load and convert png images into their hog representation
    for image in sorted(glob.glob(path+"/*.png")):
        images.append(cv2.imread(image))
        names.append(os.path.basename(image))

    if os.path.isdir("hog_features"):
        pass
    else:
        os.mkdir("hog_features")
        n = 0
        for i in images:
            fd, hog_i = hog(i, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
            hog_features.append(fd)
            plt.imshow(hog_i,cmap=plt.cm.gray)
            plt.savefig("hog_features/"+str(names[n]))
            n += 1

    #load npy representation of sheep's landmark data points
    for d in sorted(glob.glob(path+"/*s.npy")):
        npy_data.append((np.load(d, allow_pickle=True)))
        npy_names.append(os.path.basename(d))

    #save plotted landmarks over sheep faces
    if os.path.isdir("plotted"):
        pass
    else:
        os.mkdir("plotted")
        i = 0
        n = 1
        for im in images:
            plt.imshow(im,cmap=plt.cm.gray)
            for p,q in npy_data[i][1]:
                x = p 
                y = q 
                plt.scatter(x, y, color='red')
                plt.text(x, y, str(n))
                n += 1
            plt.savefig("plotted/"+str(names[i]))
            plt.clf()
            i += 1
            n = 1
    i = 0
    n = 1
    for im in images:
        plt.imshow(im,cmap=plt.cm.gray)
        for p,q in npy_data[i][1]:
            x = p 
            y = q 
            plt.scatter(x, y, color='red')
            plt.text(x, y, str(n))
            n += 1
        plt.show()
        i += 1



hog_train_img("train_hog")



