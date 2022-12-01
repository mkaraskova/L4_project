import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog


# python files imports
from angles import *
from train_svm import *
from evaluate_hog import *

def plot_out(images,names,npy_data):
    #save plotted, npy_data over sheep faces
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

    # save generated hog representation of sheep images
    if os.path.isdir("hog_features"):
        pass
    else:
        os.mkdir("hog_features")
        n = 0
        for i in images:
            fd,hog_im = hog(i, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
            plt.imshow(hog_im,cmap=plt.cm.gray)
            plt.savefig("hog_features/"+str(names[n]))
            n += 1

def main():
    # folder containing all the evaluated images of sheep
    path = "data"

    images = []
    names = []
    npy_data = []
    
    avg_pain = [0.40,0.67,0.50,0.33,0.50,0.33,0.67,0.33,0.33,0.00,
                0.00,0.00,0.00,0.00,1.33,0.25,0.00,0.00,0.00,0.67,
                0.33,1.33,0.00,0.25,0.50,0.33,0.33,0.25,0.00,0.33,
                0.33,0.75,0.00,0.33,0.67,0.75,0.40,1.00,0.50,0.67,
                0.00,0.00,0.67]
    avg_pain_f = [0,1,0,0,0,0,1,0,0,0,
                0,0,0,0,1,0,0,0,0,1,
                0,1,0,0,0,0,0,0,0,0,
                0,1,0,0,1,1,0,1,0,1,
                0,0,1]

    #load and convert png images into their numpy form
    for image in sorted(glob.glob(path+"/*.png")):
        im = cv2.imread(image)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        images.append(im)
        names.append(os.path.basename(image))

    #load npy representation of sheep's landmark data points
    for d in sorted(glob.glob(path+"/*s.npy")):
        npy_data.append((np.load(d, allow_pickle=True)))

    print("Plotting...")
    plot_out(images,names,npy_data)
    print("Cropping...")
    labels, samples,angles = crop_out(images, npy_data, names)
    print("Training...")
    evaluate(avg_pain_f, samples, angles)

if __name__ == "__main__":
    main()