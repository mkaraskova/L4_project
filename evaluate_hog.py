import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog

# python files imports
from angles import *
from train_svm import *

def crop_out(images, npy_data):
    img_no = 1
    if os.path.isdir("regions"):
        pass
    else:
        os.mkdir("regions")
        for i in range(len(images)):
            # note of important landmarks for each feature
            landmarks = {
            "left_ear" :[npy_data[i][1][0], npy_data[i][1][21], npy_data[i][1][22], npy_data[i][1][1]],
            "right_ear" : [npy_data[i][1][5], npy_data[i][1][23], npy_data[i][1][24], npy_data[i][1][4]],
            "left_eye" : [npy_data[i][1][1],npy_data[i][1][2]],
            "right_eye" : [npy_data[i][1][4], npy_data[i][1][3]],
            "nose" : [npy_data[i][1][13], npy_data[i][1][19], npy_data[i][1][6], npy_data[i][1][20], npy_data[i][1][17], npy_data[i][1][7], npy_data[i][1][18], npy_data[i][1][15], npy_data[i][1][16]]
            }

            # note of landmarks in order to calculate geometric features
            geometric_features = {
            "ear_roots":[npy_data[i][1][1],npy_data[i][1][4]],
            "left_ear":[npy_data[i][1][0],npy_data[i][1][1]],
            "right_ear":[npy_data[i][1][5],npy_data[i][1][4]],
            }

            # save angles info
            angles = calc_angles(images,npy_data)   
            np.save("regions/" + str(img_no) + '_' + "angles",angles,allow_pickle=True) 

            # creating bounding box for each facial region and cropping it out
            reg_no = 1
            for coord in landmarks.values():
                x,y,w,h = cv2.boundingRect(np.asarray(coord))
                #cv2.rectangle(images[i],(x-10,y-10),(x+w+15,y+h+15),(255,0,0),1)
                try:
                    crop_img = images[i][y-10:y+h+10, x-10:x+w+10]
                    fd,hog_im = hog(crop_img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
                except:
                    crop_img = images[i][y:y+h, x:x+w]
                    fd,hog_im = hog(crop_img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
                plt.imshow(hog_im,cmap=plt.cm.gray)
                plt.savefig("regions/" + str(img_no) + '_' + str(reg_no))
                np.save("regions/" + str(img_no) + '_' + str(reg_no),fd,allow_pickle=True) 

                reg_no +=  1
            img_no += 1



