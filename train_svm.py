import cv2
import os
from matplotlib.scale import LinearScale
from sklearn.svm import LinearSVC
from skimage import feature

def train_svm(path):

    hog_features = []
    names = []
    for file in os.listdir(path):
        if file.endswith(".png"):
            im = cv2.imread(os.path.join(path,file))
            im = cv2.resize(im,(128*4,64*4))
            names.append(os.path.basename(file))
            features, image = feature.hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
            hog_features.append(features)

    print('Training...')
    svc = LinearSVC()
    svc.fit(hog_features,names)

    print('Evaluating...')
    path = "test"
    names = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            im = cv2.imread(os.path.join(path,file))
            im = cv2.resize(im,(128*4,64*4))
            names.append(os.path.basename(file))
            features, image = feature.hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
            prediction = svc.predict(features.reshape(1,-1))[0]
            image = image.astype('float64')
            cv2.imshow('Test',image)
            cv2.putText(image, prediction.title(), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow('Test',image)
            cv2.waitKey(0)
