import glob
import os
import sys

# python files imports
from pi_ert.pi_ert_training import training, testing
from train_svm import *
from evaluate_hog import *

sys.path.insert(0, 'pi_ert/')


def main():
    # folder containing all the evaluated images of sheep
    path = "../sheep_data/pain_prediction"

    images = []
    names = []
    poses = []
    npy_data = []

    # imported from Excel file
    avg_pain = [
        0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0,
        1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
        0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0
    ]

    # load and convert png images into their numpy form
    for image in sorted(glob.glob(path + "/*.png")):
        im = cv2.imread(image)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        images.append(im)
        names.append(os.path.basename(image))

    # load npy representation of sheep's landmark data points
    for d in sorted(glob.glob(path + "/*.npy")):
        npy_data.append(list((np.load(d, allow_pickle=True))[1]))
        poses.append(list((np.load(d, allow_pickle=True))[2]))

    print("Preparing PI-ERT...")
    training('../sheep_data/landmark_localization', group=[0, 90], forests=5, trees=100)
    print("Training PI-ERT...")
    testing('../sheep_data/landmark_localization', group=[0, 90])
    print("Preparing HOGs...")
    samples, angles = crop_out(images, npy_data, poses, rotate=False)
    rotated_samples, rotated_angles = crop_out(images, npy_data, poses, rotate=True)
    print("Training SVM...")
    evaluate(avg_pain, samples, angles, poses, rotated_samples, rotated_angles)


if __name__ == "__main__":
    main()
