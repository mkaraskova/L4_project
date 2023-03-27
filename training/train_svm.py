import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


def train_svm(X, y, k):
        kf = KFold(n_splits=k)
        kf.get_n_splits(X)
        # save trained svm models
        models = []

        avg_a = []
        avg_f1 = []
        avg_p = []
        avg_r = []

        cms = []

        # training for hogs
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = svm.SVC(kernel='linear', C=0.1)
            clf.fit(X_train, y_train)
            models.append(clf)

            svm_pred = clf.predict(X_test)
            cms.append(metrics.confusion_matrix(y_test, svm_pred))
            avg_a.append(metrics.accuracy_score(y_test, svm_pred))
            avg_f1.append(metrics.f1_score(y_test, svm_pred, average='weighted'))
            avg_p.append(metrics.precision_score(y_test, svm_pred, average='weighted', zero_division=1))
            avg_r.append(metrics.recall_score(y_test, svm_pred, average='weighted', zero_division=1))

        print("Confusion Matrix for fold k = {}:".format(k))
        # cm = np.sum(cms, axis=0) / len(cms)
        # sns.heatmap(cm, annot=True, cmap='Blues')
        # plt.title('Confusion Matrix')
        # plt.xlabel('Predicted Label')
        # plt.ylabel('True Label')
        # Show the plot
        # plt.show()

        print(f"Precision Score:{np.mean(avg_p):.2f}")
        print(f"Recall Score:{np.mean(avg_r):.2f}")
        print(f"F1 Score:{np.mean(avg_f1):.2f}")
        print(f"Accuracy Score:{np.mean(avg_a):.2f}")

        return models


def evaluate(avg_pain, samples, angles, poses, rotated_samples, rotated_angles):
    # split data into training and testing sets
    hogs = np.array(samples, dtype=object)
    rotated_hogs = np.array(rotated_samples, dtype=object)
    angles = np.array(angles, dtype=object)
    rotated_angles = np.array(rotated_angles, dtype=object)
    y = np.array(avg_pain)

    combined_y = []
    combined_y.extend(avg_pain * 2)
    combined_y = np.array(combined_y)
    combined_hogs = np.concatenate((hogs, rotated_hogs), axis=0)
    combined_original = np.concatenate((hogs, angles), axis=1)
    combined_rotated = np.concatenate((rotated_hogs, rotated_angles), axis=1)

    combined = np.concatenate((combined_original, combined_rotated), axis=0)

    print("\nTRAINING HOG FEATURES")
    print(combined_hogs.shape)
    train_svm(combined_hogs, combined_y, 5)

    print("\nTRAINING ANGLE FEATURES")
    print(angles.shape)
    train_svm(angles, y, 5)

    print("\nTRAINING COMBINED FEATURES")
    print(combined.shape)
    models = train_svm(combined, combined_y, 10)

    joblib.dump(models, "svm_models.pkl")
