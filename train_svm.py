import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

def train_svm(X, y):  
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    avg_a = 0
    avg_f1 = 0
    avg_p = 0
    avg_r = 0

    # training for hogs
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)

        svm_pred=clf.predict(X_test)
        avg_a += metrics.accuracy_score(y_test, svm_pred)
        avg_f1 += metrics.f1_score(y_test, svm_pred,  average='weighted')
        avg_p += metrics.precision_score(y_test, svm_pred,  average='weighted', zero_division=1)
        avg_r += metrics.recall_score(y_test, svm_pred,  average='weighted', zero_division=1)
    
    print(f"Precision Score:{(avg_p/5):.3f}")
    print(f"Recall Score:{(avg_r/5):.3f}")
    print(f"F1 Score:{(avg_f1/5):.3f}")
    print(f"Accuracy Score:{(avg_a/5):.3f}")

def evaluate(avg_pain, samples, angles):
    #split data into training and testing sets
    hogs = np.array(samples, dtype=object)
    angles = np.array(angles, dtype=object)
    print(hogs.shape)
    
    pca = PCA(n_components=0.98)
    pca.fit(hogs)
    hogs_t = pca.transform(hogs)
    y = np.array(avg_pain)
    print(hogs_t.shape)

    print("\nTRAINING HOG FEATURES")
    train_svm(hogs_t, y)
    print("\nTRAINING ANGLE FEATURES")
    train_svm(angles, y)
