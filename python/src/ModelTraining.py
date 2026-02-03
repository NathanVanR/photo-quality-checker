import os
import numpy as np

from python.src.GetImageStats import get_image_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def get_x_y(bad_folder, good_folder):
    features_train = []
    labels_train = []

    for filename in os.listdir(bad_folder):
        stats = get_image_stats(os.path.join(bad_folder, filename))
        if stats:
            features_train.append(stats)
            labels_train.append(0)

    for filename in os.listdir(good_folder):
        stats = get_image_stats(os.path.join(good_folder, filename))
        if stats:
            features_train.append(stats)
            labels_train.append(1)

    x = np.array(features_train, dtype=np.float32)
    y = np.array(labels_train)
    return x, y


def train_model():
    bad_folder_train = '../data/training/bad_training'
    good_folder_train = '../data/training/good'
    x_train, y_train = get_x_y(bad_folder_train, good_folder_train)

    mdl = LogisticRegression()
    mdl.fit(x_train, y_train)
    return mdl


def get_predictions(model, bad_folder_test=None, good_folder_test=None):
    if bad_folder_test is None:
        bad_folder_test = '../data/test/bad_test'
    if good_folder_test is None:
        good_folder_test = '../data/test/good'

    x_test, y_test = get_x_y(bad_folder_test, good_folder_test)

    preds = model.predict(x_test)
    print(preds)
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))

