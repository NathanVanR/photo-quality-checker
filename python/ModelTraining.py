# %%
# %load_ext autoreload
# %autoreload 2

import os
import numpy as np
from GetImageStats import get_image_stats

features = []
labels = []

bad_folder = '../data/training/bad_training'
for filename in os.listdir(bad_folder):
    stats = get_image_stats(os.path.join(bad_folder, filename))
    if stats:
        features.append(stats)
        labels.append(0)

good_folder = '../data/training/good'
for filename in os.listdir(good_folder):
    stats = get_image_stats(os.path.join(good_folder, filename))
    if stats:
        features.append(stats)
        labels.append(1)

X_train = np.array(features, dtype=np.float32)
y_train = np.array(labels)
# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# %%
features_test = []
labels_test = []


bad_folder_test = '../data/test/bad_test'
for filename in os.listdir(bad_folder_test):
    stats = get_image_stats(os.path.join(bad_folder_test, filename))
    if stats:
        features_test.append(stats)
        labels_test.append(0)

good_folder_test = '../data/test/good'
for filename in os.listdir(good_folder_test):
    stats = get_image_stats(os.path.join(good_folder_test, filename))
    if stats:
        features_test.append(stats)
        labels_test.append(1)


X_test = np.array(features, dtype=np.float32)
y_test = np.array(labels)

preds = model.predict(X_test)
print(preds)
# %%
from sklearn.metrics import classification_report, confusion_matrix

print("Confusion Matrix:\n",confusion_matrix(y_test, preds))
print("Classification Report:\n",classification_report(y_test, preds))