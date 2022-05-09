import pandas as pd
import numpy as np
import cv2
import math
import os
from PIL import Image
import pywt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


def resize_and_save_img(src, destination_path):
    original_img = cv2.imread(src)
    old_image_height, old_image_width, channels = original_img.shape
    new_image_width = 60        
    new_image_height = 60
    color = (255,255,255)

    result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)

    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # Centrar imagen
    result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = original_img

    Image.fromarray(result).save(destination_path)



def generate_new_data(): 
    ''' Se corre solo una vez,
    junta la data de Train y Test en un solo directorio unificado'''

    train_dir = "Data/Train/"
    test_dir = "Data/Test/"
    destination_dir = "Data_preprocesada/"

    for class_dir in os.listdir(train_dir):
        for train_img in os.listdir(train_dir+class_dir):
            resize_and_save_img(f"{train_dir}{class_dir}/{train_img}", f"{destination_dir}{class_dir}/{train_img}")

    test_info = pd.read_csv("Data/Test.csv")
    for i, test_img in enumerate(sorted(os.listdir(test_dir))):
        resize_and_save_img(f"{test_dir}{test_img}", f"{destination_dir}{test_info.ClassId[i]}/{test_img}")



def get_vector_from_image(image, iterations):
    LL, (LH, HL, HH) = pywt.dwt2(image, 'haar')
    for _ in range(iterations - 1):
        LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
    return LL, LL.flatten()



def get_data(src_dir, iterations):
    x = []
    y = []
    raw_x = []

    for class_dir in os.listdir(src_dir):
        for train_img in os.listdir(src_dir + class_dir):
            image_path = f"{src_dir}{class_dir}/{train_img}"
            img = Image.open(image_path)

            fv = get_vector_from_image(img, iterations)
            raw_x.append(fv[0])
            x.append(fv[1])
            y.append(int(class_dir))
    return np.asarray(x), np.asarray(y), np.asarray(raw_x)


def iterate_data(X_raw):
    X = []
    X_new_raw = []
    for i in range(X_raw.shape[0]):
        LL , (LH, HL, HH) = pywt.dwt2(X_raw[i], 'haar')
        X_new_raw.append(LL)
        X.append(LL.flatten())
    return np.asarray(X), np.asarray(X_new_raw)



def normalization(data):
    columns = data.transpose()
    normalized_data = []
    for column in columns:
        minimum = min(column)
        maximum = max(column)
        normalized_column = np.asarray([(n - minimum) / (maximum - minimum) for n in column])
        normalized_data.append(normalized_column)
    return np.asarray(normalized_data).transpose() 



def unison_shuffled_copies(a, b, random_seed):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=random_seed).permutation(len(a))
    return a[p], b[p]


def get_stratified_k_fold_cross_validation(X, y, number_of_folds, random_seed):
    skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=random_seed)
    skf.get_n_splits(X, y)
    k_folds = []
    for train_index, test_index in skf.split(X, y):
        fold = {}
        fold['X_train'] = X[train_index]
        fold['X_test'] = X[test_index]
        fold['y_train'] = y[train_index]
        fold['y_test'] = y[test_index]
        k_folds.append(fold)
    return k_folds



def get_non_stratified_k_fold_cross_validation(X, y, number_of_folds, random_seed):
    X, y = unison_shuffled_copies(X,y,random_seed)
    kf = KFold(n_splits=number_of_folds, shuffle=True, random_state=random_seed)
    kf.get_n_splits(X)
    k_folds = []
    for train_index, test_index in kf.split(X):
        fold = {}
        fold['X_train'] = X[train_index]
        fold['X_test'] = X[test_index]
        fold['y_train'] = y[train_index]
        fold['y_test'] = y[test_index]
        k_folds.append(fold)
    return k_folds



def resample_x_and_y(X, y, training_sample, random_seed):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    index_used = [False for _ in range(len(X))]

    # training
    for _ in range(math.floor(len(X) * training_sample)):
        new_index = math.floor(np.RandomState(seed=random_seed).rand() * len(X))
        index_used[new_index] = True
        # print(math.floor(np.random.rand() * len(X)))
        X_train.append(X[new_index])
        y_train.append(y[new_index])

    # testing
    for i in range(len(X)):
        if not index_used[i]:
            X_test.append(X[i])
            y_test.append(y[i])

    return X_train, y_train, X_test, y_test


def get_bootstrap_subsets(X, y, k, training_sample, random_seed):
    subsets = []
    for _ in range(k):
        X_train, y_train, X_test, y_test = resample_x_and_y(X, y, training_sample, random_seed)
        subset = {}
        subset['X_train'] = X_train
        subset['X_test'] = X_test
        subset['y_train'] = y_train
        subset['y_test'] = y_test
        subsets.append(subset)
    return subsets