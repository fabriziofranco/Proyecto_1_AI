import pandas as pd
import numpy as np
import cv2
import math
import os
from PIL import Image
import pywt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

random_seed = np.random.seed(42)

def resize_and_save_img(src, destination_path):
    """
    resize_and_save_img aplica un resize a la imagen en src y la guarda en destination_path.

    :param src: Dirección a la imagen a la que se le desea hacer resize.
    :param destination_path: Dirección en la cual se debe guardar la imagen procesada.
    :return: None.
    """
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
    """
    generate_new_data se corre solo una vez. Junta la data de Train y Test en un solo directorio unificado.

    :return: None.
    """
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
    """
    get_vector_from_image obtiene el vector característico de la imagen image

    :param image: Imagen en formato vector.
    :param iterations: Entero que indica la cantidad de veces que se aplica el wavelet a la imagen.
    :return LL: Vector característico sin la compresión a 1D.
    :return LL.flatten(): Vector característico en 1D.
    """
    LL, (LH, HL, HH) = pywt.dwt2(image, 'haar')
    for _ in range(iterations - 1):
        LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
    return LL, LL.flatten()



def get_data(src_dir, iterations):
    """
    get_data

    :param src_dir: Directorio origen para leer las imágenes.
    :param iterations: Entero que indica la cantidad de veces que se aplica el wavelet a la imagen.
    :return np.asarray(x): Vector con los vectores característicos de las imágenes en 1D.
    :return np.asarray(y): Vector con los labels correspondientes a los vectores característicos.
    :return np.asarray(raw_x): Vector con los vectores característicos de las imágenes sin la compresión a 1D.
    """
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
    """
    iterate_data aplica una compresión adicional a los datos de X_raw

    :param X_raw: Data X sin comprimir.
    :return np.asarray(X): Nuevos vectores característicos en 1D.
    :return np.asarray(X_new_raw): Nuevos vectores característicos sin la compresión a 1D.
    """
    X = []
    X_new_raw = []
    for i in range(X_raw.shape[0]):
        LL , (LH, HL, HH) = pywt.dwt2(X_raw[i], 'haar')
        X_new_raw.append(LL)
        X.append(LL.flatten())
    return np.asarray(X), np.asarray(X_new_raw)



def normalization(data):
    """
    normalization aplica la normalización a un conjunto de datos.

    :param data: Datos a comprimir
    :return np.asarray(normalized_data).transpose(): Conjunto de datos normalizados.
    """
    columns = data.transpose()
    normalized_data = []
    for column in columns:
        minimum = min(column)
        maximum = max(column)
        normalized_column = np.asarray([(n - minimum) / (maximum - minimum) for n in column])
        normalized_data.append(normalized_column)
    return np.asarray(normalized_data).transpose() 



def unison_shuffled_copies(a, b, random_seed):
    """
    unison_shuffled_copies aplica una misma permutación a dos arreglos
    
    :param a: Primer arreglo.
    :param b: Segundo arreglo.
    :param random_seed: Semilla.
    :return a[p]: Arreglo a permutado.
    :return b[p]: Arreglo b permutado.
    """
    assert len(a) == len(b)
    p = np.random.RandomState(seed=random_seed).permutation(len(a))
    return a[p], b[p]


def get_stratified_k_fold_cross_validation(X, y, number_of_folds, random_seed):
    """
    get_stratified_k_fold_cross_validation genera K folds estratificados.

    :param X: Arreglo con los vectores característicos.
    :param y: Arreglo con los labels.
    :param number_of_folds: Cantidad de k folds.
    :param random_seed: Semilla.
    :return k_folds: K folds estratificados.
    """
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
    """
    get_non_stratified_k_fold_cross_validation genera K folds no estratificados.

    :param X: Arreglo con los vectores característicos.
    :param y: Arreglo con los labels.
    :param number_of_folds: Cantidad de k folds.
    :param random_seed: Semilla.
    :return k_folds: K folds no estratificados.
    """
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

def resample_x_and_y(X, y, training_sample):
    """
    resample_x_and_y es una simulación del bootstrap validation.

    :param X: Arreglo con los vectores característicos.
    :param y: Arreglo con los labels.
    :param training_sample: Porcentaje de datos de training.
    :return X_train, y_train, X_test, y_test: Bootstrap.
    """
    indices_train = np.random.randint(low = 0, high = len(X), size = math.floor(len(X) * training_sample))
    indices_train_unicos = np.unique(indices_train).tolist()
    X_train = X[indices_train]
    y_train = y[indices_train]

    X_test =  np.delete(X, indices_train_unicos, axis = 0)
    y_test =  np.delete(y, indices_train_unicos, axis = 0)

    return X_train, y_train, X_test, y_test


def get_bootstrap_subsets(X, y, k, training_sample, random_seed):
    """
    get_bootstrap_subsets obtiene k bootstrap subsets para X y y.


    :param X: Arreglo con los vectores característicos.
    :param y: Arreglo con los labels.
    :param training_sample: Porcentaje de datos de training.
    :random_seed: Semilla.
    :return subsets: K subsets bootstrap
    """
    np.random.seed(random_seed)
    subsets = []
    for _ in range(k):
        X_train, y_train, X_test, y_test = resample_x_and_y(X, y, training_sample)
        subset = {}
        subset['X_train'] = X_train
        subset['X_test'] = X_test
        subset['y_train'] = y_train
        subset['y_test'] = y_test
        subsets.append(subset)
    return subsets
