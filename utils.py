"""
utils.py
=========

Description:
This module provides utility functions to support our hands-on session on adversarial machine learning.
Typically, it includes tools for dataset handling, subsampling, and visualization.

Usage:
Import this module in your main notebook or script.
"""
import six
import sys
import os
import numpy as np
from art.utils import get_file
from art import config
import math
import matplotlib.pyplot as plt
import random as rd
from sklearn.decomposition import PCA
from keras import layers, models
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
import pickle

def random_subsample(x, ind_list=None, n=None):
    """
    Select a random subsample from several arrays, using common indices

    :param x: The arrays which subsample must be extracted.
    :param ind_list (None): The indices to be extracted.
    :param n (None): The number of examples to extract. If n=len(x), the function simply shuffles the arrays.
    :return: The subsamples for all
    """    
    if n is None:
        n = len(x[0])
    if ind_list is None:
        ind_list = list(range(len(x[0])))
        np.random.shuffle(ind_list)
    
    for i, xi in enumerate(x):
        x[i] = x[i][ind_list[:n]]
    return x

def to_categorical(labels, nb_classes) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`.
    :param nb_classes: The number of classes (possible labels).
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=int)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical

def show_images(x, y, y_fine=None, n=5, n_cols=3):
    """
    Show images in a grid, from the dataset with their label.

    :param x: the set of images
    :param y: the corresponding super classes
    :param y_fine (None): the corresponding fine classes
    :param n (5): the number of images to display
    :param n_cols (3): the number of columns for the grid of images
    """
    n = min(n, len(y))
    fine = y_fine is not None
    cols = min(math.ceil(math.sqrt(n)), n_cols)
    rows = math.ceil(n/cols)
    im_size=3
    fig, axs = plt.subplots(rows, cols, figsize=(cols*im_size,rows*im_size))
    
    indices_seen = []
    for r in range(rows):
        for c in range(cols):
            i = r*cols+c
            if i >= n or i >= len(y):
                break
            
            rd_int = rd.randint(0, len(y)-1)
            while rd_int in indices_seen:
                rd_int = rd.randint(0, len(y)-1)
            indices_seen.append(rd_int)
            
            if rows == 1:
                axs[c].imshow(x[rd_int])
                if fine:
                    axs[c].set_title(f"{rd_int}: {y[rd_int]}\n{y_fine[rd_int]}")
                else:
                    axs[c].set_title(f"{rd_int}: {y[rd_int]}")
                axs[c].axis('off')
            else:
                axs[r,c].imshow(x[rd_int])
                if fine:
                    axs[r,c].set_title(f"{rd_int}: {y[rd_int]}\n{y_fine[rd_int]}")
                else:
                    axs[r,c].set_title(f"{rd_int}: {y[rd_int]}")
                axs[r,c].axis('off')

            plt.tight_layout()
    plt.show()

def plot_images_pca(x, y=None, num_components=2, num_samples=None, title='PCA projection of images'):
    """
    Projects images to 2D using PCA and plots them as points colored by label.
    
    Parameters:
    -----------
    x : np.ndarray
        Array of images with shape (N, H, W, C) or (N, D).
    y : np.ndarray or list, optional
        Class labels for coloring the points.
    num_components : int
        Number of PCA components (default: 2 for 2D).
    num_samples : int
        Number of random images to sample for visualization.
    title : str
        Title of the plot.
    """
    # Flatten images if needed
    if len(x.shape) > 2:
        images_flat = x.reshape(x.shape[0], -1)
    else:
        images_flat = x
    
    if num_samples is None:
        num_samples = len(x)

    # Subsample if too large
    if images_flat.shape[0] > num_samples:
        idx = np.random.choice(images_flat.shape[0], num_samples, replace=False)
        images_flat = images_flat[idx]
        if y is not None:
            y = np.array(y)[idx]

    # Apply PCA
    pca = PCA(n_components=num_components)
    images_pca = pca.fit_transform(images_flat)

    # Plot
    plt.figure(figsize=(8, 6))
    if y is not None:
        scatter = plt.scatter(images_pca[:, 0], images_pca[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classes", loc="best", fontsize=8)
    else:
        plt.scatter(images_pca[:, 0], images_pca[:, 1], s=10, alpha=0.7)

    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.show()

def create_model_cifar(num_classes=20):
    """
    Build a CNN model, and compile it with usual settings.

    :param num_classes: the number of classes used for classification
    :return: the model
    """
    # inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    # base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    # base_model.trainable = False  # set True if you want to fine-tune
    # model = models.Sequential([
    #     base_model,
    #     layers.GlobalAveragePooling2D(),
    #     layers.Dense(num_classes, activation='softmax')  # or 20 for coarse classes
    # ])
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_cifar100_file():
    with open('data_handson_aml.pkl', 'rb') as file:
        [(x_train, y_train), (x_test, y_test), min_, max_, classes] = pickle.load(file)
    return (x_train, y_train), (x_test, y_test), min_, max_, classes


def plot_adversarial_comparison(classifier, x, x_adv, classes=None, n=5, amplification=10):
    """
    Show example and adversarial examples in a grid, with the predicted class, as well as the added noise.

    :param classifier: the classifier for prediction
    :param x: the set of images
    :param x_adv: 
    :param classes (None): the name of the classes for display purposes
    :param n (5): the number of examples to display from x
    :param amplification: factor to make noise visible
    """
    n = min(n, len(x))
    cols = 3
    rows = n
    im_size=3
    fig, axs = plt.subplots(rows, cols, figsize=(im_size*cols,im_size*rows))
    
    preds_clean = np.argmax(classifier.predict(x), axis=1)
    preds_adv = np.argmax(classifier.predict(x_adv), axis=1)
    
    indices_seen = []
    for r in range(rows):        
        rd_int = rd.randint(0, len(x)-1)
        while rd_int in indices_seen:
            rd_int = rd.randint(0, len(x)-1)
        indices_seen.append(rd_int)
        
        y_orig = preds_clean[rd_int]
        y_adv = preds_adv[rd_int]
        if classes is not None:
            y_orig = classes[y_orig]
            y_adv = classes[y_adv]
        
        # Compute perturbation
        delta = x_adv[rd_int] - x[rd_int]
        # Amplify perturbation for visualization
        delta_vis = delta * amplification + 0.5
        delta_vis = np.clip(delta_vis, 0, 1)
        
        if rows == 1:
            axs[0].imshow(x[rd_int])
            axs[0].set_title(f"Original\n{y_orig}")
            axs[0].axis('off')

            axs[1].imshow(delta_vis)
            axs[1].set_title("Perturbation (amplified)")
            axs[1].axis('off')

            axs[2].imshow(x_adv[rd_int])
            axs[2].set_title(f"Adversarial\n{y_adv}")
            axs[2].axis('off')
        else:
            axs[r,0].imshow(x[rd_int])
            axs[r,0].set_title(f"Original\n{y_orig}")
            axs[r,0].axis('off')

            axs[r,1].imshow(delta_vis)
            axs[r,1].set_title("Perturbation (amplified)")
            axs[r,1].axis('off')

            axs[r,2].imshow(x_adv[rd_int])
            axs[r,2].set_title(f"Adversarial\n{y_adv}")
            axs[r,2].axis('off')

        plt.tight_layout()
    plt.show()
    
#######################################################################################################################
##  DEPRECATED
#######################################################################################################################
def preprocess(
    x: np.ndarray,
    y: np.ndarray,
    nb_classes: int = 10,
    clip_values = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scales `x` to [0, 1] and converts `y` to class categorical confidences.

    :param x: Data instances.
    :param y: Labels.
    :param nb_classes: Number of classes in dataset.
    :param clip_values: Original data range allowed value for features, either one respective scalar or one value per
           feature.
    :return: Rescaled values of `x`, `y`.
    """
    if clip_values is None:
        min_, max_ = np.amin(x), np.amax(x)
    else:
        min_, max_ = clip_values

    normalized_x = (x - min_) / (max_ - min_)
    categorical_y = to_categorical(y, nb_classes)

    return normalized_x, categorical_y

def load_cifar100(
    raw: bool = False,
    fine_or_coarse=0
):
    """
    Loads CIFAR10 dataset from config.CIFAR10_PATH or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :param fine_or_coarse: 0 if we are taking the coarse labels (20 classes), 1 for the fine labels (100 classes).
    :return: `(x_train, y_train), (x_test, y_test), min, max`
    """

    def load_batch(fpath: str, fine_or_coarse=1) -> tuple[np.ndarray, np.ndarray]:
        """
        Utility function for loading CIFAR batches, as written in Keras.

        :param fpath: Full path to the batch file.
        :return: `(data, labels)`
        """
        with open(fpath, "rb") as file_:
            if sys.version_info < (3,):
                content = six.moves.cPickle.load(file_)
            else:
                content = six.moves.cPickle.load(file_, encoding="bytes")
                content_decoded = {}
                for key, value in content.items():
                    content_decoded[key.decode("utf8")] = value
                content = content_decoded
        data = content["data"]
        if not fine_or_coarse:
            labels = content["fine_labels"]
        else:
            labels = content["coarse_labels"]

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels
    
    def load_meta(fpath: str, fine_or_coarse=1) -> tuple[np.ndarray, np.ndarray]:
        """
        Utility function for loading CIFAR meta information (labels), as written in Keras.

        :param fpath: Full path to the batch file.
        :return: `(data, labels)`
        """
        with open(fpath, "rb") as file_:
            if sys.version_info < (3,):
                content = six.moves.cPickle.load(file_)
            else:
                content = six.moves.cPickle.load(file_, encoding="bytes")
                content_decoded = {}
                for key, value in content.items():
                    content_decoded[key.decode("utf8")] = value
                content = content_decoded
        if not fine_or_coarse:
            classes = content["fine_label_names"]
        else:
            classes = content["coarse_label_names"]
        return classes

    path = get_file(
        "cifar-100-python",
        extract=True,
        path=config.ART_DATA_PATH,
        url="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    )

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype=np.uint8)
    y_train = np.zeros((num_train_samples,), dtype=np.uint8)

    fpath = os.path.join(path, "train")
    data, labels = load_batch(fpath)
    x_train = data
    y_train = labels

    fpath = os.path.join(path, "test")
    x_test, y_test = load_batch(fpath)

    # Set channels last
    x_train = x_train.transpose((0, 2, 3, 1))
    x_test = x_test.transpose((0, 2, 3, 1))
        
    fpath = os.path.join(path, "meta")
    classes = load_meta(fpath, fine_or_coarse=fine_or_coarse)

    min_, max_ = 0.0, 255.0
    if not raw:
        min_, max_ = 0.0, 1.0
        x_train, y_train = preprocess(x_train, y_train, clip_values=(0, 255), nb_classes=len(classes))
        x_test, y_test = preprocess(x_test, y_test, clip_values=(0, 255), nb_classes=len(classes))

    return (x_train, y_train), (x_test, y_test), min_, max_, classes