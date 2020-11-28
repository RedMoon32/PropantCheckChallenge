import pandas as pd
import numpy as np
import cv2
import os
import random
import preprocess
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from preprocess import full_pipeline
from hough import draw_hough, AVG_H, AVG_W
from RPCC_metric_utils_for_participants_V2 import *

LABELS_PATH = "./data/labels/train.csv"
PIXEL_TO_MM_RATIO = 0.1
data_df = None



def get_train_radiuses(cur_im: int) -> np.array:
    """
    Reads given image, and computes cv2.houghcircles radius distribution for it

    Args:
        cur_im: int - imaged id
    Returns:
        np.array: - array of shape (29, ) - count of each radius
    """
    im = preprocess.read_im(cur_im)
    proc = cv2.resize(im, (AVG_W, AVG_H))
    avg_r, circles = draw_hough(proc)
    radiuses = circles[:, 2]
    distros = [
        np.where(radiuses == i)[0].shape[0] for i in range(1, 30)
    ]  # get each radius count
    return distros


def get_radiuses(data_df: pd.DataFrame) -> list:
    """
    Reads preprocessed images and retreives HoughCircles radiuses
    for each image
    Args:
        data_df:pd.DataFrame
    Returns:
        rows:list - list with radiuses for each image
    """
    rows = []
    for ind, (frac, im) in enumerate(
        list(zip(data_df.fraction.to_list(), data_df.ImageId.to_list()))
    ):
        distros = get_train_radiuses(im)
        if frac is not np.nan:
            rows.append([im, frac] + distros)
    return rows


def y_radius(y: np.array) -> (int, np.array):
    """
    Return mean bin diameter for matrix of distributions of beans

    Example : 6,7,8,10,12,14,16,  18,   20, 25,30,35,40,45,50,60,70,80,100
    y :      [[0,0,0,0,0,0,0,0,0, 0.5, 0.6, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
              [0,0,0,0,0,0,0,0,0, 0.1, 0.2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] - 2 rows
    mean is
            mean_y = [0 0 0 .............. 0.3  0.4 ...............................0]
    multiply by diam
            6,7,8,10,12,14,16,                    18,   20,      25,30,35,40,45,50,60,70,80,100,0
            [3.35,2.8, 2.36, 2, 1.7, 1.4, 1.18,   1, 0.85,     0.71, 0.6, 0.5, 0.425, 0.355, 0.3, 0.25, 0.212, 0.18, 0.15,0]:

    0.3 * 1 (18th bin) + 0.4 * 0.85 (20th bin) = 0.64 mm - mean suitable diameter of the bin

    Args:
        y: np.array - target matrix of bin distributions of shape (x, 20),
    Returns:
        int: - mean suitable diamater
        np.array: - mean distribution bin array (like mean_y)
    """
    mean_y = y.mean(axis=0)
    return np.inner(mean_y, sive_diam_pan), mean_y


def x_radius(x: np.array) -> (int, np.array):
    """
    Return mean radius for matrix of distributions of radiuses

    Example :   1,2,3,4  5   6   7     8  ,9, 10 .....
              [[0.....  0.4 0.2  0.5  0  .......]
               [0.....  0.2  0   0.8   0  .......]
    mean is
    mean_x  =  [0.....  0.3  0.2 0.65   0  .......]

    multiply by size of radiuses
               [1 2 3 4 5 6 7 8 9]

    5*0.3 + 6*0.2 + 7*0.65 = 7.25 is a mean radius over the whole matrix of radiuses
    distributions

    Args:
        x: np.array - matrix of normalized radiuses distribution of train image
        of shape (x, 29),
        [i, j] cell shows which part of all hough circles on image #i
        take circles with radius of j pixels
    Returns:
        int: - mean radius in pixels
        np.array: - mean radius bin array (mean_x of shape (29, ))
    """
    mean_x = x.mean(axis=0)
    return np.inner(mean_x, list(range(1, 30))), mean_x


def augment_data(
    data_x: np.array, data_y: np.array, shifts: int
) -> (np.array, np.array):
    """
    Helper method to overcome unknown fractions and new distributions of radiuses and bins.

    Data augmentation - shifting all radiuses in data_x array on 'shifts' points left
    and calculating new bin distributions :
        shift matrix of bin distributions until mean diameter of the bin is the same
        as diameter of the shifted radiuses

    Args:
        data_x: np.array - matrix of normalized radiuses distribution of train image
        of shape (n, 29),
        data_y: np.array - target matrix of bin distributions of shape (n, 20),
    Returns:
        augmented_x: np.array - shifted radiuses
        best_y: np.array - cacluated best target distributions of bins
    """

    augmented_x = np.roll(data_x.copy(), shifts)
    avg_r = x_radius(augmented_x)[0] * PIXEL_TO_MM_RATIO
    augmented_y = data_y.copy()
    best_y = augmented_y
    shift = -1 if shifts > 0 else 1
    while y_radius(augmented_y)[1][0] == 0 and y_radius(augmented_y)[1][-1] == 0:
        augmented_y = np.roll(augmented_y, shift)
        avg_r_y = y_radius(augmented_y)[0]
        diff = avg_r - avg_r_y
        if abs(diff) < abs(avg_r - y_radius(best_y)[0]):
            best_y = augmented_y.copy()
    return augmented_x, best_y


def augment_class(
    class_name: str,
    labels_df: pd.DataFrame,
    radius_df: pd.DataFrame,
    augment: bool = True,
) -> (np.array, np.array):
    """
    Method find all rows for the given class (fraction) and augments them
    Args:
        class_name:str - which fraction to augment
        labels_df:pd.DataFrame
        radius_df:pd.DataFrame
        augment:bool
    Returns:
        train_x:np.array
        train_y:np.array
    """
    train_class = radius_df[radius_df.fraction == class_name]
    init_x, init_y = get_data_labels(train_class, labels_df)
    train_x = init_x.copy()
    train_y = init_y.copy()
    if augment:
        for shift in range(-5, 5, 2):
            aug_x, aug_y = augment_data(init_x, init_y, shift)
            train_x = np.concatenate((train_x, aug_x))
            train_y = np.concatenate((train_y, aug_y))
    return train_x, train_y


def get_data_labels(
    radius_df: pd.DataFrame, labels_df: pd.DataFrame, normalize: bool = True
) -> (pd.DataFrame, pd.DataFrame):
    """
    Drop redundant columns and normalize target distributions by their sum

    Args:
        radius_df: pd.DataFrame - table of radiuses distributions of train image
        of shape (n, 29), [i,j] cell shows how many circles with radius j found on image #i
        labels_df: pd.DataFrame - table of target bin distributions of shape (n, 20)
        normalize: bool - flag whether to normalize each radius by number of their occurences on the image
    Returns:
        labels: pd.DataFrame - clean train.csv table without redundant columns
        data:   pd.DataFrame - normalized radiuses table for images
    """

    labels = (
        labels_df[labels_df.ImageId.isin(radius_df.ImageId)]
        .drop(["ImageId", "fraction"], axis=1)
        .to_numpy()
    )
    labels = np.array([row / row.sum() for row in labels])
    data = radius_df.drop(["ImageId", "fraction"], axis=1).to_numpy()
    if normalize:
        data = np.array([row / row.sum() for row in data])
    return data, labels


def read_data() -> (pd.DataFrame, pd.DataFrame):
    """
    Reads preprocessed images and retreives HoughCircles radiuses
    for each image, shuffles everything and assembles new pandas dataframe from the array of radiuses
    Returns:
        data_df:pd.DataFrame
        radius_df:pd.DataFrame
    """
    data_df = pd.read_csv(LABELS_PATH)
    rows = get_radiuses(data_df)
    radius_df = pd.DataFrame(rows, columns=["ImageId", "fraction"] + list(range(1, 30)))
    radius_df = radius_df.sample(frac=1).reset_index(drop=True)

    data_df = data_df.drop(
        [
            "prop_count",
            "pan",
        ],
        axis=1,
    )
    data_df = data_df.dropna(subset=["fraction"])
    data_df = data_df.set_index("ImageId")
    data_df = data_df.reindex(index=radius_df["ImageId"])
    data_df = data_df.reset_index()
    data_df["0"] = 0
    return data_df, radius_df


def train_test_split_radius(
    x_df: pd.DataFrame, y_df: pd.DataFrame
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Train test split of the radiuses (x_df) and target bin distributions (y_df)
    Args:
        x_df:pd.DataFrame
        y_df:pd.DataFrame
    Returns:
        train_df: pd.DataFrame
        train_labels_df: pd.DataFrame
        test_df: pd.DataFrame
        test_labels_df: pd.DataFrame
    """
    train_df, train_labels_df, test_df, test_labels_df = train_test_split(x_df, y_df, test_size = 0.2, stratify = y_df.fraction.to_list())
    return train_df, train_labels_df, test_df, test_labels_df


def get_training_data(
    labels_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> (np.array, np.array, np.array, np.array):
    """
    Function augments samples of each fraction except 2040pdcd and returns train and test data
    Args:
        labels_df:pd.DataFrame
        train_df:pd.DataFrame
        test_df:pd.DataFrame
    Returns:
        train_x:np.array
        train_y:np.array
        test_x:np.array
        test_y:np.array
    """
    train_1620, labels_1620 = augment_class(
        "16/20", augment=True, labels_df=labels_df, radius_df=train_df
    )
    train_2040, labels_2040 = augment_class(
        "20/40", augment=True, labels_df=labels_df, radius_df=train_df
    )
    train_20phd, labels_20phd = augment_class(
        "20/40_pdcpd_bash_lab", augment=False, labels_df=labels_df, radius_df=train_df
    )

    train_x = np.concatenate(
        (
            train_1620,
            train_20phd,
            train_2040,
        )
    )
    train_y = np.concatenate(
        (
            labels_1620,
            labels_20phd,
            labels_2040,
        )
    )
    test_x, test_y = get_data_labels(test_df, labels_df)

    return train_x, train_y, test_x, test_y


def train_model(train_x: np.array, train_y: np.array) -> DecisionTreeRegressor:
    """
    Function trains RandomForestRegressor on train x and train y
    Args:
        train_x:np.array - training data
        train_y:np.array - training labels
    Returns:
        regr: RandomForestRegressor - trained model
    """
    regr = DecisionTreeRegressor(max_depth=15, random_state=0)
    regr = regr.fit(train_x, train_y)
    return regr


def get_trained_model() -> DecisionTreeRegressor:
    """
    Function runs all the pipeline and returns the trained model
    Returns:
        regr:MultiOutputRegressor - model
    """
    global data_df

    random.seed(41)
    np.random.seed(41)

    data_df, radius_df = read_data()

    print("Training model.....")
    train_df, train_labels_df, test_df, test_labels_df = train_test_split(
        radius_df, data_df
    )
    train_x, train_y, test_x, test_y = get_training_data(data_df, train_df, test_df)
    regr = train_model(train_x, train_y)
    return regr
