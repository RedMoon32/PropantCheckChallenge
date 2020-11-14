import pandas as pd
import numpy as np
import cv2
import os
import preprocess
import hough
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import AdaBoostRegressor
import joblib

PREPROCESSED_PATH = 'preprocessed_imgs'

def read_im(index:int) -> np.array:
    """
    Read image with given index
    Args:
        index:int
    Returns:
        np.array - image read
    """
    return cv2.imread(os.path.join(PREPROCESSED_PATH, str(int(index))+'.jpg'))

def get_stats(im:np.array)->(np.array, np.array):
    """
    Returns stats and areas of connected components of the image
    Args:
        im: np.array
    Returns:
        stats: np.array - stats of every component
        label_area:np.array - area of every component
    """
    img = cv2.resize(im, (hough.AVG_W, hough.AVG_H))
    img = preprocess.cut_image(img, 30)

    mask = cv2.threshold(img[:, :, 0], 255, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    stats = cv2.connectedComponentsWithStats(mask, 4)[2]
    label_area = stats[1:, cv2.CC_STAT_AREA]
    return stats, label_area

def get_count(im:np.array, l:int = 70, r:int = 600, res:bool=True, ret_stats:bool=False):
    """
    Returns approximate number of granules, by counting mean area of component 
    that lies in [l, r]
    if res is true then im is already areas of components of image, otherwise
    we compute it beforehand
    Args:
        im: np.array - image or label_area depending on res
        l:int - left boundary for computing mean
        r:int - right boundary
        res:bool
        ret_stats: bool - do we want to return stats
    """
    if res:
        stats, label_area = get_stats(im)
    else:
        label_area = im.copy()
    min_area, max_area = l, r
    singular_mask = (min_area < label_area) & (label_area <= max_area)
    circle_area = np.mean(label_area[singular_mask])
    found = int(np.sum(np.round(label_area / circle_area)))
    if ret_stats:
        return found, stats
    return found

def precompute_stats(df:pd.DataFrame) -> (np.array, np.array, np.array):
    """
    Precompute stats and areas of every image beforehand, to 
    save computational time
    Args:
        df:pd.DataFrame
        ret_stats
    Returns:
        images:np.array - areas of images, 
        labels:np.array - prop_count of images
        imageids:np.array - id of each image
    """
    images = []
    labels = df.prop_count.to_list()
    imageids = df.ImageId.to_list()
    stats = []
    for im in df.ImageId.to_list():
        curim = read_im(im)
        stat, labarea = get_stats(curim)
        stats.append(stat.copy())
        images.append(labarea)
    
    return images, labels, imageids
    
def brute_force_bounds(images:np.array, labels:np.array, imageids:np.array) -> (np.array, np.array, np.array):
    """
    Find for each image best suitable l,r boundaries
    then we will train model for predicting them on unknown images.
    Args:
        images:np.array - areas of components
        lables:np.array - prop count of images
        imageids:np.array - ids of images in dataframe
    Returns:
        bounds:np.array - best boundaries
        train_x:np.array - data for training model where features are distribution of areas of components 
        test_x:np.array - labels of training data
    """
    train_y = np.array([])
    train_x = np.array([])
    bounds = [(1,1)] * len(images)
    bests = [1] * len(images)
    
    for id_, (im, lab, imid) in enumerate(zip(images, labels, imageids)):
        min_error = 100000
        best_l = -1
        best_r = -1
        #try all variants of (l,r) for image
        for l in range(10, 40, 10):
            l = 30
            for r in range(l+10, 1000, 10):
                try:
                    found = get_count(im,l=l, r=r, res=False)
                except ValueError:
                    found = 0
                true = lab
                error = abs(found-true)/true
                if min_error > error:
                    min_error = error
                    best_l = l
                    best_r = r

        #save best
        bounds[id_] = (best_l, best_r)
        bests[id_] = min_error
        train_y = np.append(train_y, best_l)
        train_y = np.append(train_y, best_r)
    
        found, stats = get_count(read_im(imid, PREPROCESSED_PATH), l=0, r=1000, res=True, ret_stats=True)

        #find distribution
        p = pd.DataFrame(stats, columns=['CC_STAT_LEFT', 'CC_STAT_TOP', 'CC_STAT_WIDTH', 'CC_STAT_HEIGHT', 'CC_STAT_AREA'])
        out = pd.cut(p[p.CC_STAT_AREA < 1000].CC_STAT_AREA, bins=np.arange(0, 1000, 10), include_lowest=True)
        #add to training data
        train_x = np.append(train_x, out.value_counts(normalize=True, sort=False).to_list())
    
    bounds = np.array(bounds)
    cnt = round(len(train_y)/2)
    train_y = train_y.reshape(cnt, 2)
    train_x = train_x.reshape(cnt, 99)

    return bounds, train_x, train_y

def get_model()->AdaBoostRegressor:
    """
    Full pipeline for getting trained AdaBoostRegressor model
    Returns:
        clf:AdaBoostRegressor
    """
    #Read dataframe and drop excess columns and bad images
    df_augmented = pd.read_csv('labels_hand_marked.csv').drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'], axis=1)
    df_augmented = df_augmented[~df_augmented.prop_count.isna()]
    df_augmented = df_augmented[~df_augmented.ImageId.isin([104, 908, 906, 907, 905, 904]+list(range(905, 1000)))]
    #precompute stats and brute force (l,r) boundaries
    images_aug_, labels_aug_, imageids_aug_ = precompute_stats(df_augmented)
    imageids_aug_ = df_augmented.ImageId.to_numpy()
    labels_aug_ = np.array(labels_aug_)
    bounds_aug_, train_x_aug_, train_y_aug_ = brute_force_bounds(images_aug_, labels_aug_, imageids_aug_)
    #images on which we tested model
    test_ids = [776, 675, 42, 3, 714, 312, 127, 653, 592, 205, 179, 191]
    test_indices = np.in1d(df_augmented.ImageId.to_numpy(), test_ids).nonzero()[0]
    #delete test images from train
    deleted_test_x, deleted_test_y = np.delete(train_x_aug_, test_indices, axis=0), np.delete(train_y_aug_, test_indices, axis=0)

    x_train_aug = deleted_test_x[:]
    y_train_aug = deleted_test_y[:]
    #train
    clf = MultiOutputRegressor(AdaBoostRegressor(random_state=10, n_estimators=5)).fit(x_train_aug, y_train_aug)
    return clf

if __name__ == '__main__':
    clf = get_model()
    joblib.dump(clf, 'supposed_best.pkl')