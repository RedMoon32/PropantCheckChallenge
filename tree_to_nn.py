from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
import random
from sklearn.tree import _tree
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader

paths = []
features = []
vals = []

features_l, tresholds_l, features_r, tresholds_r, less_than, greater_than = [], [], [], [], [], []


def tree_to_code(tree, feature_names, argmax_class):
    """ 
    Iterate the tree and return array of sequence of binary rules
    which leed to some leaf node (output class)
    """
    global paths, features, vals
    paths = []
    features = []
    vals = []

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    def recurse(node, depth, path):
        global features
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            f1 = [int(name), 0, threshold]
            recurse(tree_.children_left[node], depth + 1, path+[f1])

            f2 = [int(name), 1, threshold]
            recurse(tree_.children_right[node], depth + 1, path+[f2])

            features += [f1, f2]
        else:
            global vals
            curclass = tree_.value[node]
            curclass = np.argmax(curclass) if argmax_class else curclass
            paths.append(path)
            vals.append(curclass)

    recurse(0, 1, [])
    return paths, features, vals

def get_tresholds_and_bounds(features):
    """
    Return all less features
    which are compared with some tresholds

    Example:
    f1 < 5 & f3 < 10 & f2 >= -3 : return 
    
    [f1, f3] and [5, 10]
    [f2] and [-3]
    """
    global features_l, tresholds_l, features_r, tresholds_r, less_than, greater_than

    np_feats = np.array(features)

    less_than = np.where(np_feats[:, 1] == 0)[0]
    greater_than = np.where(np_feats[:, 1] == 1)[0]

    features_l = (np_feats[less_than][:, 0]).astype(int)
    tresholds_l = np_feats[less_than][:, 2]

    features_r = (np_feats[greater_than][:, 0]).astype(int)
    tresholds_r = np_feats[greater_than][:, 2]
    return features_l, tresholds_l, features_r, tresholds_r

def get_feature_vector(x_array, features):
    """
    Compute all separated predicate rules given the x array
    """
    res = np.zeros((x_array.shape[0], len(features)))
    res[:, less_than] = x_array[:, features_l] <= tresholds_l
    res[:, greater_than] = x_array[:, features_r] > tresholds_r
    return res.astype(int)

def create_torch_nn(clf, columns, argmax_class=True):
    """
    Given clf - some DecisionTreeClassifier or DecisionTreeRegressor
    return neural network, input features rules and output classes indices
    """
    paths, features, vals = tree_to_code(clf, list(map(str, columns)), argmax_class)
    get_tresholds_and_bounds(features)

    nin = len(features)
    nout = len(paths)

    fpaths = [[features.index(node) for node in path] for path in paths]
    weights = np.zeros((nin, nout))
    for ind, innernodes in enumerate(fpaths):
        for feat in innernodes:
            weights[feat, ind] = 1

    path_bias = -np.array(list(map(len, fpaths)))
    model = nn.Sequential(nn.Linear(nin, nout))
    with torch.no_grad():
        model[0].weight = torch.nn.Parameter(torch.Tensor(weights.T))
        model[0].bias = torch.nn.Parameter(torch.Tensor(path_bias))

    return model, features, np.array(vals)

def predict_torch(model, data, features=features, classes=vals):
    """
    Helper method which given the feature rules, creates
    input vector for neural network consisting of compute binary rules (tree node comparisons)
    outputs predicted values
    """
    test_tensor = torch.Tensor(get_feature_vector(data, features))
    y_pred = classes[np.argmax(model(test_tensor).detach().numpy(), axis=1)]
    return y_pred.reshape(y_pred.shape[0], y_pred.shape[1])