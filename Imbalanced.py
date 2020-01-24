import numpy as np

my_seed = 12
np.random.seed(my_seed)
import random

random.seed(my_seed)
from sklearn.preprocessing import scale, MinMaxScaler

import tensorflow
import configparser

tensorflow.set_random_seed(my_seed)

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import confusion_matrix
from keras.models import Model

from keras.models import load_model

np.set_printoptions(suppress=True)
import sys


def getResult(cm, N_CLASSES):
    tp = cm[0][0]  # attacks true
    fn = cm[0][1]  # attacs predict normal
    fp = cm[1][0]  # normal predict attacks
    tn = cm[1][1]  # normal as normal
    attacks = tp + fn
    normals = fp + tn
    OA = (tp + tn) / (attacks + normals)
    AA = ((tp / attacks) + (tn / normals)) / N_CLASSES
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * ((P * R) / (P + R))
    FAR = fp / (fp + tn)
    TPR = tp / (tp + fn)
    r = [tp, fn, fp, tn, OA, AA, P, R, F1, FAR, TPR]
    return r


def run(num, pathDataset, pathModels):
    print('Execution imbalanced % s' % num)
    col = ['Model', 'TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
    columnsTemp = ['TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
    results = pd.DataFrame(columns=col)

    # NN #
    test_Y = list()
    test_X = list()
    for i in range(1, 9):
        x = np.load(pathDataset + str(num) + '/X_test' + str(i) + '.npy')
        y = np.load(pathDataset + str(num) + '/Y_test' + str(i) + '.npy')
        test_X.append(x)
        test_Y.append(y)
    nn = load_model(pathModels + 'NN' + str(num) + '.h5')
    print(' NN prediction')
    i = 0
    r_list = list()
    for t, Y in zip(test_X, test_Y):
        i += 1
        predictionsC = nn.predict(t)
        y_pred = np.argmax(predictionsC, axis=1)
        cm75 = confusion_matrix(Y, y_pred)
        r = getResult(cm75, 2)
        r_list.append(tuple(r))

    dfResults_temp = pd.DataFrame(r_list, columns=columnsTemp)
    drMean = dfResults_temp.mean(axis=0)
    drmeanList = pd.Series(drMean).values
    r_mean = []
    for i in drmeanList:
        r_mean.append(i)

    r_mean.insert(0, "NN-" + str(num))
    dfResults = pd.DataFrame([r_mean], columns=col)
    results = results.append(dfResults, ignore_index=True)

    # CNN
    test_X_single = list()
    for t in test_X:
        t_single = t.reshape(t.shape[0], t.shape[1], 1)
        test_X_single.append(t_single)
    cnn = load_model(pathModels + 'CNN' + str(num) + '.h5')
    print('CNN prediction')

    i = 0
    r_list = list()
    for t, Y in zip(test_X_single, test_Y):
        i += 1
        predictionsC = cnn.predict(t)
        y_pred = np.argmax(predictionsC, axis=1)
        cm = confusion_matrix(Y, y_pred)
        r = getResult(cm, 2)
        r_list.append(tuple(r))

    dfResults_temp = pd.DataFrame(r_list, columns=columnsTemp)
    drMean = dfResults_temp.mean(axis=0)
    drmeanList = pd.Series(drMean).values
    r_mean = []
    for i in drmeanList:
        r_mean.append(i)

    r_mean.insert(0, "CNN-" + str(num))
    dfResults = pd.DataFrame([r_mean], columns=col)
    results = results.append(dfResults, ignore_index=True)

    # ANN #
    test_X_concatenate = list()
    for i in range(1, 9):
        x = np.load(pathDataset + str(num) + '/X_testConc' + str(i) + '.npy')
        test_X_concatenate.append(x)

    ann = load_model(pathModels + 'ANN' + str(num) + '.h5')
    print('ANN prediction')
    i = 0
    r_list = list()
    for t, Y in zip(test_X_concatenate, test_Y):
        i += 1
        predictionsC = ann.predict(t)
        y_pred = np.argmax(predictionsC, axis=1)
        cm = confusion_matrix(Y, y_pred)
        r = getResult(cm, 2)
        r_list.append(tuple(r))

    dfResults_temp = pd.DataFrame(r_list, columns=columnsTemp)
    drMean = dfResults_temp.mean(axis=0)
    drmeanList = pd.Series(drMean).values
    r_mean = []
    for i in drmeanList:
        r_mean.append(i)

    r_mean.insert(0, "ANN-" + str(num))
    dfResults = pd.DataFrame([r_mean], columns=col)
    results = results.append(dfResults, ignore_index=True)

    # ACNN #
    test_X_conACNN = list()
    for t in test_X_concatenate:
        t_conc = t.reshape(t.shape[0], t.shape[1], 1)
        test_X_conACNN.append(t_conc)

    acnn = load_model(pathModels + 'ACNN' + str(num) + '.h5')
    print('ACNN prediction')
    i = 0
    r_list = list()
    for t, Y in zip(test_X_conACNN, test_Y):
        i += 1
        predictionsC = acnn.predict(t)
        y_pred = np.argmax(predictionsC, axis=1)
        cm = confusion_matrix(Y, y_pred)
        r = getResult(cm, 2)
        r_list.append(tuple(r))

    dfResults_temp = pd.DataFrame(r_list, columns=columnsTemp)
    drMean = dfResults_temp.mean(axis=0)
    drmeanList = pd.Series(drMean).values
    r_mean = []
    for i in drmeanList:
        r_mean.append(i)

    r_mean.insert(0, "ACNN-" + str(num))
    dfResults = pd.DataFrame([r_mean], columns=col)
    results = results.append(dfResults, ignore_index=True)

    # MINDFUL #

    test_X_image = list()
    for i in range(1, 9):
        x = np.load(pathDataset + str(num) + '/X_testImage' + str(i) + '.npy')
        test_X_image.append(x)

    mindful = load_model(pathModels + 'MINDFUL' + str(num) + '.h5')
    print('MINDFUL prediction')
    i = 0
    r_list = list()
    for t, Y in zip(test_X_image, test_Y):
        i += 1
        predictionsC = mindful.predict(t)
        y_pred = np.argmax(predictionsC, axis=1)
        cm = confusion_matrix(Y, y_pred)
        r = getResult(cm, 2)
        r_list.append(tuple(r))

    dfResults_temp = pd.DataFrame(r_list, columns=columnsTemp)
    drMean = dfResults_temp.mean(axis=0)
    drmeanList = pd.Series(drMean).values
    r_mean = []
    for i in drmeanList:
        r_mean.append(i)

    r_mean.insert(0, "MINDFUL" + str(num))
    dfResults = pd.DataFrame([r_mean], columns=col)
    results = results.append(dfResults, ignore_index=True)
    print('Results for %s imbalanced' % num)
    print(results)

    return results


def main():
    pd.set_option('display.expand_frame_repr', False)

    pathDataset = 'datasets/CICIDS2017/numeric/Imbalanced/'

    pathModels = 'models/CICIDS2017/Imbalanced/'

    col = ['Model', 'TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
    results = pd.DataFrame()
    imbalanced = [100,75, 50, 25, 5]
    for n in imbalanced:
        r = run(n, pathDataset, pathModels)
        results = results.append(r, ignore_index=True)

    print(results)
    results.to_csv('Imbalanced_results.csv', index=False)


if __name__ == "__main__":
    main()