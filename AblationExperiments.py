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


def run(pathDataset,testPath,pathtest,pathModels):

    test_X = np.load(pathDataset + 'DS/X_test.npy')
    test_Y = np.load(pathDataset + 'DS/Y_test.npy')

    # NN #
    nn = load_model(pathModels + 'NN.h5')

    print('Prediction NN')
    nn.summary()
    predictionsL = nn.predict(test_X)
    y_pred = np.argmax(predictionsL, axis=1)
    cmB = confusion_matrix(test_Y, y_pred)

    # CNN
    test_X_single = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
    print('Prediction CNN')
    cnn = load_model(pathModels + 'CNN.h5')
    cnn.summary()

    predictions = cnn.predict(test_X_single)
    y_pred = np.argmax(predictions, axis=1)
    cmC = confusion_matrix(test_Y, y_pred)

    # ANN #
    test_X_concatenate = np.load(pathDataset + 'DS/X_test_conc.npy')
    ann = load_model(pathModels + 'ANN.h5')
    print('Prediction ANN')
    ann.summary()
    predictions = ann.predict(test_X_concatenate)
    y_pred = np.argmax(predictions, axis=1)
    cmA = confusion_matrix(test_Y, y_pred)

    # ACNN #

    test_X_conACNN = test_X_concatenate.reshape(test_X_concatenate.shape[0], test_X_concatenate.shape[1], 1)
    acnn = load_model(pathModels + 'ACNN.h5')
    print('Prediction ACNN')
    acnn.summary()
    predictions = acnn.predict(test_X_conACNN)
    y_pred = np.argmax(predictions, axis=1)
    cmACNN = confusion_matrix(test_Y, y_pred)

    # MINDFUL #

    test_X_image = np.load(pathDataset + 'DS/X_test_image.npy')
    mindful = load_model(pathModels + 'MINDFUL.h5')
    print('Prediction MINDFUL')
    mindful.summary()
    predictions = mindful.predict(test_X_image)
    y_pred = np.argmax(predictions, axis=1)
    cmM = confusion_matrix(test_Y, y_pred)

    print('Prediction NN:')
    print(cmB)
    print('Prediction ANN:')
    print(cmA)
    print('Prediction CNN:')
    print(cmC)
    print('Prediction ACNN:')
    print(cmACNN)
    print('Prediction MINDFUL:')
    print(cmM)

    # create pandas for results
    col = ['Model', 'TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
    columnsTemp = ['TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
    results = pd.DataFrame(columns=col)

    r = getResult(cmB, 2)
    r.insert(0, 'NN')
    dfResults = pd.DataFrame([r], columns=col)
    results = results.append(dfResults, ignore_index=True)
    r = getResult(cmA, 2)
    r.insert(0, 'ANN')
    dfResults = pd.DataFrame([r], columns=col)
    results = results.append(dfResults, ignore_index=True)
    r = getResult(cmC, 2)
    r.insert(0, 'CNN')
    dfResults = pd.DataFrame([r], columns=col)
    results = results.append(dfResults, ignore_index=True)
    r = getResult(cmACNN, 2)
    r.insert(0, 'ACNN')
    dfResults = pd.DataFrame([r], columns=col)
    results = results.append(dfResults, ignore_index=True)
    r = getResult(cmM, 2)
    r.insert(0, 'MINDFUL')
    dfResults = pd.DataFrame([r], columns=col)
    results = results.append(dfResults, ignore_index=True)
    print(results)
    results.to_csv(testPath + '_ablation_results.csv', index=False)

def runCICIDS(pathDataset,testPath,pathtest,pathModels):
    # create pandas for results
    col = ['Model', 'TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
    columnsTemp = ['TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
    results = pd.DataFrame(columns=col)


    test_Y = list()
    test_X = list()
    for i in range(1,9):
        x = np.load(pathDataset + 'DS/X_test'+str(i)+'.npy')
        y = np.load(pathDataset + 'DS/Y_test' + str(i) + '.npy')
        test_X.append(x)
        test_Y.append(y)




    # NN #
    nn = load_model(pathModels + 'NN.h5')

    print(' NN prediction')
    nn.summary()
    i = 0
    r_list=list()
    for t, Y in zip(test_X, test_Y):
        i += 1
        predictionsC = nn.predict(t)
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

    r_mean.insert(0, "NN")
    dfResults = pd.DataFrame([r_mean], columns=col)
    results = results.append(dfResults, ignore_index=True)
    print(results)




    # CNN
    test_X_single=list()
    for t in test_X:
        t_single = t.reshape(t.shape[0], t.shape[1], 1)
        test_X_single.append(t_single)
    cnn = load_model(pathModels + 'CNN.h5')
    print('CNN prediction')
    cnn.summary()
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

    r_mean.insert(0, "CNN")
    dfResults = pd.DataFrame([r_mean], columns=col)
    results = results.append(dfResults, ignore_index=True)
    print(results)

    # ANN #
    test_X_concatenate= list()
    for i in range(1,9):
        x = np.load(pathDataset + 'DS/X_testConc'+str(i)+'.npy')
        test_X_concatenate.append(x)

    ann = load_model(pathModels + 'ANN.h5')
    print('ANN prediction')
    ann.summary()
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

    r_mean.insert(0, "ANN")
    dfResults = pd.DataFrame([r_mean], columns=col)
    results = results.append(dfResults, ignore_index=True)


    # ACNN #
    test_X_conACNN = list()
    for t in test_X_concatenate:
        t_conc= t.reshape(t.shape[0], t.shape[1], 1)
        test_X_conACNN.append(t_conc)

    acnn = load_model(pathModels + 'ACNN.h5')
    print('ACNN prediction')
    acnn.summary()
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

    r_mean.insert(0, "ACNN")
    dfResults = pd.DataFrame([r_mean], columns=col)
    results = results.append(dfResults, ignore_index=True)


    # MINDFUL #

    test_X_image = list()
    for i in range(1, 9):
        x = np.load(pathDataset + 'DS/X_testImage' + str(i) + '.npy')
        test_X_image.append(x)

    mindful = load_model(pathModels + 'MINDFUL.h5')
    print('MINDFUL prediction')
    mindful.summary()
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

    r_mean.insert(0, "MINDFUL")
    dfResults = pd.DataFrame([r_mean], columns=col)
    results = results.append(dfResults, ignore_index=True)
    print(results)


    results.to_csv(testPath + '_ablation_results.csv', index=False)


def main():
    pd.set_option('display.expand_frame_repr', False)
    dataset = sys.argv[1]
    config = configparser.ConfigParser()
    config.read('MINDFUL.conf')
    dsConf = config[dataset]
    pathDataset = dsConf.get('pathDatasetNumeric')
    testPath = dsConf.get('testPath')
    if (testPath == 'CICIDS2017'):
        pathtest = dsConf.get('pathTest').split(',')
    else:
        pathtest = dsConf.get('pathTest')

    pathModels = dsConf.get('pathModels')


    if (dataset == 'CICIDS2017'):
       runCICIDS(pathDataset,testPath,pathtest,pathModels)

    else:
        print('Using %s dataset:' %dataset)
        run(pathDataset,testPath,pathtest,pathModels)








if __name__ == "__main__":
    main()