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


def main():
    pd.set_option('display.expand_frame_repr', False)


    pathDataset = 'datasets/UNSW_NB15/numeric/'


    pathModels = 'models/UNSW_NB15/Filters/'

    train_X = np.load(pathDataset + 'DS/X_train_image.npy')
    train_Y = np.load(pathDataset + 'DS/Y_train_image.npy')

    f3= load_model(pathModels + 'MINDFULF3.h5')
    print('Prediction Numer of Filters 3')
    f3.summary()
    predictions = f3.predict(train_X)
    y_pred = np.argmax(predictions, axis=1)
    cm3 = confusion_matrix(train_Y, y_pred)
    f5 = load_model(pathModels + 'MINDFULF5.h5')
    print('Prediction Numer of Filters 5')
    f5.summary()
    predictions = f5.predict(train_X)
    y_pred = np.argmax(predictions, axis=1)
    cm5 = confusion_matrix(train_Y, y_pred)
    f7 = load_model(pathModels + 'MINDFULF7.h5')
    print('Prediction Numer of Filters 7')
    f7.summary()
    predictions = f7.predict(train_X)
    y_pred = np.argmax(predictions, axis=1)
    cm7 = confusion_matrix(train_Y, y_pred)
    f9 = load_model(pathModels + 'MINDFULF9.h5')
    print('Prediction Numer of Filters 9')
    f9.summary()
    predictions = f9.predict(train_X)
    y_pred = np.argmax(predictions, axis=1)
    cm9 = confusion_matrix(train_Y, y_pred)
    f11 = load_model(pathModels + 'MINDFULF11.h5')
    print('Prediction Numer of Filters 11')
    f11.summary()
    predictions = f11.predict(train_X)
    y_pred = np.argmax(predictions, axis=1)
    cm11 = confusion_matrix(train_Y, y_pred)



    # create pandas for results
    col = ['#FIlters', 'TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
    results = pd.DataFrame(columns=col)

    r = getResult(cm3, 2)
    r.insert(0, 'Train 3')
    dfResults = pd.DataFrame([r], columns=col)
    results = results.append(dfResults, ignore_index=True)
    r = getResult(cm5, 2)
    r.insert(0,  'Train 5')
    dfResults = pd.DataFrame([r], columns=col)
    results = results.append(dfResults, ignore_index=True)
    r = getResult(cm7, 2)
    r.insert(0, 'Train 7')
    dfResults = pd.DataFrame([r], columns=col)
    results = results.append(dfResults, ignore_index=True)
    r = getResult(cm9, 2)
    r.insert(0, ' Train 9')
    dfResults = pd.DataFrame([r], columns=col)
    results = results.append(dfResults, ignore_index=True)
    r = getResult(cm11, 2)
    r.insert(0, 'Train 11')
    dfResults = pd.DataFrame([r], columns=col)
    results = results.append(dfResults, ignore_index=True)
    print(results)

    test_X = np.load(pathDataset + 'DS/X_test_image.npy')
    test_Y = np.load(pathDataset + 'DS/Y_test.npy')


    print('Prediction Numer of Filters 3 test ')
    predictions = f3.predict(test_X)
    y_pred = np.argmax(predictions, axis=1)
    cm3T = confusion_matrix(test_Y, y_pred)
    print('Prediction Numer of Filters 5 test')
    predictions = f5.predict(test_X)
    y_pred = np.argmax(predictions, axis=1)
    cm5T = confusion_matrix(test_Y, y_pred)
    print('Prediction Numer of Filters 7 test')
    predictions = f7.predict(test_X)
    y_pred = np.argmax(predictions, axis=1)
    cm7T = confusion_matrix(test_Y, y_pred)
    print('Prediction Numer of Filters 9 test')
    predictions = f9.predict(test_X)
    y_pred = np.argmax(predictions, axis=1)
    cm9T = confusion_matrix(test_Y, y_pred)
    print('Prediction Numer of Filters 11 test')
    predictions = f11.predict(test_X)
    y_pred = np.argmax(predictions, axis=1)
    cm11T = confusion_matrix(test_Y, y_pred)

    r = getResult(cm3T, 2)
    r.insert(0, 'Test 3')
    dfResults = pd.DataFrame([r], columns=col)
    results = results.append(dfResults, ignore_index=True)
    r = getResult(cm5T, 2)
    r.insert(0, 'Test 5')
    dfResults = pd.DataFrame([r], columns=col)
    results = results.append(dfResults, ignore_index=True)
    r = getResult(cm7T, 2)
    r.insert(0, 'Test 7')
    dfResults = pd.DataFrame([r], columns=col)
    results = results.append(dfResults, ignore_index=True)
    r = getResult(cm9T, 2)
    r.insert(0, ' Test 9')
    dfResults = pd.DataFrame([r], columns=col)
    results = results.append(dfResults, ignore_index=True)
    r = getResult(cm11T, 2)
    r.insert(0, 'Test 11')
    dfResults = pd.DataFrame([r], columns=col)
    results = results.append(dfResults, ignore_index=True)
    print(results)

    results.to_csv('Filters_UNSW-NB15_results.csv', index=False)






if __name__ == "__main__":
    main()