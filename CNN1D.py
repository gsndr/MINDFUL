import numpy as np

my_seed = 12
np.random.seed(my_seed)
import random

random.seed(my_seed)


import tensorflow

tensorflow.set_random_seed(my_seed)

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from Preprocessing import Preprocessing as prep
from DatasetsConfig import Datasets
from Plot import Plot
from keras import callbacks
from keras.utils import np_utils

from sklearn.metrics import confusion_matrix
from keras.models import Model

from keras.models import load_model
from keras import backend as K
from keras.utils import plot_model
np.set_printoptions(suppress=True)


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
    r = (tp, fn, fp, tn, OA, AA, P, R, F1, FAR, TPR)
    return r



class RunCNN1D():
    def __init__(self, dsConfig, config):
        self.config = config
        self.ds = dsConfig



    def createImage(self, train_X, trainA, trainN):
        rows = [train_X, trainA, trainN]
        rows = [list(i) for i in zip(*rows)]

        train_X = np.array(rows)

        if K.image_data_format() == 'channels_first':
            x_train = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2])
            input_shape = (train_X.shape[1], train_X.shape[2])
        else:
            x_train = train_X.reshape(train_X.shape[0], train_X.shape[2], train_X.shape[1])
            input_shape = (train_X.shape[2], train_X.shape[1])
        return x_train, input_shape



    def run(self):

        print('MINDFUL EXECUTION')

        dsConf = self.ds
        pathModels = dsConf.get('pathModels')
        pathPlot = dsConf.get('pathPlot')
        configuration = self.config


        VALIDATION_SPLIT = float(configuration.get('VALIDATION_SPLIT'))
        N_CLASSES = int(configuration.get('N_CLASSES'))
        pd.set_option('display.expand_frame_repr', False)

        # contains path of dataset and model and preprocessing phases
        ds = Datasets(dsConf)
        ds.preprocessing1()
        train, test = ds.getTrain_Test()
        prp = prep(train, test)

        # Preprocessing phase from original to numerical dataset
        PREPROCESSING1 = int(configuration.get('PREPROCESSING1'))
        if (PREPROCESSING1 == 1):

            train, test = ds.preprocessing2(prp)
        else:
            train, test = ds.getNumericDatasets()


        clsT, clsTest = prp.getCls()
        train_normal = train[(train[clsT] == 1)]






        train_anormal = train[(train[clsT] == 0)]
        test_normal = test[(test[clsTest] == 1)]
        test_anormal = test[(test[clsTest] == 0)]

        train_XN, train_YN, test_XN, test_YN = prp.getXY(train_normal, test_normal)

        train_XA, train_YA, test_XA, test_YA = prp.getXY(train_anormal, test_anormal)
        train_X, train_Y, test_X, test_Y = prp.getXY(train, test)




        print('Train data shape normal', train_XN.shape)
        print('Train target shape normal', train_YN.shape)
        print('Test data shape normal', test_XN.shape)
        print('Test target shape normal', test_YN.shape)

        print('Train data shape anormal', train_XA.shape)
        print('Train target shape anormal', train_YA.shape)
        print('Test data shape anormal', test_XA.shape)
        print('Test target shape anormal', test_YA.shape)

        # convert class vectors to binary class matrices fo softmax
        train_Y2 = np_utils.to_categorical(train_Y, int(configuration.get('N_CLASSES')))
        print("Target train shape after", train_Y2.shape)
        test_Y2 = np_utils.to_categorical(test_Y, int(configuration.get('N_CLASSES')))
        print("Target test shape after", test_Y2.shape)
        print("Train all", train_X.shape)
        print("Test all", test_X.shape)



        # create pandas for results
        columns = ['TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
        results = pd.DataFrame(columns=columns)

        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, restore_best_weights=True),
        ]

        if (int(configuration.get('LOAD_AUTOENCODER_NORMAL')) == 0):


            autoencoderN, p = ds.getAutoencoder_Normal(train_XN, N_CLASSES)
            autoencoderN.summary()

            history = autoencoderN.fit(train_XN, train_XN,
                                       validation_split=VALIDATION_SPLIT,
                                       batch_size=p['batch_size'],
                                       epochs=p['epochs'], shuffle=True,
                                       callbacks=callbacks_list,
                                       verbose=1)
            autoencoderN.save(pathModels + 'autoencoderNormal.h5')
            Plot.printPlotLoss(history, 'autoencoderN', pathPlot)
        else:
            print("Load autoencoder Normal from disk")
            autoencoderN = load_model(pathModels + 'autoencoderNormal.h5')
            autoencoderN.summary()
            plot_model(autoencoderN, to_file='model.png')

        train_RE = autoencoderN.predict(train_X)
        test_RE = autoencoderN.predict(test_X)



        if (int(configuration.get('LOAD_AUTOENCODER_ADV')) == 0):


            autoencoderA, p = ds.getAutoencoder_Attacks(+train_XA, N_CLASSES)


            autoencoderA.summary()

            history = autoencoderA.fit(train_XA, train_XA,
                                       validation_split=VALIDATION_SPLIT,
                                       batch_size=p['batch_size'],
                                       epochs=p['epochs'], shuffle=True,
                                       callbacks=callbacks_list,
                                       verbose=1)
            autoencoderA.save(pathModels + 'autoencoderAttacks.h5')
            Plot.printPlotLoss(history, 'autoencoderA', pathPlot)
        else:
            print("Load autoencoder Attacks from disk")
            autoencoderA = load_model(pathModels + 'autoencoderAttacks.h5')
            autoencoderA.summary()

        train_REA = autoencoderA.predict(train_X)
        test_REA = autoencoderA.predict(test_X)



        train_X_image, input_Shape = self.createImage(train_X, train_RE, train_REA)  # XS UNSW
        test_X_image, input_shape = self.createImage(test_X, test_RE, test_REA)



        if (int(configuration.get('LOAD_CNN')) == 0):
            callbacks_list = [
                callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20,
                                        restore_best_weights=True),
            ]

            model, p = ds.getMINDFUL(input_shape, N_CLASSES)



            history3 = model.fit(train_X_image, train_Y2,
                                 # validation_data=(test_X, test_Y2),
                                 validation_split=VALIDATION_SPLIT,
                                 batch_size=p['batch_size'],
                                 epochs=p['epochs'], shuffle=True,  # shuffle=false for NSL-KDD true for UNSW-NB15
                                 callbacks=callbacks_list,  # class_weight=class_weight,
                                 verbose=1)

            Plot.printPlotAccuracy(history3, 'finalModel1', pathPlot)
            Plot.printPlotLoss(history3, 'finalModel1', pathPlot)
            model.save(pathModels + 'MINDFUL.h5')
        else:
            print("Load softmax from disk")
            model = load_model(pathModels + 'MINDFUL.h5')
            model.summary()



        predictionsL = model.predict(train_X_image)
        y_pred = np.argmax(predictionsL, axis=1)
        cmC = confusion_matrix(train_Y, y_pred)
        print('Prediction Training')
        print(cmC)

        predictionsL = model.predict(test_X_image)
        y_pred = np.argmax(predictionsL, axis=1)
        cm = confusion_matrix(test_Y, y_pred)
        print('Prediction Test')
        print(cm)

        r = getResult(cm, N_CLASSES)


        dfResults = pd.DataFrame([r], columns=columns)
        print(dfResults)


        results = results.append(dfResults, ignore_index=True)


        results.to_csv(ds._testpath + '_results.csv', index=False)



