import pandas as pd
import numpy as np
from Preprocessing import Preprocessing as prep
from keras.optimizers import Adam, Nadam, SGD, RMSprop, Adadelta, Adagrad, Adamax
from NNs import Models
from keras.initializers import RandomNormal

class Datasets():
    def __init__(self, dsConf):

        #classification column
        self.pathModels = dsConf.get('pathModels')
        self._pathDataset = dsConf.get('pathDataset')
        self._path = dsConf.get('path')


        self._testpath = dsConf.get('testpath')
        self._pathDatasetNumeric = dsConf.get('pathDatasetNumeric')

        self._train = pd.read_csv(self._pathDataset + self._path + ".csv")


        self._pathOutputTrain = self._pathDatasetNumeric + self._path+'Numeric.csv'
        if(self._testpath== 'CICIDS2017'):
            self._pathTest = dsConf.get('pathTest').split(',')
        else:
            self._pathTest = dsConf.get('pathTest')




    def getTrain_Test(self):
        return self._train, self._test

    def getTrain_TestCIDIS(self):
        return self._train, self._tests[0]



    def preprocessing1(self):

        if ((self._testpath== 'KDDCUP')):
            self._test = pd.read_csv(self._pathDataset + self._pathTest + ".csv")
            print('Using:' + self._testpath)
            self._listNumerical10=self._train.columns.values
            index = np.argwhere(self._listNumerical10 == ' classification.')
            self._listNumerical10 = np.delete(self._listNumerical10, index)
            #print(self._listNumerical10)

        elif (self._testpath== 'UNSW_NB15'):
            print('Using:' + self._testpath)
            self._test = pd.read_csv(self._pathDataset + self._pathTest + ".csv")
            cls = ['classification']
            listCategorical = ['proto', 'service', 'state']
            listBinary = ['is_ftp_login', 'is_sm_ips_ports']
            listAllColumns = self._train.columns.values
            self._listNumerical10 = list(set(listAllColumns) - set(listCategorical) - set(listBinary) - set(cls))



        elif (self._testpath== 'CICIDS2017'):
            print('Using:' + self._testpath)
            self._train.rename(
                columns={' Flow Packets/s': 'Flow_Packets', 'Flow Bytes/s': 'Flow_Bytes', ' Label': 'Classification'},
                inplace=True)

            cls = ['Classification']
            label = ['BENIGN', 'ATTACK']
            print(self._train.columns)
            listCategorical = ['Flow ID, Source IP, Destination IP, Timestamp, External IP']
            self._train['Flow_Bytes'].fillna((0), inplace=True)
            self._train['Flow_Bytes'] = self._train['Flow_Bytes'].astype(float)

            Pack = self._train[self._train.Flow_Packets != 'Infinity']
            Bytes = self._train[self._train.Flow_Bytes != np.inf]
            maxPack = np.max(Pack['Flow_Packets'])
            maxBytes = np.max(Bytes['Flow_Bytes'])
            col_names = self._train.columns
            self._train['Flow_Packets'].replace(to_replace=dict(Infinity=maxPack), inplace=True)
            self._train['Flow_Bytes'].replace((np.inf, maxBytes), inplace=True)
            # train_df['Flow_Bytes'].replace(to_replace=dict(np.inf=maxBytes),inplace=True)
            self._train["Classification"].replace(to_replace=dict(BENIGN=1, ATTACK=0), inplace=True)

            nominal_inx = [] #Gargaro vuoto
            binary_inx = [
                'Fwd PSH Flags,  Bwd PSH Flags,  Fwd URG Flags,  Bwd URG Flags, FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count,'
                'ACK Flag Count	,URG Flag Count, CWE Flag Count,  ECE Flag Count,  Fwd Header Length.1,Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk,'
                ' Fwd Avg Bulk Rate	 Bwd Avg Bytes/Bulk	 Bwd Avg Packets/Bulk	Bwd Avg Bulk Rate']

            binary_inx = [30, 31, 32, 33, 43, 44, 45, 46, 47, 48, 49, 50, 56, 57, 58, 59, 60, 61]
            numeric_inx = list(set(range(78)).difference(nominal_inx).difference(binary_inx))

            self._listNumerical10 = col_names[numeric_inx].tolist()
            self._tests = list()
            for testset in self._pathTest:
                test = pd.read_csv(self._pathDataset + testset + ".csv")
                test.rename(
                    columns={' Flow Packets/s': 'Flow_Packets', 'Flow Bytes/s': 'Flow_Bytes',
                             ' Label': 'Classification'},
                    inplace=True)
                test['Flow_Packets'].replace(to_replace=dict(Infinity=maxPack), inplace=True)

                test['Flow_Bytes'].replace(to_replace=dict(Infinity=maxBytes), inplace=True)
                test["Classification"].replace(to_replace=dict(BENIGN=1, ATTACK=0), inplace=True)
                self._tests.append(test)






    def preprocessing2(self, prp):
        if(self._testpath == 'CICIDS2017'):
            train, tests = prp.preprocessingCICIDS(self._train, self._tests, prp)
            train, test = prp.scalerCICIDS(train, tests, self._listNumerical10)
            train.to_csv(self._pathOutputTrain, index=False)  # X is an array
            for t, pathTest in zip(test, self._pathTest):
                self._pathOutputTest = self._pathDatasetNumeric + pathTest + 'Numeric.csv'
                t.to_csv(self._pathOutputTest, index=False)

        else:
            self._pathOutputTest = self._pathDatasetNumeric + self._pathTest + 'Numeric.csv'
            train, test = prp.preprocessing(self._train, self._test, prp)
            train, test = prp.scaler(train, test,  self._listNumerical10)
            train.to_csv(self._pathOutputTrain, index=False)  # X is an array
            test.to_csv(self._pathOutputTest, index=False)

        return train,test



    def getAutoencoder1(self, train_XN, N_CLASSES):

        m = Models(N_CLASSES)
        if ((self._testpath == 'KDDCUP') or (self._testpath == 'KDDTest-21')):


            p = {
                'first_layer': 30,
                'second_layer': 20,
                'third_layer': 10,
                'batch_size': 64,
                'epochs': 150,
                'optimizer': Adadelta,
                'kernel_initializer': 'glorot_uniform',
                'losses': 'mse',
                'dropout_rate': 0,
                'first_activation': 'relu',
                'second_activation': 'relu',
                'third_activation': 'relu',
                'dropout': 0.11245974166556161

            }
            model = m.deepAutoEncoderUNSW(train_XN, p)
        elif (self._testpath == 'UNSW_NB15'):
            p = {
                'first_layer': 30,
                'second_layer': 20,
                'third_layer': 10,
                'four_layer': 10,
                'five_layer': 20,
                'six_layer': 10,
                'batch_size': 64,
                'epochs': 150,
                'optimizer': Adadelta,
                'dropout_rate': 0,
                'kernel_initializer': 'glorot_uniform',
                'losses': 'mse',
                'first_activation': 'relu',
                'second_activation': 'relu',
                'third_activation': 'relu'}
            model = m.deepAutoEncoderUNSW(train_XN, p)
        elif ((self._testpath== 'CICIDS2017') or (self._testpath== 'CICIDS2017_2')):
            p = {
                'first_layer': 80, #55
                'second_layer': 80, #30
                'third_layer': 30, #10 Gargaro
                'four_layer': 10,
                'five_layer': 20,
                'six_layer': 10,
                'batch_size': 64,
                'epochs': 150,
                'optimizer': Adadelta, #Adam
                'dropout_rate': 0.5,
                'kernel_initializer': 'glorot_uniform',
                'losses': 'mse',
                'first_activation': 'relu', #tanh
                'second_activation': 'relu', #tanh
                'third_activation': 'relu'} #tanh
            model = m.deepAutoEncoderUNSW(train_XN, p)
        return model, p



    def getAutoencoder1_Attacks(self, train_XN, N_CLASSES):
        """setting model and parameter for each dataset for autoencoder 1 attacks

                   Extended description of function.

                   Args: train set normalized
                       (int) numbers of classes


                   Returns: model an
                   d parameters for attacks autoencoder

                   """
        m = Models(N_CLASSES)
        if ((self._testpath == 'KDDCUP') or (self._testpath == 'KDDTest-21')):

            p = {
                'first_layer': 30,
                'second_layer': 20,
                'third_layer': 10,
                'batch_size': 64,
                'epochs': 150,
                'optimizer': Adadelta,
                'kernel_initializer': 'glorot_uniform',
                'losses': 'mse',
                'first_activation': 'relu',
                'second_activation': 'relu',
                'third_activation': 'relu',
                'dropout_rate': 0.11245974166556161

            }

            model = m.deepAutoEncoderUNSW(train_XN, p)
        elif (self._testpath == 'UNSW_NB15'):
            p = {
                'first_layer':30,
                'second_layer': 20,
                'third_layer': 10,
                'four_layer': 10,
                'five_layer': 20,
                'six_layer': 10,
                'batch_size': 64,
                'epochs': 150,
                'optimizer': Adadelta,
                'dropout_rate': 0,
                'kernel_initializer': 'glorot_uniform',
                'losses': 'mse',
                'first_activation': 'relu',
                'second_activation': 'relu',
                'third_activation': 'relu'}
            model = m.deepAutoEncoderUNSW(train_XN, p)
        elif ((self._testpath== 'CICIDS2017') or (self._testpath== 'CICIDS2017_2')):
            p = {
                'first_layer': 100,
                'second_layer': 80, #primo
                'third_layer': 30,
                'four_layer': 10,
                'five_layer': 20,
                'six_layer': 10,
                'batch_size': 64,
                'epochs': 150,
                'optimizer': Adadelta,
                'dropout_rate': 0.5,
                'kernel_initializer': 'glorot_uniform',
                'losses': 'mse',
                'first_activation': 'relu',
                'second_activation': 'relu',
                'third_activation': 'relu'}
            model = m.deepAutoEncoderUNSW(train_XN, p)
        return model, p


    def getNumericDatasets(self):


        if (self._testpath == 'CICIDS2017'):
            train = pd.read_csv(self._pathOutputTrain)
            test=list()
            for testset in self._pathTest:
                print(testset)
                self._pathOutputTest = self._pathDatasetNumeric + testset + 'Numeric.csv'
                t = pd.read_csv(self._pathOutputTest)
                test.append(t)
        else:
            self._pathOutputTest = self._pathDatasetNumeric + self._pathTest + 'Numeric.csv'
            train = pd.read_csv(self._pathOutputTrain)
            test = pd.read_csv(self._pathOutputTest)
        return train, test





    def getTrain_Test(self):
        return self._train, self._test



    def getOnlyCNN_F(self, train, N_CLASSES):

        m = Models(N_CLASSES)
        if ((self._testpath == 'KDDCUP') or (self._testpath == 'KDDTest-21')):
            p = {
                'filter': 64,
                'num_unit': 320,
                'num_unit1': 160,
                'activation': 'tanh',
                'dropout_rate1': 0.15716268492531738,
                'dropout_rate2': 0.5442420919810019,
                'dropout_rate3': 0.43404126142165134,
                'batch_size': 128,
                'epochs': 150,
                'optimizer': Nadam,
                'kernel_initializer': 'glorot_uniform',
                'losses': 'categorical_crossentropy',
                'first_activation': 'tanh',
                'second_activation': 'tanh'
                }

            model = m.Conv1D(train, p)
        elif (self._testpath == 'UNSW_NB15'):
            p = {
                'filter': 64,
                'num_unit': 320,
                'num_unit1': 160,
                'activation': 'tanh',
                'dropout_rate1': 0.3808760346037454,
                'dropout_rate2': 0.4236166283779518,
                'dropout_rate3': 0.18458586014066036,
                'batch_size': 64,
                'epochs': 150,
                'optimizer': Nadam,
                'kernel_initializer': 'glorot_uniform',
                'losses': 'categorical_crossentropy',
                'first_activation': 'tanh',
                'second_activation': 'tanh'
                }
            model = m.Conv1D(train, p)
        elif ((self._testpath == 'CICIDS2017') or (self._testpath == 'CICIDS2017_2')):
            p = {
                'filter': 64,
                'num_unit': 320,
                'num_unit1': 160,
                'activation': 'tanh',
                'dropout_rate1': 0.25634389232531474,
                'dropout_rate2': 0.12064860770171626,
                'dropout_rate3': 0.1602501347478713,
                'batch_size': 64,
                'epochs': 15,
                'optimizer': Nadam,
                'kernel_initializer': 'glorot_uniform',
                'losses': 'categorical_crossentropy',
                'first_activation': 'tanh',
                'second_activation': 'tanh'}
            model = m.Conv1D(train, p)
        return model, p





