import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import scale, MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder



class Preprocessing():
    def __init__(self, train, test):
        #classification column
        self._clsTrain = train.columns[-1]

        #only categorical features
        obj_dfTrain = train.select_dtypes(include=['object']).copy()
        self._objectListTrain = obj_dfTrain.columns.values

        #remove classification column
        self._objectListTrain = np.delete(self._objectListTrain, -1)
        #classification test column
        self._clsTest = test.columns[-1]

        # ronly categorical features test set
        obj_dfTest = test.select_dtypes(include=['object']).copy()
        self._objectListTest= obj_dfTest.columns.values

       #remove classification column from test set
        self._objectListTest = np.delete(self._objectListTest, -1)





    def getCls(self):
        return self._clsTrain, self._clsTest



    def labelCategorical(self, train,test):
        le = LabelEncoder()
        train_df=train.copy()
        test_df=test.copy()
        AllT= train_df.append(test_df, ignore_index=True)
        for col in self._objectListTrain:
            le.fit(AllT[col])
            train[col]=le.transform(train[col])
            test[col]=le.transform(test[col])
        return train,test




    def mapLabel(self, df):
        # creating labelEncoder
        le = preprocessing.LabelEncoder()
        cls_encoded = le.fit_transform(df[self._clsTrain])
        df[self._clsTrain] = le.transform(df[self._clsTrain])
        return cls_encoded



    #map label classification to number
    def preprocessinLabel(self,train, test):
        self.mapLabel(train)
        self.mapLabel(test)
        return train, test







    def minMaxScale(self, Y_train, Y_test):
        scaler = preprocessing.MinMaxScaler()
        Y_train=scaler.fit_transform(Y_train)
        Y_test=scaler.transform(Y_test)
        return Y_train, Y_test


    def getXY(self, train, test):
        clssList = train.columns.values
        target=[i for i in clssList if i.startswith(self._clsTrain)]

        # remove label from dataset to create Y ds
        train_Y=train[target]
        test_Y=test[target]
        # remove label from dataset
        train_X = train.drop(target, axis=1)
        train_X=train_X.values
        test_X = test.drop(target, axis=1)
        test_X=test_X.values

        return train_X, train_Y, test_X, test_Y


    def getXYCICIDS(self, train, tests ):
        clssList = train.columns.values
        #target = [i for i in clssList if i.startswith(' classification')]
        target=[i for i in clssList if i.startswith(self._clsTrain)]

        # remove label from dataset to create Y ds
        train_Y=train[target]
        train_X = train.drop(target, axis=1)
        train_X = train_X.values
        test_Y=list()
        test_X=list()
        for test in tests:
            t_Y=test[target]
            t_X = test.drop(target, axis=1)
            t_X = t_X.values
            test_Y.append(t_Y)
            test_X.append(t_X)

        return train_X, train_Y, test_X, test_Y

    def getXYTrain(self, train):
        clssList = train.columns.values
        # target = [i for i in clssList if i.startswith(' classification')]
        target = [i for i in clssList if i.startswith(self._clsTrain)]

        # remove label from dataset to create Y ds
        train_Y = train[target]
        train_X = train.drop(target, axis=1)
        train_X = train_X.values

        return train_X, train_Y





    def preprocessing(self,train, test, p):
        train,test=self.labelCategorical(train,test)
        train, test = p.preprocessinLabel(train, test)
        missing_cols = set(train.columns) - set(test.columns)
        for c in missing_cols:
            test[c] = 0
        test = test[train.columns]

        missing_cols = set(test.columns) - set(train.columns)
        for c in missing_cols:
            train[c] = 0
        train = train[test.columns]

        return train, test

    def preprocessingCICIDS(self, train,testList,p):
        tests=list()
        for test in testList:
            train, test = p.preprocessingOneHot(train, test)
            train, test = p.preprocessinLabel(train, test)
            missing_cols = set(train.columns) - set(test.columns)
            for c in missing_cols:
                 test[c] = 0
            test = test[train.columns]
            missing_cols = set(test.columns) - set(train.columns)
            for c in missing_cols:
                train[c] = 0
            train = train[test.columns]
            tests.append(test)
        return train,tests






    def scaler(self,train, test, listContent):
        scaler = MinMaxScaler()
        print('Scaling')

        scaler.fit(train[listContent].values)
        train[listContent] = scaler.transform(train[listContent])
        test[listContent] = scaler.transform(test[listContent])
        return train, test


    def scalerCICIDS(self,train, testList, listContent):
        tests = list()
        listContent = list(listContent)
        scaler = MinMaxScaler()
        scaler.fit(train[listContent].values)
        train[listContent] = scaler.transform(train[listContent])
        for test in testList:
            test[listContent] = scaler.transform(test[listContent])
            tests.append(test)
        return train, tests








