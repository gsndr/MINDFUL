import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Plot():

    def printPlotLoss(history,d, path):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(path+"plotLossAIDA" + str(d) + ".png")
        plt.close()
        # plt.show()


    def printPlotLossDouble(history,d, path):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(path+"plotLossAIDA" + str(d) + ".png")
        plt.close()
        # plt.show()

    def printPlotAccuracy(history, d ,path):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(path+"plotAccuracy" + str(d) + ".png")
        plt.close()

    # plot histogramm distribution class for test and train
    def plotDistribution(self, train, test, LABELS, clsT, clsTest):
        # plot frequency train
        count_classes = pd.value_counts(train[clsT], sort=True)
        count_classes.plot(kind='bar', rot=0)
        plt.title("Transaction class distribution train")
        plt.xticks(range(2), LABELS)
        plt.xlabel("Class")
        plt.ylabel("Frequency");
        plt.plot()
        plt.show()

        # plot frequency test
        count_classes = pd.value_counts(test[clsTest], sort=True)
        count_classes.plot(kind='bar', rot=0)
        plt.title("Transaction class distribution test")
        plt.xticks(range(2), LABELS)
        plt.xlabel("Class")
        plt.ylabel("Frequency");
        plt.plot()
        plt.show()

    # plot histogram distribution errror
    def plotErroHistogram(self, error_df, threshold, dsName):
        """Plot two histogram of error distribution: Normal and Attacks
           Argument:
           error_df -- dataframe with true class and error recostruction of autoencoder
           threshold -- filter error recosntruction less threshold
           dsName -- name file to save

           Returns:
           """

        fig = plt.figure()

        ax = fig.add_subplot(211)
        attack_error_df = error_df[(error_df['true_class'] == 0) & (error_df['reconstruction_error_attack'] < threshold)]
        _ = ax.hist(attack_error_df.reconstruction_error_attack.values, bins=10, label='Attacks')
        ax.set_title('Attacks')
        ax.set_ylabel('num Examples')
        ax.set_xlabel('error')
        ax1 = fig.add_subplot(212)
        normal_error_df = error_df[(error_df['true_class'] == 1) & (error_df['reconstruction_error_normal'] < threshold)]
        _ = ax1.hist(normal_error_df.reconstruction_error_normal.values, bins=10, label='Normal')
        ax1.set_title('Normal')
        ax1.set_ylabel('num Examples')
        ax1.set_xlabel('error')
        plt.tight_layout()
        plt.savefig(dsName + '.png')
        plt.close()

    # plot scatter with normal points in the foreground
    def plotScatterIndex01(self, error_df, dsName, removeOut):
        """Plot  scatter error grouped by true class with index on x and reconstruction errror on y axes
                 Argument:
                 error_df -- dataframe with true class and error recostruction of autoencoder
                 threshold -- line to plot
                 dsName -- name file to save
                 removeOut -- remove all error greater than this value

                 Returns:
        """
        error_dFilter = error_df[error_df.reconstruction_error < removeOut]
        groups = error_dFilter.groupby('true_class')
        fig, ax = plt.subplots()

        for name, group in groups:
            ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
                    label="Attacks" if name == 0 else "Normal")
        # ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for different classes in " + dsName)
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        plt.savefig(dsName + '.png')
        plt.close()

    # plot scatter with attack points in the foreground
    def plotScatterIndex(self, error_df, dsName, removeOut):
        """Plot  scatter error grouped by true class with index on x and reconstruction errror on y axes (visible attacks)
                         Argument:
                         error_df -- dataframe with true class and error recostruction of autoencoder
                         dsName -- name file to save
                         removeOut -- remove all error greater than this value

                         Returns:
        """
        error_dFilter = error_df[error_df.reconstruction_error < removeOut]
        groupsA1 = error_dFilter[error_dFilter['true_class'] == 0]
        groupsN1 = error_dFilter[error_dFilter['true_class'] == 1]
        groupsA = groupsA1.groupby('true_class')
        groupsN = groupsN1.groupby('true_class')
        fig, ax = plt.subplots()
        for name2, group2 in groupsN:
            ax.plot(group2.index, group2.reconstruction_error, marker='o', ms=3.5, linestyle='', c='b',
                    label="Normal")
        for name, group in groupsA:
            ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='', c='r',
                    label="Attacks")
        ax.legend()
        plt.title("Reconstruction error for different classes in " + dsName)
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        plt.savefig(dsName + '.png')
        plt.close()

    # plot scatter only normal predictions with normal points in the foreground
    def plotScatterIndexError01(self, error_df, dsName, removeOut):
        """Plot  scatter error on only normal predict  grouped by true class with index on x and reconstruction errror on y axes (visible true normal)
                                Argument:
                                error_df -- dataframe with true class and error recostruction of autoencoder
                                dsName -- name file to save
                                threshold -- for line
                                removeOut -- remove all error greater than this value

                                Returns:
        """
        error_dFilter = error_df[error_df.reconstruction_error < removeOut]
        error_dFilter.sort_values('true_class', ascending=False)
        # print(error_dFilter.head(8))
        groups = error_dFilter.groupby('true_class')
        fig, ax = plt.subplots()

        for name, group in groups:
            ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
                    label="False_Normal" if name == 0 else "True_Normal")
        # ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=110, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for Normal different classes in " + dsName)
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        plt.savefig(dsName + '.png')
        plt.close()

    # plot scatter only normal predictions with attacks points in the foreground
    def plotScatterIndexError(self, error_df, dsName, removeOut):
        """Plot  scatter error on only normal predict  grouped by true class with index on x and reconstruction errror on y axes (visible false normal)
                                      Argument:
                                      error_df -- dataframe with true class and error recostruction of autoencoder
                                      dsName -- name file to save
                                      threshold -- for line
                                      removeOut -- remove all error greater than this value

                                      Returns:
              """
        error_dFilter = error_df[error_df.reconstruction_error < removeOut]
        groupsA1 = error_dFilter[error_dFilter['true_class'] == 0]
        groupsN1 = error_dFilter[error_dFilter['true_class'] == 1]
        groupsA = groupsA1.groupby('true_class')
        groupsN = groupsN1.groupby('true_class')
        fig, ax = plt.subplots()
        for name2, group2 in groupsN:
            ax.plot(group2.index, group2.reconstruction_error, marker='o', ms=3.5, linestyle='', c='b',
                    label="True_Normal")
        for name, group in groupsA:
            ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='', c='r',
                    label="False_Normal")
        ax.legend()
        plt.title("Reconstruction error for Normal different classes in " + dsName)
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        plt.savefig(dsName + '.png')
        plt.close()

    # plot scatter only normal predictions with attacks points in the foreground
    def plotScatterError_Prob(self, error_df, dsName, removeOut, removeProb=None):
        """Plot  scatter error on only normal predict  grouped by true class with index on x and reconstruction errror on y axes (visible false normal)
                                      Argument:
                                      error_df -- dataframe with true class and error recostruction of autoencoder
                                      dsName -- name file to save
                                      threshold -- for line
                                      removeOut -- remove all error greater than this value

                                      Returns:
              """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        error_dFilter = error_df[error_df.reconstruction_error < removeOut]
        if (removeProb is not None):
            error_dFilter = error_dFilter[error_df.prob > removeProb]

        groupsA1 = error_dFilter[error_dFilter['predict_softmax'] == 0]
        groupsN1 = error_dFilter[error_dFilter['predict_softmax'] == 1]
        print(groupsN1.head(10))
        ATrue = groupsA1[groupsA1['true_class'] == 0]
        AFalse = groupsA1[groupsA1['true_class'] == 1]
        NTrue = groupsN1[groupsN1['true_class'] == 1]
        NFalse = groupsN1[groupsN1['true_class'] == 0]
        print(ATrue.head(8))

        groupsN = groupsN1.groupby('true_class')
        # groups = error_dFilter.groupby('true_class')
        ax.scatter(x=ATrue.prob, y=ATrue.reconstruction_error, c='r', marker='o', label='True Attacks')
        ax.legend()
        ax.scatter(x=AFalse.prob, y=AFalse.reconstruction_error, c='b', marker='o',
                   label='False Attacks'
                   )
        ax.legend()

        ax.set_title('Predict Attacks')
        ax.set_ylabel('mse')
        ax.set_xlabel('Reconstruction error')
        plt.savefig(dsName + 'Attacks.png')
        plt.close()

        fig2 = plt.figure()
        ax1 = fig2.add_subplot(111)
        ax1.scatter(x=NTrue.prob, y=NTrue.reconstruction_error, c='b', marker='o', label='True Normal')
        ax1.legend()
        ax1.scatter(x=NFalse.prob, y=NFalse.reconstruction_error, c='r', marker='o',
                    label='False Normal'
                    )
        ax1.legend()
        ax1.set_title('Predict Normal')
        ax1.set_ylabel('mse')
        ax1.set_xlabel('Probability')
        plt.savefig(dsName + 'Normal.png')
        plt.close()

    # plot scatter with two different figure for normal and attacks
    def plotScatterIndexDouble(self, error_df, dsName, removeOut):
        """Plot  scatter error with two different figure separate for normal and attacks reconstruction errors
              Argument:
              error_df -- dataframe with true class and error recostruction of autoencoder
              dsName -- name file to save
              removeOut -- remove all error greater than this value

              Returns:
        """
        fig = plt.figure()
        ax = fig.add_subplot(211)
        error_dFilter = error_df[error_df.reconstruction_error < removeOut]
        groupsA1 = error_dFilter[error_dFilter['true_class'] == 0]
        print(error_dFilter.shape)
        print(groupsA1.shape)
        groupsN1 = error_dFilter[error_dFilter['true_class'] == 1]
        groupsA = groupsA1.groupby('true_class')
        groupsN = groupsN1.groupby('true_class')
        groups = error_dFilter.groupby('true_class')
        # fig, ax = plt.subplots()

        for name, group in groupsA:
            ax.plot(group.index, group.reconstruction_error, marker='x', ms=3.5, linestyle='', c='b',
                    label="False_Normal")
        ax.set_title('False Normal')
        ax.set_ylabel('mse for False')
        # ax.set_xlabel('Reconstruction error')
        ax1 = fig.add_subplot(212)
        for name2, group2 in groupsN:
            ax1.plot(group2.index, group2.reconstruction_error, marker='o', ms=3.5, linestyle='',
                     label="True_Normal")
        ax1.set_title('True Normal')
        ax1.set_ylabel('mse  for True')
        ax1.set_xlabel('Reconstruction error')

        plt.xlabel("Data point index")
        plt.savefig(dsName + '.png')
        plt.close()

    def plotScatter(self, error_df, threshold, dsName):
        error_dFilter = error_df[error_df.reconstruction_error < threshold]
        fig, ax = plt.subplots()
        ax.scatter(y=error_dFilter['true_class'], x=error_dFilter['reconstruction_error'])
        ax.legend()
        plt.title("Reconstruction error for different classes in " + dsName)
        plt.ylabel("Class")
        plt.xlabel("Reconstruction error")
        plt.savefig(dsName + '.png')
        plt.close()

    # plot two reconstruction errors
    def plotErrors(selfself, error_df, dsName, thresholdN=None, thresholdA=None):
        """Plot  scatter with recostrunction erros attack autoencoder on y axis and
         recostrunction errors normal autoencoder on x axis
                      Argument:
                      error_df -- dataframe with true class and error recostruction of autoencoder
                      dsName -- name file to save
                      thresholdN -- remove all error on x axis error greater than this value
                      thresholdA -- remove all error on y  axis error greater than this value

                      Returns:
                """
        if thresholdN is not None:
            error_df = error_df[error_df.reconstruction_error_normal < thresholdN]
        if thresholdA is not None:
            error_df = error_df[error_df.reconstruction_error_attack < thresholdA]
        # print(error_dFilter.head(8))
        groups = error_df.groupby('true_class')
        # print(groups.groups)
        fig, ax = plt.subplots()

        for name, group in groups:
            ax.plot(group.reconstruction_error_normal, group.reconstruction_error_attack, marker='o', ms=3.5,
                    linestyle='',
                    label="Attacks" if name == 0 else "Normal")
        # ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=110, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for two autoencoder")
        plt.ylabel("Mse Attack")
        plt.xlabel("Mse Normal")
        plt.savefig(dsName + '.png')
        plt.close()

        # plot two reconstruction errors

    def plotErrorsDouble(selfself, error_df, dsName, thresholdN=None, thresholdA=None):
        """Plot  scatter with recostrunction erros attack autoencoder on y axis and
         recostrunction errors normal autoencoder on x axis
                      Argument:
                      error_df -- dataframe with true class and error recostruction of autoencoder
                      dsName -- name file to save
                      thresholdN -- remove all error on x axis error greater than this value
                      thresholdA -- remove all error on y  axis error greater than this value

                      Returns:
                """

        if thresholdN is not None:
            error_df = error_df[error_df.reconstruction_error_normal < thresholdN]
        if thresholdA is not None:
            error_df = error_df[error_df.reconstruction_error_attack < thresholdA]
        # print(error_dFilter.head(8))

        groupsA1 = error_df[error_df['true_class'] == 0]
        groupsN1 = error_df[error_df['true_class'] == 1]
        groupsA = groupsA1.groupby('true_class')
        groupsN = groupsN1.groupby('true_class')

        groups = error_df.groupby('true_class')
        # print(groups.groups)
        fig, ax = plt.subplots()
        ax.scatter(y=groupsA1.reconstruction_error_attack, x=groupsA1.reconstruction_error_normal, c='red')
        # ax.plot(groupsA1.reconstruction_error_attack, groupsA1.reconstruction_error_normal, marker='o', ms=3.5)

        # ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=110, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for two autoencoder")
        plt.ylabel("Mse Attack")
        plt.xlabel("Mse Normal")
        plt.savefig(dsName + 'Attack.png')
        plt.close()
        fig, ax = plt.subplots()
        ax.scatter(groupsN1.reconstruction_error_attack, groupsN1.reconstruction_error_normal)
        # ax.plot(groupsN1.reconstruction_error_attack, groupsN1.reconstruction_error_normal, marker='o', ms=3.5)

        # ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=110, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for two autoencoder")
        plt.ylabel("Mse Attack")
        plt.xlabel("Mse Normal")
        plt.savefig(dsName + 'Normal.png')
        plt.close()

    def plotErrors5(self, error_df, dsName, thresholdN=None, thresholdA=None):
        if thresholdN is not None:
            error_df = error_df[error_df.reconstruction_error_normal < thresholdN]
        if thresholdA is not None:
            error_df = error_df[error_df.reconstruction_error_attack < thresholdA]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        Dos = error_df[error_df['true_class5'] == 0]
        Probe = error_df[error_df['true_class5'] == 1]
        U2R = error_df[error_df['true_class5'] == 2]
        R2L = error_df[error_df['true_class5'] == 3]
        Normal = error_df[error_df['true_class5'] == 4]

        # groups = error_dFilter.groupby('true_class')
        ax.scatter(x=Normal.reconstruction_error_normal, y=Normal.reconstruction_error_attack, c='b', marker='o',
                   label='normal')
        ax.legend()
        ax.scatter(x=Dos.reconstruction_error_normal, y=Dos.reconstruction_error_attack, c='r', marker='o', label='Dos')
        ax.legend()
        ax.scatter(x=Probe.reconstruction_error_normal, y=Probe.reconstruction_error_attack, c='m', marker='o',
                   label='Probe')
        ax.legend()
        ax.scatter(x=U2R.reconstruction_error_normal, y=U2R.reconstruction_error_attack, c='y', marker='o',
                   label='U2R')
        ax.legend()
        ax.scatter(x=R2L.reconstruction_error_normal, y=R2L.reconstruction_error_attack, c='g', marker='o',
                   label='R2L')
        ax.legend()

        ax.set_title('Reconstruction for class')
        ax.set_ylabel('Reconstruction error Attack')
        ax.set_xlabel('Reconstruction error normal')
        plt.savefig(dsName + '.png')
        plt.close()

        # plot scatter only normal predictions with attacks points in the foreground

    def plotScatterError_Prob5(self, error_df, dsName, removeOut=None, removeProb=None):
        """Plot  scatter error on only normal predict  grouped by true class with index on x and reconstruction errror on y axes (visible false normal)
                                      Argument:
                                      error_df -- dataframe with true class and error recostruction of autoencoder
                                      dsName -- name file to save
                                      threshold -- for line
                                      removeOut -- remove all error greater than this value

                                      Returns:
              """
        if removeOut is not None:
            error_df = error_df[error_df.reconstruction_error < removeOut]
        if removeProb is not None:
            error_df = error_df[error_df.prob > removeProb]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        Dos = error_df[error_df['true_class5'] == 0]
        Probe = error_df[error_df['true_class5'] == 1]
        U2R = error_df[error_df['true_class5'] == 2]
        R2L = error_df[error_df['true_class5'] == 3]
        Normal = error_df[error_df['true_class5'] == 4]

        # groups = error_dFilter.groupby('true_class')
        ax.scatter(x=Normal.prob, y=Normal.reconstruction_error, c='b', marker='o',
                   label='normal')
        ax.legend()
        ax.scatter(x=Dos.prob, y=Dos.reconstruction_error, c='r', marker='o', label='Dos')
        ax.legend()
        ax.scatter(x=Probe.prob, y=Probe.reconstruction_error, c='m', marker='o',
                   label='Probe')
        ax.legend()
        ax.scatter(x=U2R.prob, y=U2R.reconstruction_error, c='y', marker='o',
                   label='U2R')
        ax.legend()
        ax.scatter(x=R2L.prob, y=R2L.reconstruction_error, c='g', marker='o',
                   label='R2L')
        ax.legend()

        ax.set_title('Reconstruction for class')
        ax.set_ylabel('Reconstruction error')
        ax.set_xlabel('Prob softmax')
        plt.savefig(dsName + '.png')
        plt.close()

    # plot histogram distribution errror
    def plotErroHistogramOnlySingle(self, error_df, dsName, threshold=1.0):
        """Plot two histogram of error distribution: Normal and Attacks
           Argument:
           error_df -- dataframe with true class and error recostruction of autoencoder
           threshold -- filter error recosntruction less threshold
           dsName -- name file to save

           Returns:
           """

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        normal_error_df = error_df[
            (error_df['true_class'] == 0) & (error_df['reconstruction_error_normal'] < threshold)]
        _ = ax1.hist(normal_error_df.reconstruction_error_normal.values, bins=10, label='Attacks', color='red')
        n_error_df = error_df[
            (error_df['true_class'] == 1) & (error_df['reconstruction_error_normal'] < threshold)]
        _ = ax2.hist(n_error_df.reconstruction_error_normal.values, bins=10, label='Normal')
        ax1.set_title('Normal')
        ax1.set_ylabel('num Examples')
        ax1.set_xlabel('error')
        plt.tight_layout()
        plt.savefig(dsName + '.png')
        plt.close()

