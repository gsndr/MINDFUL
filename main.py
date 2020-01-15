import configparser
import sys
import numpy as np


np.random.seed(12)
import tensorflow
tensorflow.set_random_seed(12)
from CICIDS2017 import RunCNN1DCICIDS
from CNN1D import RunCNN1D





def datasetException():
    try:
        dataset=sys.argv[1]

        if (dataset is None) :
            raise Exception()
        if not ((dataset == 'KDDCUP99') or (dataset == 'UNSW_NB15') or (dataset == 'CICIDS2017') ):
            raise ValueError()
    except Exception:
        print("The name of dataset is null: use KDDTest+ or KDDTest-21 or UNSW_NB15 or CICIDS2017")
    except ValueError:
        print ("Dataset not exist: must be KDDTest+ or KDDTest-21 or UNSW_NB15 or CICIDS2017")
    return dataset





def main():

    dataset=datasetException()

    config = configparser.ConfigParser()
    config.read('MINDFUL.conf')

    dsConf = config[dataset]
    configuration = config['setting']






    if(dataset=='CICIDS2017'):
        execution=RunCNN1D_CICIDS(dsConf,configuration)

    else:
        execution=RunCNN1D(dsConf,configuration)


    execution.run()



if __name__ == "__main__":
    main()