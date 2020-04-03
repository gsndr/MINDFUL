# MultI-chanNel Deep FeatUre Learning for intrusion detection (MINDFUL)

The repository contains code refered to the work:

_Giuseppina Andresini, Annalisa Appice, Nicola Di Mauro, Corrado Loglisci, Donato Malerba_

[Multi-Channel Deep Feature Learning for Intrusion Detection](https://ieeexplore.ieee.org/document/9036935) 

Please cite our work if you find it useful for your research and work.
```
  @ARTICLE{9036935, 
  author={G. {Andresini} and A. {Appice} and N. D. {Mauro} and C. {Loglisci} and D. {Malerba}}, 
  journal={IEEE Access}, 
  title={Multi-Channel Deep Feature Learning for Intrusion Detection}, 
  year={2020}, 
  volume={8}, 
  number={}, 
  pages={53346-53359},}
```



## Code requirements

The code relies on the following **python3.6+** libs.

Packages need are:
* [Tensorflow 1.13](https://www.tensorflow.org/) 
* [Keras 2.3](https://github.com/keras-team/keras) 
* [Matplotlib 2.2](https://matplotlib.org/)
* [Pandas 0.23.4](https://pandas.pydata.org/)
* [Numpy 1.15.4](https://www.numpy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)

## Data
The datasets used for experiments are accessible from [__DATASETS__](https://drive.google.com/open?id=1OIfsMv2PJljkc0aco00WB4_t8gEnXMiE). Original dataset is transformed in a binary classification: "_attack_, _normal_" (_oneCls files).
The repository contains the orginal dataset (folder: "original") and  the dataset after the preprocessing phase (folder: "numeric") 

Preprocessing phase is done mapping categorical feature and performing the Min Max scaler.

## How to use
Repository contains scripts of all experiments included in the paper:
* __main.py__ : script to run MINDFUL 
* __AblationExperiments.py__ : script to run ablation experiments (section C.2): 
* __Exp_Filters.py__ script to run experiments about filters (section C.3)
* __Imbalanced.py__ script to run experiments about imbalanced dataset (section C.4)
  
 Code contains models (autoencoder and classification) and datasets used for experiments in the work.
 
  

## Replicate the experiments

To replicate experiments reported in the work, you can use models and datasets stored in homonym folders.
Global variables are stored in __MINDFUL.conf__  file 


```python
    N_CLASSES = 2
    PREPROCESSING1 = 0  #if set to 1 code execute preprocessing phase on original date
    LOAD_AUTOENCODER_ADV = 1 #if 1 the autoencoder for attacks items  is loaded from models folder
    LOAD_AUTOENCODER_NORMAL = 1 #if 1 the autoencoder for normal items  is loaded from models folder
    LOAD_CNN = 1  #if 1 the classifier is loaded from models folder
    VALIDATION_SPLIT #the percentage of validation set used to train models
```

## Download datasets

[All datasets](https://drive.google.com/drive/folders/1OIfsMv2PJljkc0aco00WB4_t8gEnXMiE?usp=sharing)
