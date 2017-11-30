### Requirements

*older or newer version of below packages should theoretically work fine*

numpy 1.13.3

pandas 0.20.3

sklearn 0.19.1

keras 2.1.1

tensorflow 1.4.0

xgboost 0.6

lightgbm 2.0.10


### How to reproduce

Put unzipped data in `input`

`python nn_model290.py` to get a nn model that scores 0.290X

`python gbm_model291.py` to get a gbm model that scores 0.291X

*simple average of the two gives about 0.2939 on private LB, which is good enough for 2nd place*

