### Requirements

*older or newer version of below packages should theoretically work fine*

python 2.7

numpy 1.13.3

pandas 0.20.3

sklearn 0.19.1

keras 2.1.1

tensorflow 1.4.0

xgboost 0.6

lightgbm 2.0.10


### How to reproduce

#### Simple solution (recommended)
Put unzipped data in `input`

*Generate a simple solution that is good enough for 2nd place (~0.2938 on private LB)*

`cd code`

`python fea_eng0.py`

`python nn_model290.py` to get a nn model that scores 0.290X

`python gbm_model291.py` to get a gbm model that scores 0.291X

`python simple_average.py` and then you can find the submission file at `../model/simple_average.csv`

You can reproduce this solution in a few hours.

#### Exact solution (Optional)

Although not recommended but you can also reproduce the exact same solution we submitted (0.29413 on private LB).

*you can follow these steps below, in addition to the simple solution*

```
cd ../code_for_exact_solution/
python keras3.py
python keras6.py
python keras7.py
python lightgbm1.py
python lightgbm5.py
python lightgbm6.py
python lightgbm7.py
python lightgbm8.py
python logistic1.py
python xgb0.py
python xgb_linear0.py
python rank_average.py
```

*It can take up to 2 days to generate the exact solution which only has 0.0003 improvement over the simple one*