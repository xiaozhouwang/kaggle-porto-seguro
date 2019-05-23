"""xgb prediction as features"""
import xgboost as xgb
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

eta = 0.1
max_depth = 6
subsample = 0.9
colsample_bytree = 0.85
min_child_weight = 55
num_boost_round = 500

train = pd.read_csv("../input/train.csv")
train_label = train['target']
train_id = train['id']
del train['target'], train['id']

test = pd.read_csv("../input/test.csv")
test_id = test['id']
del test['id']


params = {"objective": "reg:linear",
          "booster": "gbtree",
          "eta": eta,
          "max_depth": int(max_depth),
          "subsample": subsample,
          "colsample_bytree": colsample_bytree,
          "min_child_weight": min_child_weight,
          "silent": 1
          }

data = train.append(test)
data.reset_index(inplace=True)
train_rows = train.shape[0]

feature_results = []

for target_g in ['car', 'ind', 'reg']:
    features = [x for x in list(data) if target_g not in x]
    target_list = [x for x in list(data) if target_g in x]
    train_fea = np.array(data[features])
    for target in target_list:
        print(target)
        train_label = data[target]
        kfold = KFold(n_splits=5, random_state=218, shuffle=True)
        kf = kfold.split(data)
        cv_train = np.zeros(shape=(data.shape[0], 1))
        for i, (train_fold, validate) in enumerate(kf):
            X_train, X_validate, label_train, label_validate = \
                train_fea[train_fold, :], train_fea[validate, :], train_label[train_fold], train_label[validate]
            dtrain = xgb.DMatrix(X_train, label_train)
            dvalid = xgb.DMatrix(X_validate, label_validate)
            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
            bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, verbose_eval=50,
                            early_stopping_rounds=10)
            cv_train[validate, 0] += bst.predict(xgb.DMatrix(X_validate), ntree_limit=bst.best_ntree_limit)
        feature_results.append(cv_train)

feature_results = np.hstack(feature_results)
train_features = feature_results[:train_rows, :]
test_features = feature_results[train_rows:, :]

import pickle
#for python 2
pickle.dump([train_features, test_features], open("../input/fea0.pk", 'wb'),protocol=2)
#for python 3 
# pickle.dump([train_features, test_features], open("../input/fea0.pk", 'wb'),protocol=3)

