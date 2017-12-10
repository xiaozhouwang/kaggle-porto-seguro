import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from time import time
import datetime
from util import Gini, proj_num_on_cat, interaction_features, cat_count
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import pickle

'''
simple xgboost benchmark
'''
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import SelectPercentile, f_classif
import numpy as np
import pandas as pd
from util import Gini, proj_num_on_cat, interaction_features, cat_count
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from util import proj_num_on_cat
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from scipy import sparse
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import pickle
cv_only = True
save_cv = True
full_train = True

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', Gini(labels, preds)

NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)

train = pd.read_csv("../input/train.csv")
train_label = train['target']
train_id = train['id']
del train['target'], train['id']

test = pd.read_csv("../input/test.csv")
test_id = test['id']
del test['id']

cat_fea = [x for x in list(train) if 'cat' in x]
bin_fea = [x for x in list(train) if 'bin' in x]

train['missing'] = (train==-1).sum(axis=1).astype(float)
test['missing'] = (test==-1).sum(axis=1).astype(float)

# include interactions
for e, (x, y) in enumerate(combinations(['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01'], 2)):
    train, test = interaction_features(train, test, x, y, e)

num_features = [c for c in list(train) if ('cat' not in c and 'calc' not in c)]
num_features.append('missing')
inter_fea = [x for x in list(train) if 'inter' in x]
#train['cat_sum'] = train[cat_fea].sum(axis=1)
#test['cat_sum'] = test[cat_fea].sum(axis=1)


#X = train.as_matrix()
#X_test = test.as_matrix()
#print(X.shape, X_test.shape)
#ohe
ohe = OneHotEncoder(sparse=True)

train_cat = train[cat_fea].as_matrix()
train_num = train[[x for x in list(train) if x in num_features]]
test_cat = test[cat_fea].as_matrix()
test_num = test[[x for x in list(train) if x in num_features]]
train_cat[train_cat < 0] = 99
test_cat[test_cat < 0] = 99

traintest = np.vstack((train_cat, test_cat))
traintest = pd.DataFrame(traintest, columns=cat_fea)
print(traintest.shape)
#encoder = ce.HelmertEncoder(cols=cat_fea)
#encoder.fit(traintest)
#train_enc = encoder.transform(pd.DataFrame(train_cat, columns=cat_fea))
#test_enc = encoder.transform(pd.DataFrame(test_cat, columns=cat_fea))
ohe.fit(traintest)
train_ohe = ohe.transform(train_cat)
test_ohe = ohe.transform(test_cat)
del traintest

train_fea0, test_fea0 = pickle.load(open("../input/fea0.pk"))
train_fea1, test_fea1 = pickle.load(open("../input/fea0_lgb.pk"))

cat_count_features = []
for c in cat_fea:
    d = pd.concat([train[c],test[c]]).value_counts().to_dict()
    train['%s_count'%c] = train[c].apply(lambda x:d.get(x,0))
    test['%s_count'%c] = test[c].apply(lambda x:d.get(x,0))
    cat_count_features.append('%s_count'%c)

train_list = [train_num.replace([np.inf, -np.inf, np.nan], 0), train_ohe, train[cat_count_features]]#, np.ones(shape=(train_num.shape[0], 1))]
test_list = [test_num.replace([np.inf, -np.inf, np.nan], 0), test_ohe, test[cat_count_features]]#, np.ones(shape=(test_num.shape[0], 1))]


t_fea = ['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01']
g_fea = ['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01', 'ps_ind_05_cat'] + cat_fea

t_fea = list(set(t_fea))
g_fea = list(set(g_fea))

#proj
for t in t_fea:
    for g in g_fea:
        if t != g:
            s_train, s_test = proj_num_on_cat(train, test, target_column=t, group_column=g)
            train_list.append(s_train)
            test_list.append(s_test)
X = sparse.hstack(train_list).tocsr()
X_test = sparse.hstack(test_list).tocsr()
#X = train_num
#X_test = test_num
all_data = np.vstack([X.toarray(), X_test.toarray()])
#all_data = np.vstack([X, X_test])

scaler = StandardScaler()
scaler.fit(all_data)
X = scaler.transform(X.toarray())
X_test = scaler.transform(X_test.toarray())
#X = scaler.transform(X)
#X_test = scaler.transform(X_test)

print(X.shape, X_test.shape)

final_cv_train = np.zeros(len(train_label))
final_cv_pred = np.zeros(len(test_id))
final_best_trees = []

eta = 0.1
lamb = 0.25
alpha = 1
num_boost_round = 10000

params = {"objective": "binary:logistic",
          "booster": "gbtree",
          "eta": eta,
          "lamb": lamb,
          "alpha": alpha,
          "silent": 1
          }

if cv_only:
    num_seeds = 3
    for s in xrange(num_seeds):
        print(s)
        params['seed'] = s
        kf = kfold.split(X, train_label)
        cv_train = np.zeros(len(train_label))
        cv_pred = np.zeros(len(test_id))
        best_trees = []
        fold_scores = []

        for i, (train_fold, validate) in enumerate(kf):
            X_train, X_validate, label_train, label_validate = \
                X[train_fold, :], X[validate, :], train_label[train_fold], train_label[validate]
            dtrain = xgb.DMatrix(X_train, label_train)
            dvalid = xgb.DMatrix(X_validate, label_validate)
            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
            bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, feval=evalerror, verbose_eval=100,
                            early_stopping_rounds=50, maximize=True)
            best_trees.append(bst.best_iteration)
            cv_pred += bst.predict(xgb.DMatrix(X_test))
            cv_train[validate] += bst.predict(xgb.DMatrix(X_validate), ntree_limit=bst.best_ntree_limit)
            score = Gini(label_validate, cv_train[validate])
            print score
            fold_scores.append(score)

        final_cv_train += cv_train
        final_cv_pred += cv_pred
        final_best_trees += best_trees
        print("cv score:")
        print Gini(train_label, cv_train)
        print(fold_scores)
        print(best_trees, np.mean(best_trees))
        print("current score:", Gini(train_label, final_cv_train * 1. / (s + 1)), s+1)


    final_cv_pred /= (NFOLDS * num_seeds)
    final_cv_train /= num_seeds
    pd.DataFrame({'id': test_id, 'target': final_cv_pred}).to_csv('../model/xgb0l_pred.csv', index=False)
    pd.DataFrame({'id': train_id, 'target': final_cv_train}).to_csv('../model/xgb0l_cv.csv', index=False)
    print(np.mean(final_best_trees), np.median(final_best_trees), np.std(final_best_trees))