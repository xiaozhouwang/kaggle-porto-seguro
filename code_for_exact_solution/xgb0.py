'''
simple xgboost benchmark
'''
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import pandas as pd
from util import Gini
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from util import proj_num_on_cat
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from scipy import sparse

cv_only = True
save_cv = True
full_train = False

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', Gini(labels, preds)

NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)
#kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=0)
eta = 0.05
max_depth = 7
subsample = 0.97
colsample_bytree = 0.85
gamma = 0.05
alpha = 0
min_child_weight = 55
#lamb = 0.35
colsample_bylevel = 0.8
num_boost_round = 10000

train = pd.read_csv("../input/train.csv")
train_label = train['target']
train_id = train['id']
del train['target'], train['id']

test = pd.read_csv("../input/test.csv")
test_id = test['id']
del test['id']

train_copy = train.copy()
test_copy = test.copy()
train_copy = train_copy.replace(-1, np.NaN)
test_copy = test_copy.replace(-1, np.NaN)
train['num_na'] = train_copy.isnull().sum(axis=1)
test['num_na'] = test_copy.isnull().sum(axis=1)
del train_copy, test_copy

cat_fea = [x for x in list(train) if 'cat' in x]
bin_fea = [x for x in list(train) if 'bin' in x]

#train['cat_sum'] = train[cat_fea].sum(axis=1)
#test['cat_sum'] = test[cat_fea].sum(axis=1)


#X = train.as_matrix()
#X_test = test.as_matrix()
#print(X.shape, X_test.shape)
#ohe
ohe = OneHotEncoder(sparse=True)

cat_fea = [x for x in list(train) if 'cat' in x]
train_cat = train[cat_fea].as_matrix()
train_num = train[[x for x in list(train) if x not in cat_fea]]
test_cat = test[cat_fea].as_matrix()
test_num = test[[x for x in list(train) if x not in cat_fea]]
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

train_list = [train_num, train_ohe]#, np.ones(shape=(train_num.shape[0], 1))]
test_list = [test_num, test_ohe]#, np.ones(shape=(test_num.shape[0], 1))]

X = sparse.hstack(train_list).tocsr()
X_test = sparse.hstack(test_list).tocsr()
#X, X_test = X.toarray(), X_test.toarray()
print(X.shape, X_test.shape)

final_cv_train = np.zeros(len(train_label))
final_cv_pred = np.zeros(len(test_id))
final_best_trees = []

params = {"objective": "binary:logistic",
          "booster": "gbtree",
          "eta": eta,
          "max_depth": int(max_depth),
          "subsample": subsample,
          "colsample_bytree": colsample_bytree,
          "gamma": gamma,
          #"lamb": lamb,
          "alpha": alpha,
          "min_child_weight": min_child_weight,
          "colsample_bylevel": colsample_bylevel,
          "silent": 1
          }

if cv_only:
    num_seeds = 24
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
            bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, feval=evalerror, verbose_eval=800,
                            early_stopping_rounds=25, maximize=True)
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
    pd.DataFrame({'id': test_id, 'target': final_cv_pred}).to_csv('../model/xgb_avg16_pred.csv', index=False)
    pd.DataFrame({'id': train_id, 'target': final_cv_train}).to_csv('../model/xgb_avg16_cv.csv', index=False)
    print(np.mean(final_best_trees), np.median(final_best_trees), np.std(final_best_trees))

    ## 0.1
    #0.281739276885
    #[0.28693135981084533, 0.26989064676756958, 0.28035898856108521, 0.28178381987103512, 0.29021910168396381]
    #([123, 91, 139, 97, 92], 108.40000000000001)


    #0.284350552387
    #([1057, 933, 1175, 979, 1168], 1062.4000000000001)

if full_train:
    for s in xrange(32):
        params['seed'] = s
        dtrain = xgb.DMatrix(X, train_label)
        watchlist = [(dtrain, 'train')]
        bst = xgb.train(params, dtrain, 100, evals=watchlist, feval=evalerror, verbose_eval=50, maximize=True)
        pred = bst.predict(xgb.DMatrix(X_test))
        if s == 0:
            final_pred = pred
        else:
            final_pred += pred
    pd.DataFrame({'id': test_id, 'target': final_pred / 32.}).to_csv('../model/xgb_avg_full_pred.csv', index=False)


