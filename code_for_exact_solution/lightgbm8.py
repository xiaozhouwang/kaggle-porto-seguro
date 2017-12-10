import numpy as np
import scipy as sp
from scipy import sparse as ssp
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr
import lightgbm as lgb
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from util import Gini
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from util import proj_num_on_cat
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from util import Gini
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from util import proj_num_on_cat, cat_count
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

cv_only = True
save_cv = True
full_train = False

import numpy as np
import scipy as sp
from scipy import sparse as ssp
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from util import proj_num_on_cat
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from scipy import sparse
from util import Gini, proj_num_on_cat, interaction_features, cat_count

NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)
learning_rate = 0.1
num_leaves = 15
min_data_in_leaf = 2000
# max_bin = x
feature_fraction = 0.6
num_boost_round = 10000

cv_only = True
save_cv = True
full_train = True

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', Gini(labels, preds), True

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

path = "../input/"
num_features_comb = []
import os
for p in os.listdir(path):
    if 'ps_reg_02___ps_car_07_cat' in p or 'ps_reg_01___ps_car_13___ps_car_15' in p:
        print(p)
        x,xt = pd.read_pickle(path+p)
        train[p] = x
        test[p] = xt
        num_features_comb.append(p)

num_features += num_features_comb

feature_names = list(train)
ind_features = [c for c in feature_names if 'ind' in c]
count = 0
for c in ind_features:
    if count == 0:
        train['new_ind'] = train[c].astype(str)
        count += 1
    else:
        train['new_ind'] += '_' + train[c].astype(str)

print(train['new_ind'].nunique())
ind_features = [c for c in feature_names if 'ind' in c]
count = 0
for c in ind_features:
    if count == 0:
        test['new_ind'] = test[c].astype(str)
        count += 1
    else:
        test['new_ind'] += '_' + test[c].astype(str)

reg_features = [c for c in feature_names if 'reg' in c]
count = 0
for c in reg_features:
    if count == 0:
        train['new_reg'] = train[c].astype(str)
        count += 1
    else:
        train['new_reg'] += '_' + train[c].astype(str)

print(train['new_reg'].nunique())
reg_features = [c for c in feature_names if 'reg' in c]
count = 0
for c in reg_features:
    if count == 0:
        test['new_reg'] = test[c].astype(str)
        count += 1
    else:
        test['new_reg'] += '_' + test[c].astype(str)

car_features = [c for c in feature_names if 'car' in c]
count = 0
for c in car_features:
    if count == 0:
        train['new_car'] = train[c].astype(str)
        count += 1
    else:
        train['new_car'] += '_' + train[c].astype(str)

print(train['new_car'].nunique())
car_features = [c for c in feature_names if 'car' in c]
count = 0
for c in car_features:
    if count == 0:
        test['new_car'] = test[c].astype(str)
        count += 1
    else:
        test['new_car'] += '_' + test[c].astype(str)

new_ps_reg_03 = pd.read_pickle(path + 'new_ps_reg_03.pkl')
train['ps_reg_03'] = new_ps_reg_03[:train.shape[0]]
test['ps_reg_03'] = new_ps_reg_03[train.shape[0]:]
print(train['ps_reg_03'].head(10))


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

cat_count_features = []
for c in cat_fea + ['new_ind','new_reg','new_car']:
    d = pd.concat([train[c],test[c]]).value_counts().to_dict()
    train['%s_count'%c] = train[c].apply(lambda x:d.get(x,0))
    test['%s_count'%c] = test[c].apply(lambda x:d.get(x,0))
    cat_count_features.append('%s_count'%c)


print(train_num.dtypes)
train_list = [train_num.replace([np.inf, -np.inf, np.nan], 0), train_ohe, train[cat_count_features]]#, np.ones(shape=(train_num.shape[0], 1))]
test_list = [test_num.replace([np.inf, -np.inf, np.nan], 0), test_ohe, test[cat_count_features]]#, np.ones(shape=(test_num.shape[0], 1))]

#proj
for t in ['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01']:
    for g in ['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01', 'ps_ind_05_cat']:
        if t != g:
            s_train, s_test = proj_num_on_cat(train, test, target_column=t, group_column=g)
            train_list.append(s_train)
            test_list.append(s_test)
X = sparse.hstack(train_list).tocsr()
X_test = sparse.hstack(test_list).tocsr()

params = {"objective": "binary",
          "boosting_type": "gbdt",
          "learning_rate": learning_rate,
          "num_leaves": int(num_leaves),
           "max_bin": 256,
          "feature_fraction": feature_fraction,
          "verbosity": 0,
          "drop_rate": 0.1,
          "is_unbalance": False,
          "max_drop": 50,
          "min_child_samples": 10,
          "min_child_weight": 150,
          "min_split_gain": 0,
          "subsample": 0.9
          }

x_score = []
final_cv_train = np.zeros(len(train_label))
final_cv_pred = np.zeros(len(test_id))
for s in xrange(8):
    cv_train = np.zeros(len(train_label))
    cv_pred = np.zeros(len(test_id))

    params['seed'] = s

    if cv_only:
        kf = kfold.split(X, train_label)

        best_trees = []
        fold_scores = []

        for i, (train_fold, validate) in enumerate(kf):
            X_train, X_validate, label_train, label_validate = \
                X[train_fold, :], X[validate, :], train_label[train_fold], train_label[validate]
            dtrain = lgbm.Dataset(X_train, label_train)
            dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)
            bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, feval=evalerror, verbose_eval=100,
                            early_stopping_rounds=100)
            best_trees.append(bst.best_iteration)
            cv_pred += bst.predict(X_test, num_iteration=bst.best_iteration)
            cv_train[validate] += bst.predict(X_validate)

            score = Gini(label_validate, cv_train[validate])
            print score
            fold_scores.append(score)

        cv_pred /= NFOLDS
        final_cv_train += cv_train
        final_cv_pred += cv_pred

        print("cv score:")
        print Gini(train_label, cv_train)
        print "current score:", Gini(train_label, final_cv_train / (s + 1.)), s+1
        print(fold_scores)
        print(best_trees, np.mean(best_trees))

        x_score.append(Gini(train_label, cv_train))

print(x_score)
pd.DataFrame({'id': test_id, 'target': final_cv_pred / 16.}).to_csv('../model/lgbm8_pred_avg.csv', index=False)
pd.DataFrame({'id': train_id, 'target': final_cv_train / 16.}).to_csv('../model/lgbm8_cv_avg.csv', index=False)

#cv score:
#0.287007087138
#current score: 0.289683837899 16
