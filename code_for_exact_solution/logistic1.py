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

train_copy = train.copy()
test_copy = test.copy()
train_copy = train_copy.replace(-1, np.NaN)
test_copy = test_copy.replace(-1, np.NaN)
train['num_na'] = train_copy.isnull().sum(axis=1)
test['num_na'] = test_copy.isnull().sum(axis=1)
del train_copy, test_copy

cat_fea = [x for x in list(train) if 'cat' in x]
bin_fea = [x for x in list(train) if 'bin' in x]
train_cat_count, test_cat_count = cat_count(train, test, cat_fea)
# include interactions
for e, (x, y) in enumerate(combinations(['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01'], 2)):
    train, test = interaction_features(train, test, x, y, e)


inter_fea = [x for x in list(train) if 'inter' in x]
#train['cat_sum'] = train[cat_fea].sum(axis=1)
#test['cat_sum'] = test[cat_fea].sum(axis=1)


#X = train.as_matrix()
#X_test = test.as_matrix()
#print(X.shape, X_test.shape)
#ohe
ohe = OneHotEncoder(sparse=True)

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

train_fea0, test_fea0 = pickle.load(open("../input/fea0.pk"))


train_list = [train_num.replace([np.inf, -np.inf, np.nan], 0), train_ohe, train_fea0, train_cat_count]#, np.ones(shape=(train_num.shape[0], 1))]
test_list = [test_num.replace([np.inf, -np.inf, np.nan], 0), test_ohe, test_fea0, test_cat_count]#, np.ones(shape=(test_num.shape[0], 1))]

#proj
for t in ['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01']:
    for g in ['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01', 'ps_ind_05_cat']:
        if t != g:
            s_train, s_test = proj_num_on_cat(train, test, target_column=t, group_column=g)
            train_list.append(s_train)
            test_list.append(s_test)

X = sparse.hstack(train_list).tocsr()
X_test = sparse.hstack(test_list).tocsr()
#X = train_num
#X_test = test_num
#all_data = np.vstack([X, X_test])

selector = SelectPercentile(f_classif, 75)
selector.fit(X.toarray(), train_label)
X, X_test = selector.transform(X.toarray()), selector.transform(X_test.toarray())

all_data = np.vstack([X, X_test])
scaler = StandardScaler()
scaler.fit(all_data)

X = csr_matrix(scaler.transform(X))
X_test = csr_matrix(scaler.transform(X_test))
#X = scaler.transform(X)
#X_test = scaler.transform(X_test)

print(X.shape, X_test.shape)

kf = kfold.split(X, train_label)
cv_train = np.zeros(len(train_label))
cv_pred = np.zeros(len(test_id))
best_trees = []
fold_scores = []

if cv_only:
    for i, (train_fold, validate) in enumerate(kf):
        X_train, X_validate, label_train, label_validate = \
            X[train_fold, :], X[validate, :], train_label[train_fold], train_label[validate]

        #selector = SelectPercentile(f_classif, X_train.toarray(), label_train)
        #X_train, X_validate = csr_matrix(selector.transform(X_train.toarray())), csr_matrix(selector.transform(
        #    X_validate.toarray()
        #))

        clf = LR(C=25.)
        clf.fit(X_train, label_train)
        cv_pred += clf.predict_proba(X_test)[:, 1]
        cv_train[validate] += clf.predict_proba(X_validate)[:, 1]
        score = Gini(label_validate, cv_train[validate])
        print score
        fold_scores.append(score)

    print("cv score:")
    print Gini(train_label, cv_train)
    print(fold_scores)

    if save_cv:
        pd.DataFrame({'id': test_id, 'target': cv_pred/NFOLDS}).to_csv('../model/logistic1_pred.csv', index=False)
        pd.DataFrame({'id': train_id, 'target': cv_train}).to_csv('../model/logistic1_cv.csv', index=False)

#cv score:
#0.27906438918
#[0.28773918373071583, 0.26723600995806762, 0.28158062789785737, 0.27506435916773242, 0.28394519465855006]

if full_train:
    clf = LR(C=25.)
    clf.fit(X, train_label)
    pred = clf.predict_proba(X_test)[:, 1]
    pd.DataFrame({'id': test_id, 'target': pred}).to_csv('../model/logistic1_full_pred.csv', index=False)


