from keras.layers import Dense, Dropout, Embedding, Flatten, Input, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from time import time
import datetime
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from util import Gini, interaction_features
from itertools import combinations
from util import proj_num_on_cat
from scipy import sparse
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.preprocessing import LabelEncoder

cv_only = True
save_cv = True

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

feature_names = list(train)
ind_features = [c for c in feature_names if 'ind' in c]
count = 0
for c in ind_features:
    if count == 0:
        train['new_ind'] = train[c].astype(str)
        count += 1
    else:
        train['new_ind'] += '_' + train[c].astype(str)

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

car_features = [c for c in feature_names if 'car' in c]
count = 0
for c in car_features:
    if count == 0:
        test['new_car'] = test[c].astype(str)
        count += 1
    else:
        test['new_car'] += '_' + test[c].astype(str)

train_cat = train[cat_fea]
train_num = train[[x for x in list(train) if x in num_features]]
test_cat = test[cat_fea]
test_num = test[[x for x in list(train) if x in num_features]]

max_cat_values = []
for c in cat_fea:
    le = LabelEncoder()
    x = le.fit_transform(pd.concat([train_cat, test_cat])[c])
    train_cat[c] = le.transform(train_cat[c])
    test_cat[c] = le.transform(test_cat[c])
    max_cat_values.append(np.max(x))

# xgboost prediction
train_fea0, test_fea0 = pickle.load(open("../input/fea0.pk"))

cat_count_features = []
for c in cat_fea + ['new_ind','new_reg','new_car']:
    d = pd.concat([train[c],test[c]]).value_counts().to_dict()
    train['%s_count'%c] = train[c].apply(lambda x:d.get(x,0))
    test['%s_count'%c] = test[c].apply(lambda x:d.get(x,0))
    cat_count_features.append('%s_count'%c)


print(train_num.dtypes)
train_list = [train_num.replace([np.inf, -np.inf, np.nan], 0), train[cat_count_features], train_fea0]
test_list = [test_num.replace([np.inf, -np.inf, np.nan], 0), test[cat_count_features], test_fea0]

#feature aggregation
for t in ['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01']:
    for g in ['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01', 'ps_ind_05_cat']:
        if t != g:
            s_train, s_test = proj_num_on_cat(train, test, target_column=t, group_column=g)
            train_list.append(s_train)
            test_list.append(s_test)
X = sparse.hstack(train_list).tocsr()
X_test = sparse.hstack(test_list).tocsr()

all_data = np.vstack([X.toarray(), X_test.toarray()])
scaler = StandardScaler()
scaler.fit(all_data)
X = scaler.transform(X.toarray())
X_test = scaler.transform(X_test.toarray())
print(X.shape, X_test.shape)


cv_train = np.zeros(len(train_label))
cv_pred = np.zeros(len(test_id))

X_cat = train_cat.as_matrix()
X_test_cat = test_cat.as_matrix()

x_test_cat = []
for i in xrange(X_test_cat.shape[1]):
    x_test_cat.append(X_test_cat[:, i].reshape(-1, 1))
x_test_cat.append(X_test)

def nn_model():
    inputs = []
    flatten_layers = []
    for e, c in enumerate(cat_fea):
        input_c = Input(shape=(1, ), dtype='int32')
        num_c = max_cat_values[e]
        embed_c = Embedding(
            num_c,
            6,
            input_length=1
        )(input_c)
        embed_c = Dropout(0.25)(embed_c)
        flatten_c = Flatten()(embed_c)

        inputs.append(input_c)
        flatten_layers.append(flatten_c)

    input_num = Input(shape=(X.shape[1],), dtype='float32')
    flatten_layers.append(input_num)
    inputs.append(input_num)

    flatten = merge(flatten_layers, mode='concat')

    fc1 = Dense(512, init='he_normal')(flatten)
    fc1 = PReLU()(fc1)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(0.75)(fc1)

    fc1 = Dense(64, init='he_normal')(fc1)
    fc1 = PReLU()(fc1)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(0.5)(fc1)

    outputs = Dense(1, init='he_normal', activation='sigmoid')(fc1)

    model = Model(input = inputs, output = outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return (model)

num_seeds = 5
begintime = time()
if cv_only:
    for s in xrange(num_seeds):
        np.random.seed(s)
        for (inTr, inTe) in kfold.split(X, train_label):
            xtr = X[inTr]
            ytr = train_label[inTr]
            xte = X[inTe]
            yte = train_label[inTe]

            xtr_cat = X_cat[inTr]
            xte_cat = X_cat[inTe]

            # get xtr xte cat
            xtr_cat_list, xte_cat_list = [], []
            for i in xrange(xtr_cat.shape[1]):
                xtr_cat_list.append(xtr_cat[:, i].reshape(-1, 1))
                xte_cat_list.append(xte_cat[:, i].reshape(-1, 1))

            xtr_cat_list.append(xtr)
            xte_cat_list.append(xte)

            model = nn_model()
            def get_rank(x):
                return pd.Series(x).rank(pct=True).values
            model.fit(xtr_cat_list, ytr, epochs=20, batch_size=512, verbose=2, validation_data=[xte_cat_list, yte])
            cv_train[inTe] += get_rank(model.predict(x=xte_cat_list, batch_size=512, verbose=0)[:, 0])
            print(Gini(train_label[inTe], cv_train[inTe]))
            cv_pred += get_rank(model.predict(x=x_test_cat, batch_size=512, verbose=0)[:, 0])
        print(s)
        print(Gini(train_label, cv_train / (1. * (s + 1))))
        print(str(datetime.timedelta(seconds=time() - begintime)))
    if save_cv:
        pd.DataFrame({'id': test_id, 'target': get_rank(cv_pred * 1./ (NFOLDS * num_seeds))}).to_csv('../model/keras5_pred.csv', index=False)
        pd.DataFrame({'id': train_id, 'target': get_rank(cv_train * 1. / num_seeds)}).to_csv('../model/keras5_cv.csv', index=False)

