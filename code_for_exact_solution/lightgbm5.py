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


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', Gini(labels, preds), True

path = "../input/"

train = pd.read_csv(path+'train.csv')
train_label = train['target']
train_id = train['id']
test = pd.read_csv(path+'test.csv')
test_id = test['id']

NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)
learning_rate = 0.1
num_leaves = 15
min_data_in_leaf = 2000
# max_bin = x
feature_fraction = 0.6
num_boost_round = 10000

y = train['target'].values
drop_feature = [
    'id',
    'target'
]

X = train.drop(drop_feature,axis=1)
feature_names = X.columns.tolist()
cat_features = [c for c in feature_names if ('cat' in c and 'count' not in c)]
num_features = [c for c in feature_names if ('cat' not in c and 'calc' not in c)]

train['missing'] = (train==-1).sum(axis=1).astype(float)
test['missing'] = (test==-1).sum(axis=1).astype(float)
num_features.append('missing')

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
for c in cat_features:
    le = LabelEncoder()
    le.fit(train[c])
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])

from sklearn.preprocessing import normalize
from scipy.stats import spearmanr
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(train[cat_features])
X_cat = enc.transform(train[cat_features])
X_t_cat = enc.transform(test[cat_features])

ind_features = [c for c in feature_names if 'ind' in c]
count=0
for c in ind_features:
    if count==0:
        train['new_ind'] = train[c].astype(str)+'_'
        test['new_ind'] = test[c].astype(str)+'_'
        count+=1
    else:
        train['new_ind'] += train[c].astype(str)+'_'
        test['new_ind'] += test[c].astype(str)+'_'

cat_count_features = []
for c in cat_features+['new_ind']:
    d = pd.concat([train[c],test[c]]).value_counts().to_dict()
    train['%s_count'%c] = train[c].apply(lambda x:d.get(x,0))
    test['%s_count'%c] = test[c].apply(lambda x:d.get(x,0))
    cat_count_features.append('%s_count'%c)
    score = spearmanr(train['target'],train['%s_count'%c])
    print(c,score)

train_list = [train[num_features+cat_count_features].values,X_cat,]
test_list = [test[num_features+cat_count_features].values,X_t_cat,]

# missing binary projections
#missing_list = [x for x in list(train) if np.sum(train[x] == -1) > 0]
#for miss_fea in missing_list:
#    train['{}_miss_code'.format(miss_fea)] = (train[miss_fea] == -1).astype(int)
#    test['{}_miss_code'.format(miss_fea)] = (test[miss_fea] == -1).astype(int)

X = ssp.hstack(train_list).tocsr()
X_test = ssp.hstack(test_list).tocsr()


params = {"objective": "poisson",
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
for s in xrange(10):
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

pred_result = pd.DataFrame({'id': test_id, 'target': final_cv_pred / 10.})
pred_result['target'] = pred_result['target'].rank(pct=True)
pred_result.to_csv('../model/lgbm5_pred_avg.csv', index=False)

cv_result = pd.DataFrame({'id': train_id, 'target': final_cv_train / 10.})
cv_result['target'] = cv_result['target'].rank(pct=True)
cv_result.to_csv('../model/lgbm5_cv_avg.csv', index=False)

#cv score:
#0.287007087138
#current score: 0.289683837899 16
