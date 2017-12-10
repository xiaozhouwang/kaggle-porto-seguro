import pandas as pd
from util import Gini

def get_rank(x):
    return pd.Series(x).rank(pct=True).values

train = pd.read_csv("../input/train.csv", usecols = ['target'])

keras3_train = pd.read_csv("../model/keras3_cv.csv")
keras5_train = pd.read_csv("../model/keras5_cv.csv")
keras6_train = pd.read_csv("../model/keras6_cv.csv")
keras7_train = pd.read_csv("../model/keras7_cv.csv")

lgbm1_train = pd.read_csv("../model/lgbm1_cv_avg.csv")
lgbm3_train = pd.read_csv("../model/lgbm3_cv_avg.csv")
lgbm8_train = pd.read_csv("../model/lgbm8_cv_avg.csv")
lgbm5_train = pd.read_csv("../model/lgbm5_cv_avg.csv")
lgbm6_train = pd.read_csv("../model/lgbm6_cv_avg.csv")
lgbm7_train = pd.read_csv("../model/lgbm7_cv_avg.csv")


logistic1_train = pd.read_csv("../model/logistic1_cv.csv")

xgb0_train = pd.read_csv("../model/xgb0_cv.csv")

keras3_test = pd.read_csv("../model/keras3_pred.csv")
keras5_test = pd.read_csv("../model/keras5_pred.csv")
keras6_test = pd.read_csv("../model/keras6_pred.csv")
keras7_test = pd.read_csv("../model/keras7_pred.csv")


lgbm1_test = pd.read_csv("../model/lgbm1_pred_avg.csv")
lgbm3_test = pd.read_csv("../model/lgbm3_pred_avg.csv")
lgbm8_test = pd.read_csv("../model/lgbm8_pred_avg.csv")
lgbm5_test = pd.read_csv("../model/lgbm5_pred_avg.csv")
lgbm6_test = pd.read_csv("../model/lgbm6_pred_avg.csv")
lgbm7_test = pd.read_csv("../model/lgbm7_pred_avg.csv")


logistic1_test = pd.read_csv("../model/logistic1_pred.csv")

xgb0_test = pd.read_csv("../model/xgb0_pred.csv")

xgblinear_train = pd.read_csv("../model/xgb0l_cv.csv")
xgblinear_test = pd.read_csv("../model/xgb0l_pred.csv")


result = get_rank(keras5_train['target']) * 0.4 + get_rank(lgbm3_train['target']) * 0.5 + \
         get_rank(xgb0_train['target']) * 0.1 + get_rank(lgbm1_train['target']) * (-0.1) + \
         get_rank(keras3_train['target']) * 0.1 + get_rank(logistic1_train['target']) * 0.1 + \
         get_rank(xgblinear_train['target']) * 0.1 + get_rank(lgbm8_train['target']) * 0.25 + \
         get_rank(lgbm5_train['target']) * 0.1 + \
         get_rank(lgbm6_train['target']) * (-0.1) + get_rank(lgbm7_train['target']) * (0.1) + \
         get_rank(keras6_train['target']) * (-0.1) + \
         get_rank(keras7_train['target']) * 0.3

print "cv of final averaged model:", Gini(train['target'], result)

result = get_rank(keras5_test['target']) * 0.4 + get_rank(lgbm3_test['target']) * 0.5 + \
    get_rank(xgb0_test['target']) * 0.1 + get_rank(lgbm1_test['target']) * (-0.1) + \
    get_rank(keras3_test['target']) * 0.1 + get_rank(logistic1_test['target']) * 0.1 + \
    get_rank(xgblinear_test['target']) * 0.1 + get_rank(lgbm8_test['target']) * 0.25 + \
    get_rank(lgbm5_test['target']) * 0.1 + \
    get_rank(lgbm6_test['target']) * (-0.1) + get_rank(lgbm7_test['target']) * (0.1) + \
    get_rank(keras6_test['target']) * (-0.1) + \
    get_rank(keras7_test['target']) * 0.3

pd.DataFrame({'id': keras5_test['id'], 'target': get_rank(result)}).to_csv("../model/all_average.csv", index = False)