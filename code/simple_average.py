'''
simple average of two models to get 2nd place
'''
import pandas as pd
keras5_test = pd.read_csv("../model/keras5_pred.csv")
lgbm3_test = pd.read_csv("../model/lgbm3_pred_avg.csv")

def get_rank(x):
    return pd.Series(x).rank(pct=True).values

pd.DataFrame({'id': keras5_test['id'], 'target':
    get_rank(keras5_test['target']) * 0.5 + get_rank(keras5_test['target']) * 0.5}).to_csv(
    "../model/simple_average.csv", index = False)