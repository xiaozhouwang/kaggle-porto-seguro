from util import proj_num_on_cat, Gini, interaction_features
from itertools import combinations
import numpy as np
import pandas as pd

def Multiply_Divide(train, test, features):
    """
    combinations:
    combinations(['A', 'B','C'],2)  retrun AB AC BC
    combinations(range(4), 3) --> 012 013 023 123
    """
    feature_names= []
    for e, (x, y) in enumerate(combinations(features, 2)):
        train, test, feature_name= interaction_features(train, test, x, y, e)
        for name in feature_name:
            feature_names.append(name)


    return train, test, feature_names




def Series_string(train, test, category_list):
    '''
    produce series as a string like new_ind as the following
   id        new_ind_count                    new_ind
595207            117       3_1_10_0_0_0_0_0_1_0_0_0_0_0_13_1_0_0
595208            153       5_1_3_0_0_0_0_0_1_0_0_0_0_0_6_1_0_0

    return train and test with new colunes of new_categories 
    '''
    for category in category_list:

        feature_names = list(train.columns)

        features = [c for c in feature_names if category in c]
        name= 'new_'+ category


        count = 0
        for c in features:
            if count == 0:
                train[name] = train[c].astype(str)
                count += 1
            else:
                train[name] += '_' + train[c].astype(str)

        count = 0
        for c in features:
            if count == 0:
                test[name] = test[c].astype(str)
                count += 1
            else:
                test[name] += '_' + test[c].astype(str)

    return train, test



def Features_Counts(train, test, features):
    feature_names =[]

    for c in features:
        d = pd.concat([train[c],test[c]]).value_counts().to_dict()
        train['%s_count'%c] = train[c].apply(lambda x:d.get(x,0))
        test['%s_count'%c] = test[c].apply(lambda x:d.get(x,0))
        feature_names.append('%s_count'%c)

    return train, test, feature_names


def Statistic_features(train, test, target_features, group_features):
    train_list_=[]
    test_list_=[]
    for t in target_features:
        for g in group_features:
            if t != g:
                s_train, s_test = proj_num_on_cat(train, test, target_column=t, group_column=g)
                train_list_.append(s_train)
                test_list_.append(s_test)
    return np.hstack(train_list_), np.hstack(test_list_)

def features_type(train):
    data = []
    for f in train.columns:
        
        # Defining the level
        if 'bin' in f or f == 'target':
            level = 'binary'
        elif 'cat' in f or f == 'id':
            level = 'nominal'
        elif train[f].dtype == float:
            level = 'interval'
        elif train[f].dtype == int:
            level = 'ordinal'
            
        # Initialize keep to True for all variables except for id
        keep = True
        if f == 'id':
            keep = False
        
        # Defining the data type 
        dtype = train[f].dtype
        
        # Creating a Dict that contains all the metadata for the variable
        f_dict = {
            'varname': f,
            'level': level,
            'keep': keep,
            'dtype': dtype
        }
        data.append(f_dict)
    meta = pd.DataFrame(data, columns=['varname', 'level', 'keep', 'dtype'])
    meta.set_index('varname', inplace=True)
    interval = meta[(meta.level == 'interval') & (meta.keep)].index
    ordinal = meta[(meta.level == 'ordinal') & (meta.keep)].index
    binary = meta[(meta.level == 'binary') & (meta.keep)].index
    nominal  = meta[(meta.level == 'nominal') & (meta.keep)].index
    return interval, ordinal, binary, nominal




