import numpy as np
import pandas as pd

def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred * 1. / G_true


def cat_count(train_df, test_df, cat_list):
    train_df['row_id'] = range(train_df.shape[0])
    test_df['row_id'] = range(test_df.shape[0])
    train_df['train'] = 1
    test_df['train'] = 0
    all_df = train_df[['row_id', 'train'] + cat_list].append(test_df[['row_id','train'] + cat_list])
    for e, cat in enumerate(cat_list):
        grouped = all_df[[cat]].groupby(cat)
        the_size = pd.DataFrame(grouped.size()).reset_index()
        the_size.columns = [cat, '{}_size'.format(cat)]
        all_df = pd.merge(all_df, the_size, how='left')

        selected_train = all_df[all_df['train'] == 1]
        selected_test = all_df[all_df['train'] == 0]
        selected_train.sort_values('row_id', inplace=True)
        selected_test.sort_values('row_id', inplace=True)
        selected_train.drop(['row_id', 'train'] + cat_list, axis=1, inplace=True)
        selected_test.drop(['row_id', 'train'] + cat_list, axis=1, inplace=True)

        selected_train, selected_test = np.array(selected_train), np.array(selected_test)
    print(selected_train.shape, selected_test.shape)
    return selected_train, selected_test


def proj_num_on_cat(train_df, test_df, target_column, group_column):
    """
    :param train_df: train data frame
    :param test_df:  test data frame
    :param target_column: name of numerical feature
    :param group_column: name of categorical feature
    """
    train_df['row_id'] = range(train_df.shape[0])  # 595211 create index for each row
    
    test_df['row_id'] = range(test_df.shape[0])
    train_df['train'] = 1
    test_df['train'] = 0
    
    all_df = train_df[['row_id', 'train', target_column, group_column]].append(test_df[['row_id','train',
                                                                                        target_column, group_column]]).copy()
    
    
    #count the number
    grouped = all_df[[target_column, group_column]].groupby(group_column)
    
    
    #count the number of distint value  from the list [1,1, 2,3]   
    #[1,2,3]  so answer is  3 
    
    #count the number of each distint value  [1,1,2,3]  
    #1:2 
    #2:1 
    #3:1
    
    
    #count the number of each distint value
    the_size = pd.DataFrame(grouped.size()).reset_index()
    the_size.columns = [group_column, '%s_size' % target_column]  #rename columns name

    #find the mean, std, median, max, min of each distint value
    the_mean = pd.DataFrame(grouped.mean()).reset_index()
    the_mean.columns = [group_column, '%s_mean' % target_column] #rename columns name
    the_std = pd.DataFrame(grouped.std()).reset_index().fillna(0)
    the_std.columns = [group_column, '%s_std' % target_column]
    the_median = pd.DataFrame(grouped.median()).reset_index()
    the_median.columns = [group_column, '%s_median' % target_column]
    the_max = pd.DataFrame(grouped.max()).reset_index()
    the_max.columns = [group_column, '%s_max' % target_column]
    the_min = pd.DataFrame(grouped.min()).reset_index()
    the_min.columns = [group_column, '%s_min' % target_column]
    
    #merge them 
    the_stats=pd.concat([the_size,the_mean.iloc[:,1],the_std.iloc[:,1]
                         ,the_median.iloc[:,1] ,the_max.iloc[:,1],the_min.iloc[:,1]]
                        ,axis=1, join_axes=[the_size.index])

    #insert value to the original data
    all_df = pd.merge(all_df, the_stats, how='left')

    #splite to train and test
    selected_train = all_df[all_df['train'] == 1].copy()
    selected_test = all_df[all_df['train'] == 0].copy()
    selected_train.sort_values('row_id', inplace=True)
    selected_test.sort_values('row_id', inplace=True)
    
    #remove target_column, group_column, 'row_id', 'train' columns
    selected_train.drop([target_column, group_column, 'row_id', 'train'], axis=1, inplace=True)
    selected_test.drop([target_column, group_column, 'row_id', 'train'], axis=1, inplace=True)

    selected_train, selected_test = np.array(selected_train), np.array(selected_test)
    return selected_train, selected_test


def interaction_features(train, test, fea1, fea2, prefix):
    train['inter_{}*'.format(prefix)] = train[fea1] * train[fea2]
    train['inter_{}/'.format(prefix)] = train[fea1] / train[fea2]

    test['inter_{}*'.format(prefix)] = test[fea1] * test[fea2]
    test['inter_{}/'.format(prefix)] = test[fea1] / test[fea2]
    feature_name = ['inter_{}*'.format(prefix), 'inter_{}/'.format(prefix), ]

    return train, test, feature_name
