from functions.hyper_selec import get_best_optunaparameters
from functions.preprocessing import get_data2, LabelEncoder, class_data
from functions.custom_functions import pickle_load, xgb_train, pickle_save
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from multi_imbalance.resampling.mdo import MDO
from multi_imbalance.resampling.soup import SOUP
from multi_imbalance.resampling.spider import SPIDER3


X_hierarchical, Y_hierarchical, Y_original = get_data2()

subclass = input(
    'subclass? {Hierarchical, Periodic, Stochastic, Transient}:\n')
print(subclass)

method = input('method? [mdo, soup, spider]\n')

metric_name = 'recall'

cluster = bool(int(input('cluster? {T -> 1, F -> 0}:\n')))
print(cluster)

if cluster:
    load_carpet = '/home/jmolina/carpet/to_transfer/'
    save_carpet = '/home/jmolina/carpet/to_transfer/pickle/models/'
    parameters_carpet = '/home/jmolina/carpet/to_transfer/pickle/parameters/'
else:
    load_carpet = 'pickle2/models/'
    save_carpet = 'pickle2/models/multi_imb/'
    parameters_carpet = 'pickle2/parameters/'

iteration = input('iteration?\n')
if iteration == '':
    iteration = 0
    model_dict = {}
else:
    iteration = int(iteration)-1
    load = bool(int(input('Load?: \n')))
    if load:
        model_pickle = pickle_load(save_carpet + 'xgb_'+subclass.lower()+'_'+method+'_'+metric_name+'_model_CV_dict.pkl')
        model_dict = model_pickle['model_dict']
    else:
        model_dict = {}
    
n_splits = 10
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=5)
i = 0

for train_index, test_index in sss.split(X_hierarchical, Y_original):
    print('\n')
    print('{}/{}'.format(i+1, n_splits))

    if i >= iteration:
        # Shuffle data
        X_train_hierarchical, X_test_hierarchical = X_hierarchical.iloc[
            train_index], X_hierarchical.iloc[test_index]
        y_train_hierarchical, y_test_hierarchical = Y_hierarchical[
            train_index], Y_hierarchical[test_index]
        y_train_original, y_test_original = Y_original[train_index], Y_original[test_index]

        # Classsifier data
        if subclass == 'Hierarchical':
            X_train = X_train_hierarchical
            y_train_enc = LabelEncoder(y_train_hierarchical)
            X_test = X_test_hierarchical
            y_test_enc = LabelEncoder(y_test_hierarchical)
        else:
            X_train, y_train_og, X_test, y_test_og = class_data(X_train_hierarchical, X_test_hierarchical, y_train_hierarchical,
                                                                y_test_hierarchical, y_train_original, y_test_original, subclass)
            y_train_enc = LabelEncoder(y_train_og)
            y_test_enc = LabelEncoder(y_test_og)

        subclass_method_params = {'Periodic': {'k': 5, 'maj_int_min': {'maj': [0, 1, 2], 'int': [], 'min': [3, 4, 5]}, 'prop': 0.9},
                                  'Stochastic': {'k': 5, 'maj_int_min': {'maj': [0], 'int': [], 'min': [1, 2, 3, 4]}, 'prop': 0.9},
                                  'Transient': {'k': 3, 'maj_int_min': {'maj': [0], 'int': [], 'min': [1, 2, 3]}, 'prop': 0.9}}

        method_params = subclass_method_params[subclass]
        sampler_dict = {'mdo': MDO(k=method_params['k'], maj_int_min=method_params['maj_int_min'], prop=method_params['prop']),
                        'soup': SOUP(k=method_params['k'], maj_int_min=method_params['maj_int_min']),
                        'spider': SPIDER3(k=method_params['k'], majority_classes=method_params['maj_int_min']['maj'], intermediate_classes=method_params['maj_int_min']['int'], minority_classes=method_params['maj_int_min']['min'], cost=None)}
        print('Fitting hybrid method')
        X_train_over, y_train_over = sampler_dict[method].fit_transform(X_train.values, np.array(y_train_enc))
        X_train_over_df = pd.DataFrame(data=X_train_over, columns=X_train.columns)

        dtrain = xgb.DMatrix(data=X_train_over_df, label=pd.Series(y_train_over))
        dtest = xgb.DMatrix(data=X_test, label=pd.Series(y_test_enc))

        if subclass == 'Hierarchical':
            n_evals = 8
        else:
            n_evals = 12

        # Get best parameters
        pbar = tqdm(total=n_evals, desc="Optuna")
        xgb_params = get_best_optunaparameters(
            dtrain, metric=metric_name, pbar=pbar, n_evals=n_evals)

        # XGBoost train
        print('Training XGB')
        xgb_model = xgb_train(xgb_params, dtrain, dtest,
                              feval=metric_name, verbose_eval=False)

        # Store model in dict
        model_dict[i] = xgb_model

        # Save model and params
        pickle_dict = {'model_dict': model_dict}
        file_name = save_carpet + 'xgb_'+subclass.lower()+'_'+method+'_'+metric_name + \
            '_model_CV_dict.pkl'
        pickle_save(file_name, pickle_dict)

    i += 1

print("_____All Good_____")
