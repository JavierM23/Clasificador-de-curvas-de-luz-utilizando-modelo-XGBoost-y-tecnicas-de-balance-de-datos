from hyper_selec import get_best_hyperparameter, get_best_optunaparameters
from preprocessing import get_data2, weights_compute, LabelEncoder, class_data
from custom_functions import xgb_train, pickle_save, pickle_load
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sampling_functions import get_strategy, f_undersampler, f_oversampler
from tqdm import tqdm


X_hierarchical, Y_hierarchical, Y_original  = get_data2()

subclass = input('subclass? {Hierarchical, Periodic, Stochastic, Transient}:\n')
#subclass = 'Periodic'
print(subclass)

method = input("Method? {weight, sampling}:\n")
#method = ''
print(method)

#metric_name = input('metrics? {recall, CBA}:\n')
metric_name = 'recall'

cluster = bool(int(input('cluster? {T -> 1, F -> 0}:\n')))
#cluster = True
print(cluster)

#optuna = bool(int(input('Optuna or Hyperopt? {O -> 1, H -> 0}:\n')))
optuna = True

if cluster:
    carpet = '/home/jmolina/carpet/to_transfer/'
else:
    carpet = 'pickle2/models/'

iteration = input('iteration?\n')
if iteration=='':
    iteration = 0
    model_dict = {}
else:
    iteration = int(iteration)-1
    load = bool(int(input('Load? {T -> 1, F -> 0}:\n')))
    if load:
        model_pickle = pickle_load(carpet + 'xgb_'+subclass.lower()+'_'+method+'_'+metric_name+'_model_CV_dict.pkl')
        model_dict = model_pickle['model_dict']
    else:
        model_dict = {}

n_splits = 10
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=5)
i=0

#Outer layer
for train_index, test_index in sss.split(X_hierarchical, Y_original):
    print('{}/{}'.format(i+1,n_splits))
    if i>=iteration:
        X_train_hierarchical, X_test_hierarchical = X_hierarchical.iloc[train_index], X_hierarchical.iloc[test_index]
        y_train_hierarchical, y_test_hierarchical = Y_hierarchical[train_index], Y_hierarchical[test_index]
        y_train_original, y_test_original = Y_original[train_index], Y_original[test_index]

        if method == 'weight':
            if subclass=='Hierarchical':
                weights_dict = weights_compute(LabelEncoder(Y_hierarchical.values))
                X_train = X_train_hierarchical
                y_train_enc = LabelEncoder(y_train_hierarchical)
                X_test = X_test_hierarchical
                y_test_enc = LabelEncoder(y_test_hierarchical)
            else:
                weights_dict = weights_compute(LabelEncoder(Y_original[Y_hierarchical==subclass].values))
                X_train, y_train_og, X_test, y_test_og = class_data(X_train_hierarchical, X_test_hierarchical, y_train_hierarchical,
                                         y_test_hierarchical, y_train_original, y_test_original, subclass)
                y_train_enc = LabelEncoder(y_train_og)
                y_test_enc = LabelEncoder(y_test_og)
            dtrain = xgb.DMatrix(data=X_train, label=pd.Series(y_train_enc), weight=pd.Series(y_train_enc).map(lambda x: weights_dict[x]))
        elif method == 'sampling':
            print('sampling')
            if subclass=='Hierarchical':
                X_train = X_train_hierarchical
                y_train_enc = LabelEncoder(y_train_hierarchical)
                X_test = X_test_hierarchical
                y_test_enc = LabelEncoder(y_test_hierarchical)
                over_strategy = get_strategy(y_train_enc, 2, kind='under')
                X_train, y_train_enc = f_oversampler(X_train, y_train_enc, fn='ros', strategy=over_strategy)
            else:
                rosrus_dict = {'Periodic': 2, 'Stochastic': 1, 'Transient': 1} #OJO periodic 1 o 2
                X_train, y_train_og, X_test, y_test_og = class_data(X_train_hierarchical, X_test_hierarchical, y_train_hierarchical,
                                         y_test_hierarchical, y_train_original, y_test_original, subclass)
                y_train_enc = LabelEncoder(y_train_og)
                y_test_enc = LabelEncoder(y_test_og)
                under_strategy = get_strategy(y_train_enc, rosrus_dict[subclass], kind='under')
                X_train_under, y_train_under = f_undersampler(X_train, y_train_enc, fn='rus', strategy=under_strategy)
                over_strategy = get_strategy(y_train_under, rosrus_dict[subclass], kind='over')
                X_train, y_train_enc = f_oversampler(X_train_under, y_train_under, fn='ros', strategy=over_strategy)
            dtrain = xgb.DMatrix(data=X_train, label=pd.Series(y_train_enc))
        elif method == '':
            if subclass=='Hierarchical':
                X_train = X_train_hierarchical
                y_train_enc = LabelEncoder(y_train_hierarchical)
                X_test = X_test_hierarchical
                y_test_enc = LabelEncoder(y_test_hierarchical)
            else:
                rosrus_dict = {'Periodic': 2, 'Stochastic': 1, 'Transient': 1} #OJO periodic 1 o 2
                X_train, y_train_og, X_test, y_test_og = class_data(X_train_hierarchical, X_test_hierarchical, y_train_hierarchical,
                                         y_test_hierarchical, y_train_original, y_test_original, subclass)
                y_train_enc = LabelEncoder(y_train_og)
                y_test_enc = LabelEncoder(y_test_og)
            dtrain = xgb.DMatrix(data=X_train, label=pd.Series(y_train_enc))

        dtest = xgb.DMatrix(data=X_test, label=pd.Series(y_test_enc))
        
        if subclass == 'Hierarchical': n_evals=8
        else: n_evals=10

        #Get best parameters
        if optuna:
            pbar = tqdm(total=n_evals, desc="Optuna")
            params = get_best_optunaparameters(dtrain, metric=metric_name, pbar=pbar, n_evals=n_evals)
        else:
            params = get_best_hyperparameter(dtrain, metric=metric_name, pbar=True, n_evals=n_evals)
    
        #XGBoost train
        xgb_model = xgb_train(params, dtrain, dtest, feval=metric_name, verbose_eval=False)
        print(xgb_model.predict(dtest).shape)

        #Store model in dict
        model_dict[i]=xgb_model

        #Save model
        pickle_dict = {'model_dict': model_dict}
        file_name = carpet + 'xgb_'+subclass.lower()+'_'+method+'_'+metric_name+'_model_CV_dict.pkl'
        pickle_save(file_name, pickle_dict)

    i+=1

print("_____All Good_____")



    

    
    









