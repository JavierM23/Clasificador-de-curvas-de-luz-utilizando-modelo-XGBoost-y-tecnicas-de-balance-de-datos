from functions.hyper_selec import get_best_hyperparameter_gamma, get_best_hyperparameter_gamma_fixed
from functions.preprocessing import get_data2, weights_compute, LabelEncoder, class_data
from functions.custom_functions import xgb_train, pickle_save, pickle_load
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np



X_hierarchical, Y_hierarchical, Y_original  = get_data2()

subclass = input('subclass? {Hierarchical, Periodic, Stochastic, Transient}:\n')
print(subclass)

weighted = bool(int(input("Weighted? {T -> 1, F -> 0}:\n")))
print(weighted)

if weighted:
    method = 'weight_'
else:
    method = ''

metric_name = 'recall'

cluster = bool(int(input('cluster? {T -> 1, F -> 0}:\n')))
print(cluster)

tune_gamma = bool(int(input('tune gamma? {T -> 1, F -> 0}:\n')))
if tune_gamma == False:
    gamma = float(input('gamma: \n'))
    print(gamma)
    
if cluster:
    carpet = '/home/jmolina/carpet/to_transfer/'
else:
    carpet = 'pickle2/models/'

iteration = input('iteration?\n')
if iteration=='':
    iteration = 0
    model_dict = {}
    gamma_dict = {}
else:
    iteration = int(iteration)-1
    load = bool(int(input('load? {T -> 1, F -> 0}:\n')))
    if load:
        model_pickle = pickle_load(carpet + 'xgb_'+subclass.lower()+'_FL_'+method+metric_name+'_model_CV_dict2.pkl')
        model_dict = model_pickle['model_dict']
        gamma_pickle = pickle_load(carpet + 'xgb_'+subclass.lower()+'_FL_'+method+metric_name+'_gamma_CV_dict.pkl')
        gamma_dict = gamma_pickle['gamma_dict']
    else:
        model_dict = {}
        gamma_dict = {}


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

        if weighted:
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
        else:
            if subclass=='Hierarchical':
                X_train = X_train_hierarchical
                y_train_enc = LabelEncoder(y_train_hierarchical)
                X_test = X_test_hierarchical
                y_test_enc = LabelEncoder(y_test_hierarchical)
            else:
                X_train, y_train_og, X_test, y_test_og = class_data(X_train_hierarchical, X_test_hierarchical, y_train_hierarchical,
                                         y_test_hierarchical, y_train_original, y_test_original, subclass)
                y_train_enc = LabelEncoder(y_train_og)
                y_test_enc = LabelEncoder(y_test_og)
            dtrain = xgb.DMatrix(data=X_train, label=pd.Series(y_train_enc))
        dtest = xgb.DMatrix(data=X_test, label=pd.Series(y_test_enc))
        
        
        if subclass == 'Hierarchical': n_evals=10
        elif subclass in ['Stochastic', 'Periodic']: n_evals = 12
        else: n_evals=18
        
        if tune_gamma:
            best_params = get_best_hyperparameter_gamma(dtrain, metric=metric_name, pbar=True, n_evals=n_evals)

            #Separate params
            xgb_params = {}
            for key in list(best_params.keys()):
                if key != 'FL_gamma': xgb_params[key] = best_params[key]
            print(xgb_params)
            #XGBoost train
            xgb_model = xgb_train(xgb_params, dtrain, dtest, feval=metric_name, verbose_eval=False, FL = True, gamma = best_params['FL_gamma'])
            gamma_dict[i] = best_params['FL_gamma']
            gamma_dict = {'gamma_dict': gamma_dict}
            file_name = carpet + 'xgb_'+subclass.lower()+'_FL_'+method+metric_name+'_gamma_CV2_dict.pkl'
            pickle_save(file_name, gamma_dict)
        else:
            print('yei')
            best_params = get_best_hyperparameter_gamma_fixed(dtrain, gamma = gamma, metric=metric_name, pbar=True, n_evals=n_evals)
            xgb_model = xgb_train(best_params, dtrain, dtest, feval=metric_name, verbose_eval=False, FL = True, gamma = gamma)

        #Store model in dict
        model_dict[i]=xgb_model
        #Save model
        pickle_dict = {'model_dict': model_dict}
        file_name = carpet + 'xgb_'+subclass.lower()+'_FL_'+method+metric_name+'_model_CV2_dict.pkl'
        pickle_save(file_name, pickle_dict)
        

    i+=1

print("_____All Good_____")
