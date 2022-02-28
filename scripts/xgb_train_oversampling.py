from functions.hyper_selec import best_over2_parameters_combination
from functions.preprocessing import get_data2, LabelEncoder, class_data
from functions.custom_functions import pickle_load, xgb_train, pickle_save, get_strategy
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import json
from imblearn import over_sampling

X_hierarchical, Y_hierarchical, Y_original  = get_data2()

subclass = input('subclass? {Hierarchical, Periodic, Stochastic, Transient}:\n')
print(subclass)

over_kind = input('method? [ada, bsm, ros, sm, svm]\n')

metric_name = 'recall'

cluster = bool(int(input('cluster? {T -> 1, F -> 0}:\n')))
print(cluster)

if cluster:
    load_carpet = '/home/jmolina/carpet/to_transfer/'
    save_carpet = '/home/jmolina/carpet/to_transfer/pickle/models/'
    parameters_carpet = '/home/jmolina/carpet/to_transfer/pickle/parameters/'
else:
    load_carpet = 'pickle2/models/'
    save_carpet = 'pickle2/models/sampling/'
    parameters_carpet = 'pickle2/parameters/'

iteration = input('iteration?\n')
if iteration=='':
    iteration = 0
    model_dict = {}
    params_dict = {}
else:
    iteration = int(iteration)-1
    model_dict = {}
    params_dict = {}
    params_pickle = pickle_load(parameters_carpet + 'xgb_'+subclass.lower()+'__'+over_kind+'_'+metric_name+'_params_CV_dict.pkl')
    params_dict = params_pickle['params_dict']
    model_pickle = pickle_load(save_carpet + 'xgb_'+subclass.lower()+'__'+over_kind+'_'+metric_name+'_model_CV_dict.pkl')
    model_dict = model_pickle['model_dict']

file_name = load_carpet + 'xgb_'+subclass.lower()+'_sampling_'+metric_name+'_model_CV_dict.pkl'
xgb_dict = pickle_load(file_name)['model_dict']

n_splits = 10
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=5)
i=0

for train_index, test_index in sss.split(X_hierarchical, Y_original):
    print('\n')
    print('{}/{}'.format(i+1,n_splits))

    if i>=iteration:
        #Shuffle data
        X_train_hierarchical, X_test_hierarchical = X_hierarchical.iloc[train_index], X_hierarchical.iloc[test_index]
        y_train_hierarchical, y_test_hierarchical = Y_hierarchical[train_index], Y_hierarchical[test_index]
        y_train_original, y_test_original = Y_original[train_index], Y_original[test_index]

        #Classsifier data
        X_train, y_train_og, X_test, y_test_og = class_data(X_train_hierarchical, X_test_hierarchical, y_train_hierarchical,
        y_test_hierarchical, y_train_original, y_test_original, subclass)
        y_train_enc = LabelEncoder(y_train_og)
        y_test_enc = LabelEncoder(y_test_og)
        dtest = xgb.DMatrix(data=X_test, label=pd.Series(y_test_enc))
        
        #Get xgb params
        model = xgb_dict[i]
        config = json.loads(model.save_config())
        train_params = config['learner']['gradient_booster']['updater']['grow_colmaker']['train_param']
        xgb_params = {'max_depth':train_params['max_depth'],
                'min_child_weight':train_params['min_child_weight'],
                'learning_rate':train_params['learning_rate'],
                'gamma':train_params['gamma'],
                'subsample':train_params['subsample'],
                'colsample_bytree':train_params['colsample_bytree'],
                'reg_alpha':train_params['reg_alpha'],
                'reg_lambda':train_params['reg_lambda'],
                'objective':'multi:softprob',
                'verbosity': 0,
                'objective': 'multi:softprob',
                'eval_metric': "merror",
                'num_class':len(np.unique(y_train_enc))}    
    
        #Oversampling params tunning
        over_params, best_value = best_over2_parameters_combination(X_train, y_train_enc, xgb_params, over_kind, subclass, metric=metric_name)

        #Oversampling default params
        o_params = {'strategy':'minority', 'k_neighbors':3, 'n_neighbors':5, 'kind':'borderline-1'}

        for key in list(over_params.keys()):
            o_params[key]=over_params[key]

        oversampler_dict = {'sm':over_sampling.SMOTE(sampling_strategy=get_strategy(y_train_enc, o_params['strategy'], kind='over'), k_neighbors=o_params['k_neighbors']),
                            'ada':over_sampling.ADASYN(sampling_strategy=get_strategy(y_train_enc, o_params['strategy'], kind='over'), n_neighbors=o_params['n_neighbors']),
                            'bsm':over_sampling.BorderlineSMOTE(sampling_strategy=get_strategy(y_train_enc, o_params['strategy'], kind='over'), k_neighbors=o_params['k_neighbors'], kind=o_params['kind']),
                            'svm':over_sampling.SVMSMOTE(sampling_strategy=get_strategy(y_train_enc, o_params['strategy'], kind='over'), k_neighbors=o_params['k_neighbors']),
                            'ros':over_sampling.RandomOverSampler(sampling_strategy=get_strategy(y_train_enc, o_params['strategy'], kind='over'))}

        X_train_over, y_train_over = oversampler_dict[over_kind].fit_resample(X_train, y_train_enc)
        dtrain = xgb.DMatrix(data=X_train_over, label=pd.Series(y_train_over))
                    
        #XGBoost train
        xgb_model = xgb_train(xgb_params, dtrain, dtest, feval=metric_name, verbose_eval=False)

        #Store model in dict
        model_dict[i] = xgb_model
        params_dict[i] =  {'params':over_params, 'score':best_value}

        #Save model and params
        pickle_dict = {'model_dict': model_dict}
        file_name = save_carpet + 'xgb_'+subclass.lower()+'__'+over_kind+'_'+metric_name+'_model_CV_dict.pkl'
        pickle_save(file_name, pickle_dict)

        pickle_dict = {'params_dict': params_dict}
        file_name = parameters_carpet + 'xgb_'+subclass.lower()+'__'+over_kind+'_'+metric_name+'_params_CV_dict.pkl'
        pickle_save(file_name, pickle_dict)

    i+=1

print("_____All Good_____")
