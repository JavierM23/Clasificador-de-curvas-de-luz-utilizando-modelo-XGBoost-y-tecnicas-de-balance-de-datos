from collections import Counter
import numpy as np

import xgboost as xgb
import warnings

import hyperopt
from hyperopt import hp
from functools import partial
from numpy.random import RandomState
from metrics import IAM, CBA, recall, MPAUC, MAvG, MPR, MCC, min_recall
from sklearn import metrics

import pandas as pd
from tqdm import tqdm
import optuna
from optuna import samplers
from sampling_functions import get_strategy
from imblearn import under_sampling, over_sampling
import itertools

from multi_imbalance.resampling.mdo import MDO
from multi_imbalance.resampling.soup import SOUP
from multi_imbalance.resampling.spider import SPIDER3
from multi_imbalance.resampling.static_smote import StaticSMOTE

from MC_CCR import CCR, MultiClassCCR
from focal_loss_CE import focal_loss_obj


def hyperopt_objective(params, data, nclass, pbar, metric:str):
    
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)    
    
    model = xgb.XGBClassifier(learning_rate=params['learning_rate'], #
                              max_depth = int(params['max_depth']),
                              gamma = params['gamma'], #
                              colsample_bytree=params['colsample_bytree'],
                              subsample=params['subsample'], #
                              reg_alpha = int(params['reg_alpha']),
                              reg_lambda= params['reg_lambda'],
                              #min_child_weight=params['min_child_weight'],
                              min_child_weight=1,
                              #colsample_bynode=params['colsample_bynode'],
                              #colsample_bylevel=params['colsample_bylevel'],
                              max_delta_step=0,
                              objective='multi:softprob',
                              eval_metric = 'merror',
                              n_jobs=-1,
                              num_class=nclass,
                              verbosity = 0)
    feval_dict = {'IAM': IAM, 'CBA': CBA, 'recall': recall, 'MPAUC': MPAUC, 'MAvG': MAvG, 'MPR':MPR, 'MCC': MCC, 'min_recall':min_recall}

    res = xgb.cv(model.get_params(), data, num_boost_round=80, nfold=5, metrics='mlogloss',
                 feval=feval_dict[metric], stratified=True, early_stopping_rounds=25, seed=5)
    if type(pbar)!=type(None): pbar.update()
    return np.min(res['test-'+metric+'-mean']) # as hyperopt minimises

def hyperparameter_tunning(Dtrain, params_space, nclass, pbar, n_evals, metric:str, seed:int):
    trials = hyperopt.Trials()
    fmin_objetive = partial(hyperopt_objective, data=Dtrain, nclass=nclass, pbar=pbar, metric=metric)
    best_params = hyperopt.fmin(fmin_objetive, space=params_space, algo=hyperopt.tpe.suggest,
                         max_evals=n_evals, trials=trials, rstate=RandomState(seed))
    if type(pbar)!=type(None): pbar.close()
    return best_params

def get_best_hyperparameter(Dtrain, metric='recall', seed=123, pbar=True, n_evals=50, params_space=None):
    if pbar==True: 
        pbar = tqdm(total=n_evals, desc="Hyperopt")
    else:
        pbar = None
    if type(params_space)==type(None):
        params_space = {'learning_rate':hp.uniform('learning_rate', 9e-2, 7e-1),
                'max_depth':hp.quniform("max_depth", 3,35, 2),
                'min_child_weight':hp.quniform('min_child_weight', 2, 6, 2),
                'gamma':hp.uniform('gamma', 0,7),
                'colsample_bytree':hp.uniform('colsample_bytree', 0.5,0.99),
                'subsample':hp.uniform('subsample', 0.5,0.99),
                'reg_alpha':hp.quniform('reg_alpha', 0,15,3),
                'reg_lambda':hp.uniform('reg_lambda', 0,0.8)}
    else:
        print('GOOD')
    
    nclass = len(np.unique(np.array(Dtrain.get_label(), dtype=int, ndmin=1)))
        
    return hyperparameter_tunning(Dtrain, params_space, nclass=nclass, metric=metric, seed=seed, n_evals=n_evals, pbar=pbar)

def hyperopt_objective_gamma(params, data, nclass, pbar, metric:str):
    
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)    
    
    xgb_params = {'verbosity': 0,
             'eval_metric': "merror",
             'num_class':nclass,
             'nthread':-1,
             'objective':'multi:softprob'}
    xgb_params['reg_alpha'] = params['reg_alpha']
    xgb_params['max_depth'] = int(params['max_depth']) 
    xgb_params['min_child_weight'] = params['min_child_weight'] 
    xgb_params['learning_rate'] = params['learning_rate'] 
    xgb_params['gamma'] = params['gamma']
    xgb_params['subsample'] = params['subsample'] 
    xgb_params['reg_lambda'] = params['reg_lambda'] 
    xgb_params['colsample_bytree'] = params['colsample_bytree']
    
    FL_gamma = params['FL_gamma']

    feval_dict = {'IAM': IAM, 'CBA': CBA, 'recall': recall, 'MPAUC': MPAUC, 'MAvG': MAvG, 'MPR':MPR, 'MCC': MCC, 'min_recall':min_recall}

    focal_loss = partial(focal_loss_obj, gamma=FL_gamma)

    res = xgb.cv(xgb_params, data, num_boost_round=100, nfold=5, metrics='mlogloss', obj=focal_loss,
                 feval=feval_dict[metric], stratified=True, early_stopping_rounds=30, seed=5)
    if type(pbar)!=type(None): pbar.update()

    return np.min(res['test-'+metric+'-mean'])

def hyperparameter_tunning_gamma(Dtrain, params_space, nclass, pbar, n_evals, metric:str, seed:int):
    trials = hyperopt.Trials()
    fmin_objetive = partial(hyperopt_objective_gamma, data=Dtrain, nclass=nclass, pbar=pbar, metric=metric)
    best_params = hyperopt.fmin(fmin_objetive, space=params_space, algo=hyperopt.tpe.suggest,
                         max_evals=n_evals, trials=trials, rstate=RandomState(seed))
    if type(pbar)!=type(None): pbar.close()
    return best_params

def get_best_hyperparameter_gamma(Dtrain, metric='recall', seed=123, pbar=True, n_evals=50, params_space=None):
    if pbar==True: 
        pbar = tqdm(total=n_evals, desc="Hyperopt")
    else:
        pbar = None
    if type(params_space)==type(None):
        params_space = {'learning_rate':hp.uniform('learning_rate', 1e-1, 7e-1),
                'max_depth':hp.quniform("max_depth", 3,36, 3),
                'min_child_weight':hp.quniform('min_child_weight', 0, 6, 2),
                'gamma':hp.uniform('gamma', 1e-3, 1.0),
                'colsample_bytree':hp.uniform('colsample_bytree', 0.5,0.99),
                'subsample':hp.uniform('subsample', 0.6, 0.99),
                'reg_alpha':hp.uniform('reg_alpha', 1e-3, 10),
                'reg_lambda':hp.uniform('reg_lambda', 0,0.8),
                'FL_gamma':hp.uniform('FL_gamma', 1e-1, 2)}
    else:
        print('GOOD')
    
    nclass = len(np.unique(np.array(Dtrain.get_label(), dtype=int, ndmin=1)))
        
    return hyperparameter_tunning_gamma(Dtrain, params_space, nclass=nclass, metric=metric, seed=seed, n_evals=n_evals, pbar=pbar)

def hyperopt_objective_gamma_fixed(params, data, gamma, nclass, pbar, metric:str):
    
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)    
    
    xgb_params = {'verbosity': 0,
             'eval_metric': "merror",
             'num_class':nclass,
             'nthread':-1,
             'objective':'multi:softprob'}
    xgb_params['reg_alpha'] = params['reg_alpha']
    xgb_params['max_depth'] = int(params['max_depth']) 
    xgb_params['min_child_weight'] = params['min_child_weight'] 
    xgb_params['learning_rate'] = params['learning_rate'] 
    xgb_params['gamma'] = params['gamma']
    xgb_params['subsample'] = params['subsample'] 
    xgb_params['reg_lambda'] = params['reg_lambda'] 
    xgb_params['colsample_bytree'] = params['colsample_bytree']
    
    feval_dict = {'IAM': IAM, 'CBA': CBA, 'recall': recall, 'MPAUC': MPAUC, 'MAvG': MAvG, 'MPR':MPR, 'MCC': MCC, 'min_recall':min_recall}

    focal_loss = partial(focal_loss_obj, gamma=gamma)

    res = xgb.cv(xgb_params, data, num_boost_round=100, nfold=5, metrics='mlogloss', obj=focal_loss,
                 feval=feval_dict[metric], stratified=True, early_stopping_rounds=30, seed=5)
    if type(pbar)!=type(None): pbar.update()

    return np.min(res['test-'+metric+'-mean'])

def hyperparameter_tunning_gamma_fixed(Dtrain, params_space, gamma, nclass, pbar, n_evals, metric:str, seed:int):
    trials = hyperopt.Trials()
    fmin_objetive = partial(hyperopt_objective_gamma_fixed, data=Dtrain, gamma=gamma, nclass=nclass, pbar=pbar, metric=metric)
    best_params = hyperopt.fmin(fmin_objetive, space=params_space, algo=hyperopt.tpe.suggest,
                         max_evals=n_evals, trials=trials, rstate=RandomState(seed))
    if type(pbar)!=type(None): pbar.close()
    return best_params

def get_best_hyperparameter_gamma_fixed(Dtrain, gamma, metric='recall', seed=123, pbar=True, n_evals=50, params_space=None):
    if pbar==True: 
        pbar = tqdm(total=n_evals, desc="Hyperopt")
    else:
        pbar = None
    if type(params_space)==type(None):
        params_space = {'learning_rate':hp.uniform('learning_rate', 1e-1, 7e-1),
                'max_depth':hp.quniform("max_depth", 3,36, 3),
                'min_child_weight':hp.quniform('min_child_weight', 0, 6, 2),
                'gamma':hp.uniform('gamma', 1e-3, 1.0),
                'colsample_bytree':hp.uniform('colsample_bytree', 0.5,0.99),
                'subsample':hp.uniform('subsample', 0.6, 0.99),
                'reg_alpha':hp.uniform('reg_alpha', 1e-3, 10),
                'reg_lambda':hp.uniform('reg_lambda', 0,0.8)}
    else:
        print('GOOD')
    
    nclass = len(np.unique(np.array(Dtrain.get_label(), dtype=int, ndmin=1)))
        
    return hyperparameter_tunning_gamma_fixed(Dtrain, params_space, gamma, nclass=nclass, metric=metric, seed=seed, n_evals=n_evals, pbar=pbar)





def optuna_objective(trial, data, nclass, pbar, metric:str):
    
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)    
    
    params = {'verbosity': 0,
             'objective': 'multi:softprob',
             'eval_metric': "merror",
             'num_class':nclass,
             'nthread':-1}
    params['reg_alpha'] = trial.suggest_loguniform('reg_alpha', 1e-3, 10)
    params['max_depth'] = trial.suggest_int('max_depth', 3, 36, 3)
    params['min_child_weight'] = trial.suggest_int('min_child_weight', 0, 6, 2)
    params['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-1, 7e-1)
    params['gamma'] = trial.suggest_loguniform('gamma', 1e-3, 1.0)
    params['subsample'] = trial.suggest_loguniform('subsample', 0.6, 1.0)
    params['reg_lambda'] = trial.suggest_uniform('reg_lambda', 0, 0.8)
    params['colsample_bytree'] = trial.suggest_uniform('colsample_bytree', 0, 0.8)


    feval_dict = {'IAM': IAM, 'CBA': CBA, 'recall': recall, 'MPAUC': MPAUC, 'MAvG': MAvG, 'MPR':MPR, 'MCC': MCC, 'min_recall':min_recall}

    res = xgb.cv(params, data, num_boost_round=80, nfold=5, metrics='mlogloss',
                 feval=feval_dict[metric], stratified=True, early_stopping_rounds=25, seed=5)
    if type(pbar)!=type(None): pbar.update()

    return np.min(res['test-'+metric+'-mean'])

def get_best_optunaparameters(Dtrain, metric='recall', seed=123, pbar=None, n_evals=50):
    sampler = samplers.TPESampler(seed=seed) 
    optuna_hpt = optuna.create_study(sampler=sampler, direction='minimize', study_name='optuna_hpt')
    nclass = len(np.unique(np.array(Dtrain.get_label(), dtype=int, ndmin=1)))
    opt_objetive =  partial(optuna_objective, data=Dtrain, nclass=nclass, pbar=pbar, metric=metric)
    optuna_hpt.optimize(opt_objetive, n_trials=n_evals)
    return optuna_hpt.best_params

def optuna_objective_gamma(trial, data, nclass, pbar, metric:str):
    
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)    
    
    params = {'verbosity': 0,
             'objective': 'multi:softprob',
             'eval_metric': "merror",
             'num_class':nclass,
             'nthread':-1}
    params['reg_alpha'] = trial.suggest_loguniform('reg_alpha', 1e-3, 10)
    params['max_depth'] = trial.suggest_int('max_depth', 3, 36, 3)
    params['min_child_weight'] = trial.suggest_int('min_child_weight', 0, 6, 2)
    params['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-1, 7e-1)
    params['gamma'] = trial.suggest_loguniform('gamma', 1e-3, 1.0)
    params['subsample'] = trial.suggest_loguniform('subsample', 0.6, 1.0)
    params['reg_lambda'] = trial.suggest_uniform('reg_lambda', 0, 0.8)
    params['colsample_bytree'] = trial.suggest_uniform('colsample_bytree', 0, 0.8)
    FL_gamma = trial.suggest_loguniform('FL_gamma', 0.1, 2.0)

    feval_dict = {'IAM': IAM, 'CBA': CBA, 'recall': recall, 'MPAUC': MPAUC, 'MAvG': MAvG, 'MPR':MPR, 'MCC': MCC, 'min_recall':min_recall}

    focal_loss_obj = partial(focal_loss_obj1, gamma=FL_gamma)

    res = xgb.cv(params, data, num_boost_round=100, nfold=5, metrics='mlogloss', obj=focal_loss_obj,
                 feval=feval_dict[metric], stratified=True, early_stopping_rounds=30, seed=5)
    if type(pbar)!=type(None): pbar.update()

    return np.min(res['test-'+metric+'-mean'])

def get_best_optunaparameters_gamma(Dtrain, metric='recall', seed=123, pbar=None, n_evals=50):
    sampler = samplers.TPESampler(seed=seed) 
    optuna_hpt = optuna.create_study(sampler=sampler, direction='minimize', study_name='optuna_hpt')
    nclass = len(np.unique(np.array(Dtrain.get_label(), dtype=int, ndmin=1)))
    opt_objetive =  partial(optuna_objective_gamma, data=Dtrain, nclass=nclass, pbar=pbar, metric=metric)
    optuna_hpt.optimize(opt_objetive, n_trials=n_evals)
    return optuna_hpt.best_params

def best_under_parameters_combination(X, y, xgb_params, under_kind, subclass, metric='recall', seed=123):

    strategy_dict = {'Periodic':[(2,[0,1])],
                     'Stochastic':[(1,[0])],
                     'Transient':[(1,[0])]}

    strategy = strategy_dict[subclass]
                     
    n_neighbours = [3,5,7]
    version = [2,3]
    parameters_combination_list = {'cnn':list(itertools.product(strategy, n_neighbours)),
                                   'enn':list(itertools.product(strategy, n_neighbours)),
                                   'renn':list(itertools.product(strategy, n_neighbours)),
                                   'allknn':list(itertools.product(strategy, n_neighbours)),
                                   'oss':list(itertools.product(strategy, n_neighbours)),
                                   'ncr':list(itertools.product(strategy, n_neighbours)),
                                   'tl':list(itertools.product(strategy)),
                                   'iht':list(itertools.product(strategy)),
                                   'nm':list(itertools.product(strategy, n_neighbours, version)),
                                   'rus':list(itertools.product(strategy))}

    n_evals = len(parameters_combination_list[under_kind])
    pbar = tqdm(total=n_evals, desc="Combinations")
    params_array = []
    score_array = np.zeros(n_evals)
    for i, combination in enumerate(parameters_combination_list[under_kind]):
        if under_kind=='cnn':
            params = {'strategy':combination[0], 'n_neighbors':combination[1]}
            under_model = under_sampling.CondensedNearestNeighbour(sampling_strategy=params['strategy'][1], n_neighbors=params['n_neighbors'])
        elif under_kind=='enn':
            params = {'strategy':combination[0], 'n_neighbors':combination[1],'kind_sel':'all'}
            under_model=under_sampling.EditedNearestNeighbours(sampling_strategy=params['strategy'][1], n_neighbors=params['n_neighbors'], kind_sel=params['kind_sel'])
        elif under_kind=='renn':
            params = {'strategy':combination[0], 'n_neighbors':combination[1],'kind_sel':'all'}
            under_model=under_sampling.RepeatedEditedNearestNeighbours(sampling_strategy=params['strategy'][1], n_neighbors=params['n_neighbors'], kind_sel=params['kind_sel'])
        elif under_kind=='allknn':
            params = {'strategy':combination[0], 'n_neighbors':combination[1],'kind_sel':'all'}
            under_model=under_sampling.AllKNN(sampling_strategy=params['strategy'][1], n_neighbors=params['n_neighbors'], kind_sel=params['kind_sel'])
        elif under_kind=='oss':
            params = {'strategy':combination[0], 'n_neighbors':combination[1]}
            under_model = under_sampling.OneSidedSelection(sampling_strategy=params['strategy'][1], n_neighbors=params['n_neighbors'])
        elif under_kind=='ncr':
            params = {'strategy':combination[0], 'n_neighbors':combination[1],'kind_sel':'all'}
            under_model=under_sampling.NeighbourhoodCleaningRule(sampling_strategy=params['strategy'][1], n_neighbors=params['n_neighbors'], kind_sel=params['kind_sel'])
        elif under_kind=='tl':
            params = {'strategy':combination[0]}
            under_model=under_sampling.TomekLinks(sampling_strategy=params['strategy'][1])
        elif under_kind=='iht':
            params = {'strategy':combination[0]}
            under_model=under_sampling.InstanceHardnessThreshold(sampling_strategy=get_strategy(y, params['strategy'][0], kind='under'))
        elif under_kind=='nm':
            params = {'strategy':combination[0], 'n_neighbors':combination[1],'version':combination[2][0]}
            under_model=under_sampling.NearMiss(sampling_strategy=get_strategy(y, params['strategy'][0], kind='under'), n_neighbors=params['n_neighbors'], version=params['version'])
        elif under_kind=='rus':
            params = {'strategy':combination[0][0]}
            under_model=under_sampling.RandomUnderSampler(sampling_strategy=get_strategy(y, params['strategy'][0], kind='under'))

        X_train_under, y_train_under = under_model.fit_resample(X, y)
        dtrain = xgb.DMatrix(data=X_train_under, label=pd.Series(y_train_under))

        feval_dict = {'IAM': IAM, 'CBA': CBA, 'recall': recall, 'MPAUC': MPAUC, 'MAvG': MAvG, 'MPR':MPR, 'MCC': MCC}
        res = xgb.cv(xgb_params, dtrain, num_boost_round=80, nfold=5, metrics='mlogloss',
                 feval=feval_dict[metric], stratified=True, early_stopping_rounds=25, seed=seed)
        print('params {}, score {}'.format(params, round(np.min(res['test-'+metric+'-mean'])*(-1),2)))
        if type(pbar)!=type(None): pbar.update()

        params_array.append(params)
        score_array[i] = np.min(res['test-'+metric+'-mean'])*(-1)

    best_params, best_value = params_array[np.argmax(score_array)], np.max(score_array)

    return best_params, best_value

def best_over_parameters_combination(X, y, xgb_params, over_kind, subclass, metric='recall', seed=123):

    strategy_dict = {'Periodic':[2],
                     'Stochastic':[0],
                     'Transient':[0]}
                     
    strategy_array = strategy_dict[subclass]

    k_neighbours_dict = {'Periodic':[5,7],
                        'Stochastic':[5,7],
                        'Transient':[3,5]}
    
    k_neighbours = k_neighbours_dict[subclass]

    kind = ['borderline-1', 'borderline-2']
    parameters_combination_list = {'sm':list(itertools.product(strategy_array, k_neighbours)),
                                   'ada':list(itertools.product(strategy_array, k_neighbours)),
                                   'bsm':list(itertools.product(strategy_array, k_neighbours, kind)),
                                   'svm':list(itertools.product(strategy_array, k_neighbours)),
                                   'ros':list(itertools.product(strategy_array))}
                                   
    n_evals = len(parameters_combination_list[over_kind])
    params_array = []
    score_array = np.zeros(n_evals)

    for i, combination in enumerate(parameters_combination_list[over_kind]):
        if over_kind=='sm':
                over_params = {'strategy':combination[0], 'k_neighbors':combination[1]}
                over_model = over_sampling.SMOTE(sampling_strategy=get_strategy(y, over_params['strategy'], kind='over'), k_neighbors=over_params['k_neighbors'])
        elif over_kind=='ada':
                over_params = {'strategy':combination[0], 'n_neighbors':combination[1]}
                over_model=over_sampling.ADASYN(sampling_strategy=get_strategy(y, over_params['strategy'], kind='over'), n_neighbors=over_params['n_neighbors'])
        elif over_kind=='bsm':
                over_params = {'strategy':combination[0], 'k_neighbors':combination[1], 'kind':combination[2]}
                over_model=over_sampling.BorderlineSMOTE(sampling_strategy=get_strategy(y, over_params['strategy'], kind='over'), k_neighbors=over_params['k_neighbors'], kind=over_params['kind'])
        elif over_kind=='svm':
                over_params = {'strategy':combination[0], 'k_neighbors':combination[1]}
                over_model=over_sampling.SVMSMOTE(sampling_strategy=get_strategy(y, over_params['strategy'], kind='over'), k_neighbors=over_params['k_neighbors'])
        elif over_kind=='ros':
                over_params = {'strategy':combination[0]}
                over_model=over_sampling.RandomOverSampler(sampling_strategy=get_strategy(y, over_params['strategy'], kind='over'))     

        if n_evals>1:
            pbar = tqdm(total=n_evals, desc="Combinations")
            X_train_over, y_train_over = over_model.fit_resample(X, y)
            dtrain = xgb.DMatrix(data=X_train_over, label=pd.Series(y_train_over))

            feval_dict = {'IAM': IAM, 'CBA': CBA, 'recall': recall, 'MPAUC': MPAUC, 'MAvG': MAvG, 'MPR':MPR, 'MCC': MCC}
            res = xgb.cv(xgb_params, dtrain, num_boost_round=80, nfold=5, metrics='mlogloss',
                 feval=feval_dict[metric], stratified=True, early_stopping_rounds=25, seed=seed)
            print('params {}, score {}'.format(over_params, round(np.min(res['test-'+metric+'-mean'])*(-1),2)))
            if type(pbar)!=type(None): pbar.update()

            params_array.append(over_params)
            score_array[i] = np.min(res['test-'+metric+'-mean'])*(-1)
        else:
            params_array.append(over_params)
            score_array[i] = 1

    best_params, best_value = params_array[np.argmax(score_array)], np.max(score_array)

    return best_params, best_value

def best_over2_parameters_combination(X, y, xgb_params, over_kind, subclass, metric='recall', seed=123):

    strategy_dict = {'Periodic':[2],
                     'Stochastic':[0],
                     'Transient':[0]}
                         
    k_neighbours_dict = {'Periodic':[5,7],
                        'Stochastic':[5,7],
                        'Transient':[3,5]}

    km_neighbours = [2,4,6]
    kind = ['borderline-1', 'borderline-2']
    parameters_combination_list = {'sm':list(itertools.product(strategy_dict[subclass], k_neighbours_dict[subclass])),
                                   'ada':list(itertools.product(strategy_dict[subclass], k_neighbours_dict[subclass])),
                                   'bsm':list(itertools.product(strategy_dict[subclass], k_neighbours_dict[subclass], kind)),
                                   'ksm':list(itertools.product(strategy_dict[subclass], km_neighbours)),
                                   'svm':list(itertools.product(strategy_dict[subclass], k_neighbours_dict[subclass])),
                                   'ros':list(itertools.product(strategy_dict[subclass]))}
                                   
    n_evals = len(parameters_combination_list[over_kind])
    pbar = tqdm(total=n_evals, desc="Combinations")
    params_array = []
    score_array = np.zeros(n_evals)

    for i, combination in enumerate(parameters_combination_list[over_kind]):
        if over_kind=='sm':
                over_params = {'strategy':combination[0], 'k_neighbors':combination[1]}
                over_model = over_sampling.SMOTE(sampling_strategy=get_strategy(y, over_params['strategy'], kind='over'), k_neighbors=over_params['k_neighbors'])
        elif over_kind=='ada':
                over_params = {'strategy':combination[0], 'n_neighbors':combination[1]}
                over_model=over_sampling.ADASYN(sampling_strategy=get_strategy(y, over_params['strategy'], kind='over'), n_neighbors=over_params['n_neighbors'])
        elif over_kind=='bsm':
                over_params = {'strategy':combination[0], 'k_neighbors':combination[1],'kind':combination[2]}
                over_model=over_sampling.BorderlineSMOTE(sampling_strategy=get_strategy(y, over_params['strategy'], kind='over'), k_neighbors=over_params['k_neighbors'], kind=over_params['kind'])
        elif over_kind=='ksm':
                over_params = {'strategy':combination[0], 'k_neighbors':combination[1]}
                over_model=over_sampling.KMeansSMOTE(sampling_strategy=get_strategy(y, over_params['strategy'], kind='over'), k_neighbors=over_params['k_neighbors'])
        elif over_kind=='svm':
                over_params = {'strategy':combination[0], 'k_neighbors':combination[1]}
                over_model=over_sampling.SVMSMOTE(sampling_strategy=get_strategy(y, over_params['strategy'], kind='over'), k_neighbors=over_params['k_neighbors'])
        elif over_kind=='ros':
                over_params = {'strategy':combination[0]}
                over_model=over_sampling.RandomOverSampler(sampling_strategy=get_strategy(y, over_params['strategy'], kind='over'))     

        X_train_over, y_train_over = over_model.fit_resample(X, y)
        dtrain = xgb.DMatrix(data=X_train_over, label=pd.Series(y_train_over))

        feval_dict = {'IAM': IAM, 'CBA': CBA, 'recall': recall, 'MPAUC': MPAUC, 'MAvG': MAvG, 'MPR':MPR, 'MCC': MCC}
        res = xgb.cv(xgb_params, dtrain, num_boost_round=80, nfold=5, metrics='mlogloss',
                 feval=feval_dict[metric], stratified=True, early_stopping_rounds=25, seed=seed)

        print('params {}, score {}'.format(over_params, round(np.min(res['test-'+metric+'-mean'])*(-1),2)))
        if type(pbar)!=type(None): pbar.update()

        params_array.append(over_params)
        score_array[i] = np.min(res['test-'+metric+'-mean'])*(-1)

    best_params, best_value = params_array[np.argmax(score_array)], np.max(score_array)

    return best_params, best_value
    
def best_under_over_parameters_combination(X, y, xgb_params, under_kind, over_kind, subclass, metric='recall', seed=123):

    strategy_dict = {'Periodic':[(2,[0,1])],
                     'Stochastic':[(1,[0])],
                     'Transient':[(1,[0])]}

    strategy_tuple = strategy_dict[subclass]

    n_neighbours = [5,7]
    version = [2,3]

    under_parameters_combination_list = {'cnn':list(itertools.product(strategy_tuple, n_neighbours)),
                                   'enn':list(itertools.product(strategy_tuple, n_neighbours)),
                                   'renn':list(itertools.product(strategy_tuple, n_neighbours)),
                                   'allknn':list(itertools.product(strategy_tuple, n_neighbours)),
                                   'oss':list(itertools.product(strategy_tuple, n_neighbours)),
                                   'ncr':list(itertools.product(strategy_tuple, n_neighbours)),
                                   'tl':list(itertools.product(strategy_tuple)),
                                   'iht':list(itertools.product(strategy_tuple)),
                                   'nm':list(itertools.product(strategy_tuple, n_neighbours, version)),
                                   'rus':list(itertools.product(strategy_tuple))}
    k_neighbours = [5,7]
    km_neighbours = [2,4,6]
    kind = ['borderline-1', 'borderline-2']
    over_parameters_combination_list = {'sm':list(itertools.product(k_neighbours)),
                                   'ada':list(itertools.product(k_neighbours)),
                                   'bsm':list(itertools.product(k_neighbours, kind)),
                                   'ksm':list(itertools.product(km_neighbours)),
                                   'svm':list(itertools.product(k_neighbours)),
                                   'ros':list(itertools.product(['x']))}

    n_evals = len(under_parameters_combination_list[under_kind])*len(over_parameters_combination_list[over_kind])

    if n_evals!=1:
        pbar = tqdm(total=n_evals, desc="Combinations")
        params_array = []
        score_array = []
        
        for under_combination in under_parameters_combination_list[under_kind]:
            combination_params = {}
            if under_kind=='cnn':
                under_params = {'strategy':under_combination[0], 'n_neighbors':under_combination[1]}
                under_model = under_sampling.CondensedNearestNeighbour(sampling_strategy=under_params['strategy'][1], n_neighbors=under_params['n_neighbors'])
            elif under_kind=='enn':
                under_params = {'strategy':under_combination[0], 'n_neighbors':under_combination[1],'kind_sel':'all'}
                under_model=under_sampling.EditedNearestNeighbours(sampling_strategy=under_params['strategy'][1], n_neighbors=under_params['n_neighbors'], kind_sel=under_params['kind_sel'])
            elif under_kind=='renn':
                under_params = {'strategy':under_combination[0], 'n_neighbors':under_combination[1],'kind_sel':'all'}
                under_model=under_sampling.RepeatedEditedNearestNeighbours(sampling_strategy=under_params['strategy'][1], n_neighbors=under_params['n_neighbors'], kind_sel=under_params['kind_sel'])
            elif under_kind=='allknn':
                under_params = {'strategy':under_combination[0], 'n_neighbors':under_combination[1],'kind_sel':'all'}
                under_model=under_sampling.AllKNN(sampling_strategy=under_params['strategy'][1], n_neighbors=under_params['n_neighbors'], kind_sel=under_params['kind_sel'])
            elif under_kind=='oss':
                under_params = {'strategy':under_combination[0], 'n_neighbors':under_combination[1]}
                under_model = under_sampling.OneSidedSelection(sampling_strategy=under_params['strategy'][1], n_neighbors=under_params['n_neighbors'])
            elif under_kind=='ncr':
                under_params = {'strategy':under_combination[0], 'n_neighbors':under_combination[1],'kind_sel':'all'}
                under_model=under_sampling.NeighbourhoodCleaningRule(sampling_strategy=under_params['strategy'][1], n_neighbors=under_params['n_neighbors'], kind_sel=under_params['kind_sel'])
            elif under_kind=='tl':
                under_params = {'strategy':under_combination[0]}
                under_model=under_sampling.TomekLinks(sampling_strategy=under_params['strategy'][1])
            elif under_kind=='iht':
                under_params = {'strategy':under_combination[0]}
                under_model=under_sampling.InstanceHardnessThreshold(sampling_strategy=get_strategy(y, under_params['strategy'][0], kind='under'))
            elif under_kind=='nm':
                under_params = {'strategy':under_combination[0], 'n_neighbors':under_combination[1],'version':under_combination[2]}
                under_model=under_sampling.NearMiss(sampling_strategy=get_strategy(y, under_params['strategy'][0], kind='under'), n_neighbors=under_params['n_neighbors'], version=under_params['version'])
            elif under_kind=='rus':
                under_params = {'strategy':under_combination[0]}
                under_model=under_sampling.RandomUnderSampler(sampling_strategy=get_strategy(y, under_params['strategy'][0], kind='under'))
        
            combination_params['under']=under_params
            X_train_under, y_train_under = under_model.fit_resample(X, y)        

            for over_combination in over_parameters_combination_list[over_kind]:
                if over_kind=='sm':
                    over_params = {'strategy':under_combination[0], 'k_neighbors':over_combination[0]}
                    over_model = over_sampling.SMOTE(sampling_strategy=get_strategy(y_train_under, over_params['strategy'][0], kind='over'), k_neighbors=over_params['k_neighbors'])
                elif over_kind=='ada':
                    over_params = {'strategy':under_combination[0], 'n_neighbors':over_combination[0]}
                    over_model=over_sampling.ADASYN(sampling_strategy=get_strategy(y_train_under, over_params['strategy'][0], kind='over'), n_neighbors=over_params['n_neighbors'])
                elif over_kind=='bsm':
                    over_params = {'strategy':under_combination[0], 'k_neighbors':over_combination[0],'kind':over_combination[1]}
                    over_model=over_sampling.BorderlineSMOTE(sampling_strategy=get_strategy(y_train_under, over_params['strategy'][0], kind='over'), k_neighbors=over_params['k_neighbors'], kind=over_params['kind'])
                elif over_kind=='ksm':
                    over_params = {'strategy':under_combination[0], 'k_neighbors':over_combination[0]}
                    over_model=over_sampling.KMeansSMOTE(sampling_strategy=get_strategy(y_train_under, over_params['strategy'][0], kind='over'), k_neighbors=over_params['k_neighbors'])
                elif over_kind=='svm':
                    over_params = {'strategy':under_combination[0], 'k_neighbors':over_combination[0]}
                    over_model=over_sampling.SVMSMOTE(sampling_strategy=get_strategy(y_train_under, over_params['strategy'][0], kind='over'), k_neighbors=over_params['k_neighbors'])
                elif over_kind=='ros':
                    over_params = {'strategy':under_combination[0]}
                    over_model=over_sampling.RandomOverSampler(sampling_strategy=get_strategy(y_train_under, over_params['strategy'][0], kind='over'))
        
                combination_params['over']=over_params
                under_counter = Counter(y_train_under)

                if under_counter[under_combination[0][0]]<under_counter[under_combination[0][0]+1]:
                    print(Counter(y_train_under))
                    print(get_strategy(y_train_under, under_combination[0][0], kind='over'))
                    params_array.append(combination_params)
                    score_array.append(0)
                else:
                    X_train_over, y_train_over = over_model.fit_resample(X_train_under, y_train_under)
                    dtrain = xgb.DMatrix(data=X_train_over, label=pd.Series(y_train_over))

                    feval_dict = {'IAM': IAM, 'CBA': CBA, 'recall': recall, 'MPAUC': MPAUC, 'MAvG': MAvG, 'MPR':MPR, 'MCC': MCC}
                    res = xgb.cv(xgb_params, dtrain, num_boost_round=80, nfold=5, metrics='mlogloss',
                    feval=feval_dict[metric], stratified=True, early_stopping_rounds=25, seed=seed)

                    if type(pbar)!=type(None): pbar.update()

                    params_array.append(combination_params)
                    score_array.append(np.min(res['test-'+metric+'-mean'])*(-1))
        score_array = np.array(score_array)
        best_params, best_value = params_array[np.argmax(score_array)], np.max(score_array)
    else:
        best_params, best_value = {'under':{'strategy':strategy_tuple[0]}, 'over':{'strategy':strategy_tuple[0]}}, 1

    return best_params, best_value

def best_multi_imbalance(X, y, xgb_params, kind, subclass, metric='recall', seed=123):

    maj_int_min_dict = {'Periodic':[{'maj':[0,1], 'int':[2], 'min':[3,4,5]}, {'maj':[0,1,2], 'int':[], 'min':[3,4,5]}],
                        'Stochastic':[{'maj':[0], 'int':[1], 'min':[2,3,4]}, {'maj':[0], 'int':[], 'min':[1,2,3,4]}],
                        'Transient':[{'maj':[0], 'int':[], 'min':[1,2,3]}]}
    
    k_neighbours_dict = {'Periodic':[5,7],
                        'Stochastic':[5,7],
                        'Transient':[3,5]}

    prop = [0.6, 0.8, 1]

    parameters_combination_list = {'mdo':list(itertools.product(maj_int_min_dict[subclass], k_neighbours_dict[subclass], prop)),
                                   'soup':list(itertools.product(maj_int_min_dict[subclass], k_neighbours_dict[subclass])),
                                   'spider':list(itertools.product(maj_int_min_dict[subclass], k_neighbours_dict[subclass])),
                                   'ssm':list(itertools.product(['x']))}

    n_evals = len(parameters_combination_list[kind])
    pbar = tqdm(total=n_evals, desc="Combinations")
    params_array = []
    score_array = np.zeros(n_evals)

    for i, combination in enumerate(parameters_combination_list[kind]):
        if kind=='mdo':
            params = {'maj_int_min':combination[0], 'k':combination[1], 'prop':combination[2]}
            resampler_model = MDO(k=params['k'], maj_int_min=params['maj_int_min'], prop=params['prop'])
        elif kind=='soup':
            params = {'maj_int_min':combination[0], 'k':combination[1]}
            resampler_model = SOUP(k=params['k'], maj_int_min=params['maj_int_min'])
        elif kind=='spider':
            params = {'maj_int_min':combination[0], 'k':combination[1]}
            resampler_model = SPIDER3(k=params['k'], majority_classes=params['maj_int_min']['maj'], intermediate_classes=params['maj_int_min']['int'], minority_classes=params['maj_int_min']['min'], cost=None)
        elif kind=='ssm':
            resampler_model = StaticSMOTE()
        
        X_train_over, y_train_over = resampler_model.fit_transform(X.values, np.array(y))
        dtrain = xgb.DMatrix(data=X_train_over, label=pd.Series(y_train_over))

        feval_dict = {'IAM': IAM, 'CBA': CBA, 'recall': recall, 'MPAUC': MPAUC, 'MAvG': MAvG, 'MPR':MPR, 'MCC': MCC}
        res = xgb.cv(xgb_params, dtrain, num_boost_round=80, nfold=5, metrics='mlogloss',
                 feval=feval_dict[metric], stratified=True, early_stopping_rounds=25, seed=seed)

        print('params {}, score {}'.format(params, round(np.min(res['test-'+metric+'-mean'])*(-1),2)))
        if type(pbar)!=type(None): pbar.update()

        params_array.append(params)
        score_array[i] = np.min(res['test-'+metric+'-mean'])*(-1)

    best_params, best_value = params_array[np.argmax(score_array)], np.max(score_array)

    return best_params, best_value
    
def MC_CCR_tunning(trial, X, y, xgb_params, method, pbar, subclass):
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)    
    if subclass == 'Periodic':
        energy = trial.suggest_uniform('energy', 1, 30)
    else:
        energy = trial.suggest_uniform('energy', 5, 50)
    cleaning_strategy = trial.suggest_categorical('cleaning', ['ignore', 'translate', 'remove'])

    if method == 'ccr':
        resampler_model = CCR(energy=energy, cleaning_strategy=cleaning_strategy)
    else:
        resampler_model = MultiClassCCR(energy=energy, cleaning_strategy=cleaning_strategy)

    X_resampled, y_resampled = resampler_model.fit_sample(X.values, np.array(y))

    dtrain = xgb.DMatrix(data=X_resampled, label=pd.Series(y_resampled))

    res = xgb.cv(xgb_params, dtrain, num_boost_round=80, nfold=5, metrics='mlogloss',
                 feval=recall, stratified=True, early_stopping_rounds=25, seed=5)
    if type(pbar)!=type(None): pbar.update()

    return np.min(res['test-recall-mean'])

def get_MC_CCR_energy(X, y, xgb_params, method, subclass, seed=123, n_evals=50):
    pbar = tqdm(total=n_evals, desc="Optuna")
    sampler = samplers.TPESampler(seed=seed) 
    optuna_hpt = optuna.create_study(sampler=sampler, direction='minimize', study_name='optuna_hpt')
    opt_objetive =  partial(MC_CCR_tunning, X=X, y=y, xgb_params=xgb_params, method=method, pbar=pbar, subclass=subclass)
    optuna_hpt.optimize(opt_objetive, n_trials=n_evals)
    return optuna_hpt.best_params, optuna_hpt.best_value