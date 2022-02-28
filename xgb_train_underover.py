from functions.hyper_selec import best_under_over_parameters_combination, best_over_parameters_combination
from functions.preprocessing import get_data2, LabelEncoder, class_data
from functions.custom_functions import pickle_load, xgb_train, pickle_save
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import json
from imblearn import under_sampling, over_sampling
from functions.sampling_functions import get_strategy

X_hierarchical, Y_hierarchical, Y_original  = get_data2()

subclass = input('subclass? {Hierarchical, Periodic, Stochastic, Transient}:\n')
print(subclass)

metric_name = 'recall'

under_kind = input('method? [allknn, cnn, enn, iht, ncr, nm, oss, renn, rus, tl]\n')
print(under_kind)


cluster = bool(int(input('cluster? {T -> 1, F -> 0}:\n')))
print(cluster)

if cluster:
    load_carpet = '/home/jmolina/carpet/to_transfer/'
    load_params_carpet = load_carpet
    save_carpet = '/home/jmolina/carpet/to_transfer/pickle/models/'
    parameters_carpet = '/home/jmolina/carpet/to_transfer/pickle/parameters/'
else:
    load_carpet = 'pickle2/models/'
    save_carpet = 'pickle2/models/sampling/'
    parameters_carpet = 'pickle2/parameters/'
    load_params_carpet = parameters_carpet
    

over_methods = {'Periodic':['svm','bsm','ada','sm','ros'],
                'Stochastic':['ada','bsm','ros','sm','svm'],
                'Transient':['ada','bsm','ros','sm','svm']}


iteration = input('iteration?\n')
if iteration=='':
    iteration = 0
    model_dict = {}
    params_dict = {}
    for key in over_methods[subclass]:
        model_dict[key] = {}
        params_dict[key] = {}
    model_iteration = 0
else:
    iteration = int(iteration)-1
    model_dict = {}
    params_dict = {}
    over_array = ['svm','bsm','ada','sm','ros']
    for over_method in over_array:
        params_pickle = pickle_load(parameters_carpet + 'xgb_'+subclass.lower()+'_'+under_kind+'_'+over_method+'_'+metric_name+'_params_CV_dict.pkl')
        params_dict[over_method] = params_pickle['params_dict']
        model_pickle = pickle_load(save_carpet + 'xgb_'+subclass.lower()+'_'+under_kind+'_'+over_method+'_'+metric_name+'_model_CV_dict.pkl')
        model_dict[over_method] = model_pickle['model_dict']
    model_iteration = int(input('[svm-0, bsm-1, ada-2, sm-3, ros-4]?\n'))

#Load xgb models
file_name = load_carpet + 'xgb_'+subclass.lower()+'_sampling_'+metric_name+'_model_CV_dict.pkl'
xgb_dict = pickle_load(file_name)['model_dict']

#Load sampling params
if (under_kind in ['cnn', 'enn', 'renn', 'allknn', 'ncr']) and (subclass in ['Periodic', 'Stochastic']):
    pickle_params = pickle_load(load_params_carpet +'xgb_'+subclass.lower()+'_'+under_kind+'_'+metric_name+'_params_dict.pkl')
    under_params = pickle_params['params_dict'][0]

n_splits = 10
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=5)

i=0
#Outer layer
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

        #Undersampling default params
        u_params = {'strategy':'majority', 'n_neighbors':3, 'kind_sel':'all','allow_minority':False, 'threshold_cleaning':0.5,
                'n_seeds_S':1, 'version':1}

        if (under_kind in ['cnn', 'enn', 'renn', 'allknn', 'ncr']) and (subclass in ['Periodic', 'Stochastic']): #Undersampling method w/ loaded params

            #Update params
            for key in list(under_params.keys()):
                u_params[key]=under_params[key]

            undersampler_dict={'cnn':under_sampling.CondensedNearestNeighbour(sampling_strategy=u_params['strategy'][1], n_neighbors=u_params['n_neighbors']),
                        'enn':under_sampling.EditedNearestNeighbours(sampling_strategy=u_params['strategy'][1], n_neighbors=u_params['n_neighbors'], kind_sel=u_params['kind_sel']),
                        'renn':under_sampling.RepeatedEditedNearestNeighbours(sampling_strategy=u_params['strategy'][1], n_neighbors=u_params['n_neighbors'], kind_sel=u_params['kind_sel']),
                        'allknn':under_sampling.AllKNN(sampling_strategy=u_params['strategy'][1], n_neighbors=u_params['n_neighbors'], kind_sel=u_params['kind_sel']),
                        'ncr':under_sampling.NeighbourhoodCleaningRule(sampling_strategy=u_params['strategy'][1], n_neighbors=u_params['n_neighbors'], kind_sel=u_params['kind_sel'])}

            print('Training {} method'.format(under_kind))
            #Apply undersampling method
            X_train_under, y_train_under = undersampler_dict[under_kind].fit_resample(X_train, y_train_enc)

            #Iteration over oversampling methods

            if i==iteration:
                over_array=over_methods[subclass][model_iteration:]
            else:
                over_array = over_methods[subclass]

            for over_kind in over_array:
                print('\n')
                print(under_kind, '+', over_kind)

                #Oversampling params tunning
                over_params, best_value = best_over_parameters_combination(X_train_under, y_train_under, xgb_params, over_kind, subclass, metric=metric_name)

                #Oversampling default params
                o_params = {'strategy':'minority', 'k_neighbors':3, 'n_neighbors':5, 'kind':'borderline-1'}
                #Update params
                for key in list(over_params.keys()):
                    o_params[key]=over_params[key]

                oversampler_dict = {'sm':over_sampling.SMOTE(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'), k_neighbors=o_params['k_neighbors']),
                            'ada':over_sampling.ADASYN(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'), n_neighbors=o_params['n_neighbors']),
                            'bsm':over_sampling.BorderlineSMOTE(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'), k_neighbors=o_params['k_neighbors'], kind=o_params['kind']),
                            'ksm':over_sampling.KMeansSMOTE(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'), k_neighbors=o_params['k_neighbors']),
                            'svm':over_sampling.SVMSMOTE(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'), k_neighbors=o_params['k_neighbors']),
                            'ros':over_sampling.RandomOverSampler(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'))}

                #Apply oversampling method
                X_train_over, y_train_over = oversampler_dict[over_kind].fit_resample(X_train_under, y_train_under)
                dtrain = xgb.DMatrix(data=X_train_over, label=pd.Series(y_train_over))
            
                #XGBoost train
                xgb_model = xgb_train(xgb_params, dtrain, dtest, feval=metric_name, verbose_eval=False)

                #Store model in dict
                model_dict[over_kind][i] = xgb_model
                params_dict[over_kind][i] = {'params':{'under':under_params, 'over':over_params}, 'score':best_value}

                #Save model and params
                pickle_dict = {'model_dict': model_dict[over_kind]}
                file_name = save_carpet + 'xgb_'+subclass.lower()+'_'+under_kind+'_'+over_kind+'_'+metric_name+'_model_CV_dict.pkl'
                pickle_save(file_name, pickle_dict)

                pickle_dict = {'params_dict': params_dict[over_kind]}
                file_name = parameters_carpet + 'xgb_'+subclass.lower()+'_'+under_kind+'_'+over_kind+'_'+metric_name+'_params_CV_dict.pkl'
                pickle_save(file_name, pickle_dict)

        else: #['tl', 'iht', 'nm', 'rus','oss']
            if under_kind == 'nm' and subclass == 'Stochastic': over_array = ['bsm','ros','sm','svm']
            elif under_kind in ['oss','rus', 'iht'] and subclass == 'Periodic': over_array = ['svm','ada','bsm','ros','sm',]
            else: over_array = over_methods[subclass]

            for over_kind in over_array:
                print('\n')
                print(under_kind, '+', over_kind)

                #Over and under sampling params tunning
                best_params, best_value = best_under_over_parameters_combination(X_train, y_train_enc, xgb_params, under_kind, over_kind, subclass, metric=metric_name)
                under_params, over_params = best_params['under'], best_params['over']

                #Update params
                for key in list(under_params.keys()):
                    u_params[key]=under_params[key]

                undersampler_dict={'tl':under_sampling.TomekLinks(sampling_strategy=u_params['strategy'][1]),
                                'oss':under_sampling.OneSidedSelection(sampling_strategy=u_params['strategy'][1], n_neighbors=u_params['n_neighbors']),
                                'ncr':under_sampling.NeighbourhoodCleaningRule(sampling_strategy=u_params['strategy'][1], n_neighbors=u_params['n_neighbors'], kind_sel=u_params['kind_sel']),
                                'iht':under_sampling.InstanceHardnessThreshold(sampling_strategy=get_strategy(y_train_enc, u_params['strategy'][0], kind='under')),
                                'nm':under_sampling.NearMiss(sampling_strategy=get_strategy(y_train_enc, u_params['strategy'][0], kind='under'), n_neighbors=u_params['n_neighbors'], version=u_params['version']),
                                'rus':under_sampling.RandomUnderSampler(sampling_strategy=get_strategy(y_train_enc, u_params['strategy'][0], kind='under'))}

                #Apply undersampling method
                X_train_under, y_train_under = undersampler_dict[under_kind].fit_resample(X_train, y_train_enc)

                #Oversampling default params
                o_params = {'strategy':'minority', 'k_neighbors':3, 'n_neighbors':5, 'kind':'borderline-1'}
                #Update params
                for key in list(over_params.keys()):
                    o_params[key]=over_params[key]

                oversampler_dict = {'sm':over_sampling.SMOTE(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'), k_neighbors=o_params['k_neighbors']),
                            'ada':over_sampling.ADASYN(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'), n_neighbors=o_params['n_neighbors']),
                            'bsm':over_sampling.BorderlineSMOTE(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'), k_neighbors=o_params['k_neighbors'], kind=o_params['kind']),
                            'ksm':over_sampling.KMeansSMOTE(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'), k_neighbors=o_params['k_neighbors']),
                            'svm':over_sampling.SVMSMOTE(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'), k_neighbors=o_params['k_neighbors']),
                            'ros':over_sampling.RandomOverSampler(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'))}

                #Apply oversampling method
                X_train_over, y_train_over = oversampler_dict[over_kind].fit_resample(X_train_under, y_train_under)
                dtrain = xgb.DMatrix(data=X_train_over, label=pd.Series(y_train_over))
            
                #XGBoost train
                xgb_model = xgb_train(xgb_params, dtrain, dtest, feval=metric_name, verbose_eval=False)

                #Store model in dict
                model_dict[over_kind][i] = xgb_model
                params_dict[over_kind][i] = {'params':{'under':under_params, 'over':over_params}, 'score':best_value}

                #Save model and params
                pickle_dict = {'model_dict': model_dict[over_kind]}
                file_name = save_carpet + 'xgb_'+subclass.lower()+'_'+under_kind+'_'+over_kind+'_'+metric_name+'_model_CV_dict.pkl'
                pickle_save(file_name, pickle_dict)

                pickle_dict = {'params_dict': params_dict[over_kind]}
                file_name = parameters_carpet + 'xgb_'+subclass.lower()+'_'+under_kind+'_'+over_kind+'_'+metric_name+'_params_CV_dict.pkl'
                pickle_save(file_name, pickle_dict)
    i+=1

print("_____All Good_____")