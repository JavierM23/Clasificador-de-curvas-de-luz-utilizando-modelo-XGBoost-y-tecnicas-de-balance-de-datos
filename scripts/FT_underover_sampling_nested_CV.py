from functions.hyper_selec import get_best_optunaparameters
from functions.preprocessing import get_data2, new_get_data2, LabelEncoder, class_data
from functions.custom_functions import get_strategy, xgb_train, pickle_save
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from imblearn import under_sampling, over_sampling

X_hierarchical, Y_hierarchical, Y_original  = get_data2()
X_hierarchical_new, Y_hierarchical_new, Y_original_new  = new_get_data2()

#subclass = input('subclass? {Hierarchical, Periodic, Stochastic, Transient}:\n')
subclass = 'Periodic'
print(subclass)

metric_name = 'recall'

#under_kind = input('under method? [allknn, cnn, iht, nm]\n')
under_kind = 'cnn'
print(under_kind)

#over_kind = input('over method? [ada, bsm, ros, sm, svm]\n')
#over_kind = 'ada'
#print(under_kind)


#cluster = bool(int(input('cluster? {T -> 1, F -> 0}:\n')))
cluster = 'True'
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

#over_array_dict = {'Periodic': ['ada', 'bsm', 'ros', 'sm', 'svm'], 'Stochastic': ['ada', 'bsm', 'ros', 'sm', 'svm'], 'Transient': ['ros', 'sm', 'svm']}
over_array_dict = {'Periodic': ['ros'], 'Stochastic': ['ada', 'bsm', 'ros', 'sm', 'svm'], 'Transient': ['ros', 'sm', 'svm']}




#iteration = input('iteration?\n')
iteration = ''
if iteration=='':
    iteration = 0
    model_dict = {}
    for key in over_array_dict[subclass]:
        model_dict[key] = {}
else:
    iteration = int(iteration)-1
    #model_pickle = pickle_load(save_carpet + 'xgb_'+subclass.lower()+'_'+under_kind+'_'+over_kind+'_'+metric_name+'_model_CV_dict2.pkl')
    #model_dict = model_pickle['model_dict']

n_splits = 10
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=5)

i=0
#Outer layer
for train_index, test_index in sss.split(X_hierarchical, Y_original):
    print('\n')
    print('{}/{}'.format(i+1,n_splits))

    if i>=iteration:
        #Shuffle data
        X_train_hierarchical_old, X_test_hierarchical_old = X_hierarchical.iloc[train_index], X_hierarchical.iloc[test_index]
        train_index_new, test_index_new = X_hierarchical_new.index.intersection(X_train_hierarchical_old.index), X_hierarchical_new.index.intersection(X_test_hierarchical_old.index)

        X_train_hierarchical, X_test_hierarchical = X_hierarchical_new.loc[train_index_new], X_hierarchical_new.loc[test_index_new]
        y_train_hierarchical, y_test_hierarchical = Y_hierarchical_new[train_index_new], Y_hierarchical_new[test_index_new]
        y_train_original, y_test_original = Y_original_new[train_index_new], Y_original_new[test_index_new]

        #Classsifier data
        X_train, y_train_og, X_test, y_test_og = class_data(X_train_hierarchical, X_test_hierarchical, y_train_hierarchical,
        y_test_hierarchical, y_train_original, y_test_original, subclass)
        y_train_enc = LabelEncoder(y_train_og)
        y_test_enc = LabelEncoder(y_test_og)
        dtest = xgb.DMatrix(data=X_test, label=pd.Series(y_test_enc))

        #UNDERSAMPLING
        under_params = {'Periodic':{'strategy':(2,[0,1]), 'n_neighbors':3, 'kind_sel':'all', 'version':2},
                        'Stochastic':{'strategy':(1,[0]), 'n_neighbors':5, 'kind_sel':'all','version':2},
                        'Transient':{'strategy':(1,[0]), 'n_neighbors':5, 'kind_sel':'all', 'version':2}}
        u_params = under_params[subclass]

        undersampler_dict={'cnn':under_sampling.CondensedNearestNeighbour(sampling_strategy=u_params['strategy'][1], n_neighbors=u_params['n_neighbors']),
                               'allknn':under_sampling.AllKNN(sampling_strategy=u_params['strategy'][1], n_neighbors=u_params['n_neighbors'], kind_sel=u_params['kind_sel']),
                               'iht':under_sampling.InstanceHardnessThreshold(estimator='gradient-boosting', sampling_strategy=get_strategy(y_train_enc, u_params['strategy'][0], kind='under')),
                                'nm':under_sampling.NearMiss(sampling_strategy=get_strategy(y_train_enc, u_params['strategy'][0], kind='under'), n_neighbors=u_params['n_neighbors'], version=u_params['version'])}
            
        X_train_under, y_train_under = undersampler_dict[under_kind].fit_resample(X_train, y_train_enc)

        #OVERSAMPLING

        
        for over_kind in over_array_dict[subclass]:
            over_params = {'Periodic':{'strategy':(2,[0,1]), 'k_neighbors':5, 'n_neighbors':5, 'kind':'borderline-2'},
                    'Stochastic':{'strategy':(1,[0]), 'k_neighbors':5, 'n_neighbors':5, 'kind':'borderline-2'},
                    'Transient':{'strategy':(1,[0]), 'k_neighbors':3, 'n_neighbors':3, 'kind':'borderline-2'}}
            o_params = over_params[subclass]

            oversampler_dict = {'sm':over_sampling.SMOTE(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'), k_neighbors=o_params['k_neighbors']),
                            'ada':over_sampling.ADASYN(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'), n_neighbors=o_params['n_neighbors']),
                            'bsm':over_sampling.BorderlineSMOTE(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'), k_neighbors=o_params['k_neighbors'], kind=o_params['kind']),
                            'svm':over_sampling.SVMSMOTE(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'), k_neighbors=o_params['k_neighbors']),
                            'ros':over_sampling.RandomOverSampler(sampling_strategy=get_strategy(y_train_under, o_params['strategy'][0], kind='over'))}

            #Apply oversampling method
            X_train_over, y_train_over = oversampler_dict[over_kind].fit_resample(X_train_under, y_train_under)
            dtrain = xgb.DMatrix(data=X_train_over, label=pd.Series(y_train_over))

            if subclass == 'Hierarchical': n_evals=8
            else: n_evals=12

            #Get best parameters
            pbar = tqdm(total=n_evals, desc="Optuna")
            xgb_params = get_best_optunaparameters(dtrain, metric=metric_name, pbar=pbar, n_evals=n_evals)
    
            #XGBoost train
            xgb_model = xgb_train(xgb_params, dtrain, dtest, feval=metric_name, verbose_eval=False)

            #Store model in dict
            model_dict[over_kind][i] = xgb_model

            #Save model and params
            pickle_dict = {'model_dict': model_dict[over_kind]}
            file_name = save_carpet + 'xgb_FT_'+subclass.lower()+'_'+under_kind+'_'+over_kind+'_'+metric_name+'_model_CV_dict.pkl'
            pickle_save(file_name, pickle_dict)

    i+=1

print("_____All Good_____")
