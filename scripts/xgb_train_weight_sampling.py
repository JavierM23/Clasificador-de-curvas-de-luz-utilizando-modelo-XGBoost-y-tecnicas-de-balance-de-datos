from functions.hyper_selec import get_best_optunaparameters
from functions.preprocessing import get_data2, LabelEncoder, class_data, weights_compute
from functions.custom_functions import xgb_train, pickle_save, pickle_load, get_strategy
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from imblearn import under_sampling, over_sampling

X_hierarchical, Y_hierarchical, Y_original  = get_data2()

subclass = input('subclass? {Periodic, Stochastic, Transient}:\n')
#subclass = 'Stochastic'
print(subclass)

metric_name = 'recall'

under_kind = input('under method? [rus, allknn, cnn, iht, nm]\n')
#under_kind = 'iht'
print(under_kind)


cluster = bool(int(input('cluster? {T -> 1, F -> 0}:\n')))
#cluster = True
print(cluster)

if cluster:
    save_carpet = '/home/jmolina/carpet/to_transfer/pickle/models/'
else:
    save_carpet = 'pickle2/models/sampling/weight/'

over_array_dict = {'Periodic': ['ada', 'ros', 'sm'],
                    'Stochastic': ['ada', 'ros', 'sm'],
                    'Transient': ['ada', 'ros', 'sm']}    

iteration = input('iteration?\n')
#iteration = ''
if iteration=='':
    iteration = 0
    model_dict = {}
    for key in over_array_dict[subclass]:
        model_dict[key] = {}
    model_iteration = 0
else:
    iteration = int(iteration)-1
    model_dict = {}
    over_array = ['ada','ros','sm']
    for over_kind in over_array:
        model_pickle = pickle_load(save_carpet + 'xgb_'+subclass.lower()+'_weight_'+under_kind+'_'+over_kind+'_'+metric_name+'_model_CV_dict.pkl')
        model_dict[over_kind] = model_pickle['model_dict']
    model_iteration = int(input('[ada-0, ros-1, sm-2]?\n'))

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
        weights_dict = weights_compute(LabelEncoder(Y_original[Y_hierarchical==subclass].values))
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
                            'iht':under_sampling.InstanceHardnessThreshold(sampling_strategy=get_strategy(y_train_enc, u_params['strategy'][0], kind='under')),
                            'nm':under_sampling.NearMiss(sampling_strategy=get_strategy(y_train_enc, u_params['strategy'][0], kind='under'), n_neighbors=u_params['n_neighbors'], version=u_params['version']),
                            'rus':under_sampling.RandomUnderSampler(sampling_strategy=get_strategy(y_train_enc, u_params['strategy'][0], kind='under'))}
            
        X_train_under, y_train_under = undersampler_dict[under_kind].fit_resample(X_train, y_train_enc)

        #OVERSAMPLING

        if i==iteration:
            over_array=over_array_dict[subclass][model_iteration:]
        else:
            over_array = over_array_dict[subclass]

        for over_kind in over_array:
            print(over_kind)
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
            dtrain = xgb.DMatrix(data=X_train_over, label=pd.Series(y_train_over), weight=pd.Series(y_train_over).map(lambda x: weights_dict[x]))

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
            file_name = save_carpet + 'xgb_'+subclass.lower()+'_weight_'+under_kind+'_'+over_kind+'_'+metric_name+'_model_CV_dict.pkl'
            pickle_save(file_name, pickle_dict)

    i+=1

print("_____All Good_____")
    
