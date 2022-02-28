from functions.preprocessing import get_data2, LabelEncoder, class_data
from functions.custom_functions import prediction_union, pickle_save
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.ensemble import BalancedRandomForestClassifier as RandomForestClassifier
from functions.metrics import eval_metrics, plot_confusion_matrix
import numpy as np

model_dict = {'Hierarchical':{}, 'Periodic':{}, 'Stochastic':{}, 'Transient':{}}

cnf_array_dict = {'Hierarchical':[], 'Periodic':[], 'Stochastic':[], 'Transient':[], 'LC':[]}
metrics_dict = {'LC':{'precision':[], 'recall':[], 'F1':[], 'CBA':[], 'MCC':[]}}

carpet = 'pickle2/models/'

X_hierarchical, Y_hierarchical, Y_original  = get_data2()

n_splits = 10
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=5)
i=0

#Outer layer
for train_index, test_index in sss.split(X_hierarchical, Y_original):
    print(train_index[:10])
    print('{}/{}'.format(i+1,n_splits))
    X_train_hierarchical, X_test_hierarchical = X_hierarchical.iloc[train_index], X_hierarchical.iloc[test_index]
    y_train_hierarchical, y_test_hierarchical = Y_hierarchical[train_index], Y_hierarchical[test_index]
    y_train_original, y_test_original = Y_original[train_index], Y_original[test_index]

    y_predictions = []

    rf_model_hierarchical = RandomForestClassifier(
                n_estimators=500,
                max_features='auto',
                max_depth=None,
                n_jobs=-1,
                bootstrap=True,
                class_weight='balanced_subsample',
                criterion='entropy',
                min_samples_split=2,
                min_samples_leaf=1)

    rf_model_periodic = RandomForestClassifier(
            n_estimators=500,
            max_features='auto',
            max_depth=None,
            n_jobs=-1,
            class_weight='balanced_subsample',
            #bootstrap=False,
            criterion='entropy',
            min_samples_split=2,
            min_samples_leaf=1)

    rf_model_stochastic = RandomForestClassifier(
            n_estimators=500,
            max_features=0.2,#'auto',
            max_depth=None,
            n_jobs=-1,
            bootstrap=True,
            class_weight='balanced_subsample',
            criterion='entropy',
            min_samples_split=2,
            min_samples_leaf=1)

    rf_model_transient = RandomForestClassifier(
            n_estimators=500,
            max_features='auto',
            max_depth=None,
            n_jobs=-1,
            bootstrap=True,
            class_weight='balanced_subsample',
            criterion='entropy',
            min_samples_split=2,
            min_samples_leaf=1)

    rf_dict={'Hierarchical':rf_model_hierarchical, 'Periodic':rf_model_periodic, 'Stochastic':rf_model_stochastic, 'Transient':rf_model_transient}

    for subclass in ['Hierarchical','Periodic','Stochastic','Transient']:
        print(subclass)
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

        #Train
        rf_dict[subclass].fit(X_train, y_train_enc)
        
        #Store model in dict
        model_dict[subclass][i] = rf_dict[subclass]

    i+=1

#Save model and params
for subclass in ['Hierarchical','Periodic','Stochastic','Transient']:
    pickle_dict = {'model_dict': model_dict[subclass]}
    file_name = carpet + 'brf_'+subclass.lower()+'_model_CV_dict2.pkl'
    pickle_save(file_name, pickle_dict)

print("_____All Good_____")