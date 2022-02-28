from functions.preprocessing import get_data2, LabelEncoder, class_data, new_get_data2
from functions.custom_functions import pickle_load, save_excel
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from functions.metrics import eval_metrics, plot_confusion_matrix, plot_confusion_matrix_perc, cnf_matrix_metrics

X_hierarchical, Y_hierarchical, Y_original  = get_data2()
X_hierarchical_new, Y_hierarchical_new, Y_original_new  = new_get_data2()

subclass = input('subclass? {Hierarchical, Periodic, Stochastic, Transient}:\n')
print(subclass)

metric_name_array = ['recall'] #['recall','CBA']

method_array = ['weight'] #['weight', 'sampling', 'FL_weight']

load_carpet = 'pickle2/models/'

clases_order_dict = {'Hierarchical':['Periodic', 'Stochastic', 'Transient'],
                    'Periodic':['E', 'RRL', 'LPV', 'Periodic-Other', 'DSCT', 'CEP'],
                    'Stochastic':['QSO', 'AGN', 'YSO', 'Blazar', 'CV/Nova'],
                    'Transient':['SNIa', 'SNII','SNIbc','SLSN']}

n_splits = 10
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=5)

for method in method_array:
    for metric_name in metric_name_array:
        method_metric = method+'_'+metric_name
        print(method_metric)

        #Empty objects for later
        cnf_array = []
        metrics_dict = {'precision':[], 'recall':[], 'f1':[],
                    'CBA':[], 'Gmean':[], 'MAUC':[], 'min_recall':[]}
        cnf_metrics_dict = {'max':[], 'min':[], 'mean':[], 'median':[], 'std':[]}
        #Load xgb models
        file_name = load_carpet + 'xgb_FT_'+subclass.lower()+'_'+method_metric+'_model_CV_dict.pkl'
        xgb_dict = pickle_load(file_name)['model_dict']
        i=0
        for train_index, test_index in sss.split(X_hierarchical, Y_original):
            print('{}/{}'.format(i+1,n_splits))
            X_train_hierarchical_old, X_test_hierarchical_old = X_hierarchical.iloc[train_index], X_hierarchical.iloc[test_index]
            train_index_new, test_index_new = X_hierarchical_new.index.intersection(X_train_hierarchical_old.index), X_hierarchical_new.index.intersection(X_test_hierarchical_old.index)

            X_train_hierarchical, X_test_hierarchical = X_hierarchical_new.loc[train_index_new], X_hierarchical_new.loc[test_index_new]
            y_train_hierarchical, y_test_hierarchical = Y_hierarchical_new[train_index_new], Y_hierarchical_new[test_index_new]
            y_train_original, y_test_original = Y_original_new[train_index_new], Y_original_new[test_index_new]
            
            if subclass == 'Hierarchical':
                X_test = X_test_hierarchical
                y_test_enc = LabelEncoder(y_test_hierarchical)
            else:
                X_train, y_train_og, X_test, y_test_og = class_data(X_train_hierarchical, X_test_hierarchical, y_train_hierarchical,
                                         y_test_hierarchical, y_train_original, y_test_original, subclass)
                y_test_enc = LabelEncoder(y_test_og)
            dtest = xgb.DMatrix(data=X_test, label=pd.Series(y_test_enc))
            
            #Prediction
            xgb_model = xgb_dict[i]

            #Prediction & Evaluate performance
            y_pred = xgb_model.predict(dtest)
            y_true = np.array(dtest.get_label(), dtype=int, ndmin=1)
            Metrics, cnf_matrix = eval_metrics(y_true, y_pred, print_metrics=False)
            norm_cnf_metrics = plot_confusion_matrix(cnf_matrix, '', plot=False, normalize=True)
            cnf_metrics = cnf_matrix_metrics(norm_cnf_metrics)

            #Store metrics
            cnf_array.append(norm_cnf_metrics)
            for key in list(metrics_dict.keys()):
                metrics_dict[key].append(Metrics[key])

            for key in list(cnf_metrics_dict.keys()):
                cnf_metrics_dict[key].append(cnf_metrics[key])
            
            i+=1

            #Back to outer
 
        save_excel("excel_metrics/CV_metrics2.xlsx", 0, 'XGB FT', subclass, method, metric_name,
                np.mean(metrics_dict['precision']), np.std(metrics_dict['precision']),
                np.mean(metrics_dict['recall']), np.std(metrics_dict['recall']),
                np.mean(metrics_dict['f1']), np.std(metrics_dict['f1']),
                np.mean(metrics_dict['min_recall']), np.std(metrics_dict['min_recall']),
                np.mean(cnf_metrics_dict['max']), np.mean(cnf_metrics_dict['min']), np.mean(cnf_metrics_dict['mean']),
                np.mean(cnf_metrics_dict['median']), np.mean(cnf_metrics_dict['std']))
        
        cnf_file = 'images2/conf_matrix_CV_xgb_FT_'+subclass.lower()+'_'+method_metric+'.png'
        plot_confusion_matrix_perc(np.median(cnf_array,axis=0), np.percentile(cnf_array,5,axis=0),np.percentile(cnf_array,95,axis=0),clases_order_dict[subclass],cnf_file,font=21, fig_x = 10, fig_y = 8, normalize=True)

print("_____All Good_____")
 
