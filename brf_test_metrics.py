from functions.preprocessing import get_data2, LabelEncoder, class_data
from functions.custom_functions import pickle_load, save_excel
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from functions.metrics import eval_metrics, plot_confusion_matrix, plot_confusion_matrix_perc, cnf_matrix_metrics

X_hierarchical, Y_hierarchical, Y_original  = get_data2()

load_carpet = 'pickle2/models/'

clases_order_dict = {'Hierarchical':['Periodic', 'Stochastic', 'Transient'],
                    'Periodic':['E', 'RRL', 'LPV', 'Periodic-Other', 'DSCT', 'CEP'],
                    'Stochastic':['QSO', 'AGN', 'YSO', 'Blazar', 'CV/Nova'],
                    'Transient':['SNIa', 'SNII','SNIbc','SLSN']}

n_splits = 10
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=5)

for subclass in ['Hierarchical', 'Periodic', 'Stochastic', 'Transient']:
    print(subclass)

    #Empty objects for later
    cnf_array = []
    metrics_dict = {'precision':[], 'recall':[], 'f1':[],
                    'CBA':[], 'Gmean':[], 'MAUC':[], 'min_recall':[]}
    cnf_metrics_dict = {'max':[], 'min':[], 'mean':[], 'median':[], 'std':[]}
    #Load xgb models
    file_name = load_carpet + 'brf_'+subclass.lower()+'_model_CV_dict.pkl'
    brf_dict = pickle_load(file_name)['model_dict']
    i=0
    for train_index, test_index in sss.split(X_hierarchical, Y_original):
        print('{}/{}'.format(i+1,n_splits))
        X_train_hierarchical, X_test_hierarchical = X_hierarchical.iloc[train_index], X_hierarchical.iloc[test_index]
        y_train_hierarchical, y_test_hierarchical = Y_hierarchical[train_index], Y_hierarchical[test_index]
        y_train_original, y_test_original = Y_original[train_index], Y_original[test_index]

        if subclass == 'Hierarchical':
            X_test = X_test_hierarchical
            y_test_enc = LabelEncoder(y_test_hierarchical)
        else:
            X_train, y_train_og, X_test, y_test_og = class_data(X_train_hierarchical, X_test_hierarchical, y_train_hierarchical,
                                         y_test_hierarchical, y_train_original, y_test_original, subclass)
            y_test_enc = LabelEncoder(y_test_og)
            
        #Prediction
        brf_model = brf_dict[i]

        #Prediction & Evaluate performance
        y_pred = brf_model.predict_proba(X_test)
        Metrics, cnf_matrix = eval_metrics(y_test_enc, y_pred, print_metrics=False)
        norm_cnf_metrics = plot_confusion_matrix(cnf_matrix, '', plot=False)
        cnf_metrics = cnf_matrix_metrics(norm_cnf_metrics)

        #Store metrics
        cnf_array.append(norm_cnf_metrics)
        for key in list(metrics_dict.keys()):
            metrics_dict[key].append(Metrics[key])

        for key in list(cnf_metrics_dict.keys()):
            cnf_metrics_dict[key].append(cnf_metrics[key])
            
        i+=1

        #Back to outer

    save_excel("excel_metrics/CV_metrics.xlsx", 0, 'BRF', subclass, 'X', 'X',
                np.mean(metrics_dict['precision']), np.std(metrics_dict['precision']),
                np.mean(metrics_dict['recall']), np.std(metrics_dict['recall']),
                np.mean(metrics_dict['f1']), np.std(metrics_dict['f1']),
                np.mean(metrics_dict['CBA']), np.std(metrics_dict['CBA']),
                np.mean(metrics_dict['Gmean']), np.std(metrics_dict['Gmean']),
                np.mean(metrics_dict['MAUC']), np.std(metrics_dict['MAUC']),
                np.mean(metrics_dict['min_recall']), np.std(metrics_dict['min_recall']),
                np.mean(cnf_metrics_dict['max']), np.mean(cnf_metrics_dict['min']), np.mean(cnf_metrics_dict['mean']),
                np.mean(cnf_metrics_dict['median']), np.mean(cnf_metrics_dict['std']))

    cnf_file = 'images2/conf_matrix_CV_brf_'+subclass.lower()+'.png'
    plot_confusion_matrix_perc(np.median(cnf_array,axis=0), np.percentile(cnf_array,5,axis=0),np.percentile(cnf_array,95,axis=0),clases_order_dict[subclass],cnf_file,font=21, fig_x = 10, fig_y = 8)

print("_____All Good_____")

