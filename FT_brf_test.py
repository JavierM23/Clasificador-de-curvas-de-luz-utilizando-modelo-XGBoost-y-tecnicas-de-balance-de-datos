from functions.preprocessing import new_get_final_test, LabelEncoder, class_data_single
from functions.custom_functions import pickle_load, save_excel
import numpy as np
from functions.metrics import eval_metrics, plot_confusion_matrix, plot_confusion_matrix_perc, cnf_matrix_metrics

X_test_hierarchical, y_test_hierarchical, y_test_original  = new_get_final_test()

subclass = input('subclass? {Hierarchical, Periodic, Stochastic, Transient}:\n')
print(subclass)

load_carpet = 'pickle2/models/FT/'

clases_order_dict = {'Hierarchical':['Periodic', 'Stochastic', 'Transient'],
                    'Periodic':['E', 'RRL', 'LPV', 'Periodic-Other', 'DSCT', 'CEP'],
                    'Stochastic':['QSO', 'AGN', 'YSO', 'Blazar', 'CV/Nova'],
                    'Transient':['SNIa', 'SNII','SNIbc','SLSN']}

n_splits = 10
#Empty objects for later
cnf_array = []
metrics_dict = {'precision':[], 'recall':[], 'f1':[],
                    'CBA':[], 'Gmean':[], 'MAUC':[], 'min_recall':[]}
cnf_metrics_dict = {'max':[], 'min':[], 'mean':[], 'median':[], 'std':[]}
#Load xgb models
file_name = load_carpet + 'brf_FT_'+subclass.lower()+'_model_CV_dict.pkl'
brf_dict = pickle_load(file_name)['model_dict']

for i in range(n_splits):
    print(i)
    if subclass == 'Hierarchical':
        X_test = X_test_hierarchical
        y_test_enc = LabelEncoder(y_test_hierarchical)
    else:
        X_test, y_test_og = class_data_single(X_test_hierarchical, y_test_hierarchical, y_test_original, subclass)
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
            
#Back to outer

save_excel("metrics/CV_metrics.xlsx", 0, 'BRF FT', subclass, '', '',
                np.mean(metrics_dict['precision']), np.std(metrics_dict['precision']),
                np.mean(metrics_dict['recall']), np.std(metrics_dict['recall']),
                np.mean(metrics_dict['f1']), np.std(metrics_dict['f1']),
                np.mean(metrics_dict['CBA']), np.std(metrics_dict['CBA']),
                np.mean(metrics_dict['Gmean']), np.std(metrics_dict['Gmean']),
                np.mean(metrics_dict['MAUC']), np.std(metrics_dict['MAUC']),
                np.mean(metrics_dict['min_recall']), np.std(metrics_dict['min_recall']),
                np.mean(cnf_metrics_dict['max']), np.mean(cnf_metrics_dict['min']), np.mean(cnf_metrics_dict['mean']),
                np.mean(cnf_metrics_dict['median']), np.mean(cnf_metrics_dict['std']))
cnf_file = 'images2/conf_matrix_CV_brf_FT_'+subclass.lower()+'.png'
plot_confusion_matrix_perc(np.median(cnf_array,axis=0), np.percentile(cnf_array,5,axis=0),np.percentile(cnf_array,95,axis=0),clases_order_dict[subclass],cnf_file,font=21, fig_x = 10, fig_y = 8)

print("_____All Good_____")

