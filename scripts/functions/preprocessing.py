import pandas as pd
import numpy as np

from sklearn.utils import class_weight

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter

def get_data2():
    X_hierarchical = pd.read_csv('files/features_LCC.csv', index_col='oid')
    labels = pd.read_csv('files/labels_LCC.csv',index_col='oid')
    Y_hierarchical = labels['class_hierachical']#.values
    Y_original = labels['class_original']#.values
    return X_hierarchical, Y_hierarchical, Y_original

def new_get_data2():
    X_hierarchical = pd.read_csv('files/new_old_features_LCC.csv', index_col='oid')
    labels = pd.read_csv('files/new_old_labels_LCC.csv',index_col='oid')
    Y_hierarchical = labels['class_hierachical']#.values
    Y_original = labels['class_original']#.values
    return X_hierarchical, Y_hierarchical, Y_original

def get_final_test():
    X_hierarchical = pd.read_csv('files/new_features_LCC.csv', index_col='oid')
    labels = pd.read_csv('files/new_labels_LCC.csv',index_col='oid')
    Y_hierarchical = labels['class_hierachical']#.values
    Y_original = labels['class_original']#.values
    return X_hierarchical, Y_hierarchical, Y_original

def new_get_final_test():
    X_hierarchical = pd.read_csv('files/new_new_features_LCC.csv', index_col='oid')
    labels = pd.read_csv('files/new_new_labels_LCC.csv',index_col='oid')
    Y_hierarchical = labels['class_hierachical']#.values
    Y_original = labels['class_original']#.values
    return X_hierarchical, Y_hierarchical, Y_original

def get_final_test_v2():
    print('v2')
    X_hierarchical = pd.read_csv('files/new_features_LCC_v2.csv', index_col='oid')
    labels = pd.read_csv('files/new_labels_LCC_v2.csv',index_col='oid')
    Y_hierarchical = labels['class_hierachical']#.values
    Y_original = labels['class_original']#.values
    return X_hierarchical, Y_hierarchical, Y_original

def class_data(X_train_hierarchical, X_test_hierarchical, y_train_hierarchical, y_test_hierarchical, y_train_original,
                 y_test_original, subclass):
    X_train = X_train_hierarchical.loc[y_train_hierarchical==subclass, :]
    y_train = y_train_original.loc[y_train_hierarchical==subclass]
    X_test = X_test_hierarchical.loc[y_test_hierarchical==subclass, :]
    y_test = y_test_original.loc[y_test_hierarchical==subclass]
    return X_train, y_train, X_test, y_test

def class_data_single(X, y_hierarchical, y_original, subclass):
    X_subclass = X.loc[y_hierarchical==subclass, :]
    y_subclass = y_original.loc[y_hierarchical==subclass]
    return X_subclass, y_subclass

def LabelEncoder(y_data):
    nclass = len(np.unique(y_data))
    LabelEncoder_dict = {
    4:{'SNIa':0,'SNII':1,'SNIbc':2,'SLSN':3},
    5:{'QSO':0,'AGN':1, 'YSO':2,'Blazar':3, 'CV/Nova':4},
    6:{'E':0,'RRL':1,'LPV':2,'Periodic-Other':3,'DSCT':4,'CEP':5},
    3:{'Transient':0, 'Stochastic':1, 'Periodic':2},
    15:{'SNIa':0,'SNII':1,'SNIbc':2,'SLSN':3,'QSO':4,'AGN':5, 'YSO':6,'Blazar':7, 'CV/Nova':8,'E':9,'RRL':10,'LPV':11,'Periodic-Other':12,'DSCT':13,'CEP':14}}
    y_data_encoded = [LabelEncoder_dict[nclass][y] for y in y_data]
    return y_data_encoded

def ReverseEncoder(y_data_encoded, nclass=None):
    if type(nclass)==type(None):
        nclass = len(np.unique(y_data_encoded))    
    ReverseEncoder_dict = {
    4:{0:'SNIa', 1:'SNII', 2:'SNIbc', 3:'SLSN'},
    5:{0:'QSO', 1:'AGN', 2:'YSO', 3:'Blazar', 4:'CV/Nova'},
    6:{0:'E', 1:'RRL', 2:'LPV', 3:'Periodic-Other', 4:'DSCT', 5:'CEP'},
    3: {0:'Transient', 1:'Stochastic', 2:'Periodic'},
    15:{0:'SNIa',1:'SNII',2:'SNIbc',3:'SLSN',4:'QSO',5:'AGN',6:'YSO',7:'Blazar',8:'CV/Nova',9:'E',10:'RRL',11:'LPV',12:'Periodic-Other',13:'DSCT',14:'CEP'}}
    y_data = [ReverseEncoder_dict[nclass][y] for y in y_data_encoded]
    return y_data

def weights_compute(y):
    weights_dict = {} #dict con los pesos calculados
    class_weights = list(class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y))
    for i in np.unique(y): 
        weights_dict[i] = class_weights[i]
    return weights_dict

def efective_weights(y, beta):
    weights_dict = {} #dict con los pesos calculados
    for i in np.unique(y):
        ny = np.count_nonzero(y == i)
        weights_dict[i] = (1-beta)/(1-beta**ny)
    return weights_dict

def label_order(nclass):
    LabelOrder={
        4:['SNIa','SNII','SNIbc','SLSN'],
        5:['QSO','AGN', 'YSO','Blazar', 'CV/Nova'],
        6:['E','RRL','LPV','Periodic-Other','DSCT','CEP'],
        3:['Periodic', 'Stochastic', 'Transient'],
        15:['SNIa','SNII','SNIbc','SLSN','QSO','AGN', 'YSO','Blazar', 'CV/Nova','E','RRL','LPV','Periodic-Other','DSCT','CEP']}
    return LabelOrder[nclass]

def resampler(X, y, N=1, strategy='ros'):
    if strategy=='ros':
        sampler = RandomOverSampler(sampling_strategy='all')
    else:
        sampler = RandomUnderSampler(sampling_strategy='all')  
    out_sampler_X, out_sampler_Y = [], []
    i=0
    while i<N:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        out_sampler_X.append(X_resampled_df)
        out_sampler_Y.append(y_resampled)
        i+=1        
    X_final_resampled = pd.concat(out_sampler_X)
    y_final_resampled=np.array(out_sampler_Y).flatten()
    return X_final_resampled, y_final_resampled

def ros_rus(X, y, rep_class=1):
    count1 = Counter(y)
    strategy = {i:count1[i] for i in range(rep_class+1)}
    for i in range(rep_class+1,len(count1)):
        strategy[i]=count1[rep_class]
    oversample = RandomOverSampler(sampling_strategy=strategy)
    X_ros, y_ros = oversample.fit_resample(X, y)
    X_resampled, y_resampled = resampler(X_ros, y_ros, strategy='rus')
    return X_resampled, y_resampled
    





