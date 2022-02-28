from collections import Counter
from imblearn import under_sampling, over_sampling
import numpy as np

def get_strategy(y, rep_class, kind: str = 'over'):
    count = Counter(y)
    if kind == 'over':
        strategy = {i:count[i] for i in range(rep_class+1)}
        for i in range(rep_class+1,len(count)):
            strategy[i]=count[rep_class]
    elif kind == 'under':
        strategy = {i:count[rep_class] for i in range(rep_class+1)}
        for i in range(rep_class+1,len(count)):
            strategy[i]=count[i]
    else:
        raise Warning("kind = {'over', 'under'}")
    return strategy

def f_undersampler(X, y, fn, strategy, **kwargs):

    """
    fn: 'cnn': CondensedNearestNeighbour([list], n_neighbors:(int)=None),
        'enn': EditedNearestNeighbours([list], n_neighbors:(int)=3, kind_sel:{‘all’, ‘mode’}='all')),
        'renn': RepeatedEditedNearestNeighbours([list], n_neighbors:(int)=3, max_iter:(int)=100, kind_sel:{‘all’, ‘mode’}='all'),
        'allknn': AllKNN([list], n_neighbors:(int)=3, kind_sel:{‘all’, ‘mode’}='all', allow_minority:(bool)=False),
        'ncr': NeighbourhoodCleaningRule([list], n_neighbors:(int)=3, kind_sel:{‘all’, ‘mode’}='all'), threshold_cleaning:(float)=0.5),
        'oss': OneSidedSelection([list], n_neighbors:(int)=3, n_seeds_S:(int)=1),
        'tl': TomekLinks([list]),
        'iht': InstanceHardnessThreshold([dict], estimator:(str)=None, cv:(int)=5),
        'nm': NearMiss([dict], n_neighbors:(int)=None, version:(int)=1, n_neighbors_ver3:(int)=3),
        'rus': RandomUnderSampler([dict], replacement:(bool)=False)
        
    }
    """
    
    


    fn_dict = {'cnn': under_sampling.CondensedNearestNeighbour,
        'enn': under_sampling.EditedNearestNeighbours,
        'renn': under_sampling.RepeatedEditedNearestNeighbours,
        'allknn': under_sampling.AllKNN,
        'iht': under_sampling.InstanceHardnessThreshold,
        'nm': under_sampling.NearMiss,
        'ncr':under_sampling.NeighbourhoodCleaningRule,
        'oss': under_sampling.OneSidedSelection,
        'rus': under_sampling.RandomUnderSampler,
        'tl': under_sampling.TomekLinks}

    method = fn_dict[fn]

    xx = method(sampling_strategy = strategy, **kwargs)
    X_train_us, y_train_us = xx.fit_resample(X, y)

    return X_train_us, y_train_us

def f_oversampler(X, y, fn, strategy, **kwargs):

    """
    fn: 'ros': RandomOverSampler([dict]),
        'sm': SMOTE([dict], k_neighbors:(int)=5)
        'ada': ADASYN([dict], n_neighbors:(int)=5)
        'bsm': BorderlineSMOTE([dict], k_neighbors:(int)=5, m_neighbors:(int)=5, kind:{“borderline-1”, “borderline-2”}=’borderline-1’),
        'ksm': KMeansSMOTE([dict], k_neighbors:(int)=5, kmeans_estimator:(int)=None),
        'svm': SVMSMOTE([dict], k_neighbors:(int)=5, m_neighbors:(int)=10,)
    }
    """

    fn_dict = {'ros': over_sampling.RandomOverSampler,
        'sm': over_sampling.SMOTE ,
        'ada': over_sampling.ADASYN,
        'bsm': over_sampling.BorderlineSMOTE,
        'ksm': over_sampling.KMeansSMOTE,
        'svm': over_sampling.SVMSMOTE}

    method = fn_dict[fn]

    xx = method(sampling_strategy = strategy, **kwargs)
    X_train_os, y_train_os = xx.fit_resample(X, y)

    return X_train_os, y_train_os

