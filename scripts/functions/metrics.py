import numpy as np
import matplotlib.pylab as plt
from sklearn import metrics
from preprocessing import ReverseEncoder, label_order
import itertools
from sklearn.preprocessing import OneHotEncoder
from scipy import interp
from itertools import cycle
from scipy.stats import t

def a_value(y_true, y_pred, zero_label=0, one_label=1):
    """
    Args:
        probabilities (list): A zipped list of the labels and the
            class probabilities in the form (m = # data instances):
             [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             ...
              (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             ]
        zero_label (optional, int): The label to use as the class '0'.
            Must be an integer, see above for details.
        one_label (optional, int): The label to use as the class '1'.
            Must be an integer, see above for details.
    Returns:
        The A-value as a floating point.
    """
    probabilities = []
    for i in range(len(y_true)): probabilities.append((y_true[i], y_pred[i]))
    
    # Obtain a list of the probabilities for the specified zero label class
    expanded_points = []
    for instance in probabilities:
        if instance[0] == zero_label or instance[0] == one_label:
            expanded_points.append((instance[0], instance[1][zero_label]))
    sorted_ranks = sorted(expanded_points, key=lambda x: x[1])

    n0, n1, sum_ranks = 0, 0, 0
    # Iterate through ranks and increment counters for overall count and ranks of class 0
    for index, point in enumerate(sorted_ranks):
        if point[0] == zero_label:
            n0 += 1
            sum_ranks += index + 1  # Add 1 as ranks are one-based
        elif point[0] == one_label:
            n1 += 1
        else:
            pass  # Not interested in this class

    return (sum_ranks - (n0*(n0+1)/2.0)) / float(n0 * n1)  # Eqn 3


def MAUC(y_true, y_pred):
    """
    Args:
        data (list): A zipped list (NOT A GENERATOR) of the labels and the
            class probabilities in the form (m = # data instances):
             [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             ...
              (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             ]
        num_classes (int): The number of classes in the dataset.
    Returns:
        The MAUC as a floating point value.
    """
    data = []
    for i in range(len(y_true)): data.append((y_true[i], y_pred[i]))
    num_classes = len(np.unique(y_true))
    # Find all pairwise comparisons of labels
    class_pairs = [x for x in itertools.combinations(range(num_classes), 2)]

    # Have to take average of A value with both classes acting as label 0 as this
    # gives different outputs for more than 2 classes
    sum_avals = 0
    for pairing in class_pairs:
        sum_avals += (a_value(y_true, y_pred, zero_label=pairing[0], one_label=pairing[1]) +
                      a_value(y_true, y_pred, zero_label=pairing[1], one_label=pairing[0])) / 2.0

    return sum_avals * (2 / float(num_classes * (num_classes-1)))  # Eqn 7

def MPR(preds,dtrain):
    for i in range(len(preds)):
        preds[i] = np.exp(preds[i])/np.sum(np.exp(preds[i]))
    #y_pred = np.array(np.argmax(np.array(preds),axis=1), ndmin=1)
    y_true = np.array(dtrain.get_label(), dtype=int, ndmin=1)
    ohe = OneHotEncoder(sparse=True)
    y_true_ohe=ohe.fit_transform(np.array(y_true, ndmin=2).reshape(-1, 1)).toarray()
    precision = dict()
    recall = dict()
    rev_precision = dict()
    rev_recall = dict()
    n_classes = len(np.unique(y_true))
    for i in range(n_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_true_ohe[:, i], preds[:, i])
        rev_precision[i] = np.flip(precision[i])
        rev_recall[i] = np.flip(recall[i])
    all_recall = np.unique(np.concatenate([rev_recall[i] for i in range(n_classes)]))
    mean_precision = np.zeros_like(all_recall)
    for i in range(n_classes):
        mean_precision += interp(all_recall, rev_recall[i], rev_precision[i])
    mean_precision /= n_classes
    score = metrics.auc(all_recall, mean_precision)
    return 'MPR', -score

def Gmean(y_true, y_pred):
    recall = metrics.recall_score(y_true, y_pred, average=None)
    precision = metrics.precision_score(y_true, y_pred, average=None)
    pr = recall * precision
    pr_sqrt = [value**0.5 for value in pr]
    score = np.mean(pr_sqrt)
    return score 

def MCC(preds, dtrain):
    for i in range(len(preds)):
        preds[i] = np.exp(preds[i])/np.sum(np.exp(preds[i]))
    y_pred = np.array(np.argmax(np.array(preds),axis=1), ndmin=1)
    y_true = np.array(dtrain.get_label(), dtype=int, ndmin=1)
    score = metrics.matthews_corrcoef(y_true, y_pred)
    return 'MCC', -score

def kappa(preds, dtrain):
    for i in range(len(preds)):
        preds[i] = np.exp(preds[i])/np.sum(np.exp(preds[i]))
    y_pred = np.array(np.argmax(np.array(preds),axis=1), ndmin=1)
    y_true = np.array(dtrain.get_label(), dtype=int, ndmin=1)
    score = metrics.cohen_kappa_score(y_true, y_pred)
    return 'kappa', -score

def f1(preds, dtrain):
    for i in range(len(preds)):
        preds[i] = np.exp(preds[i])/np.sum(np.exp(preds[i]))
    y_pred = np.array(np.argmax(np.array(preds),axis=1), ndmin=1)
    y_true = np.array(dtrain.get_label(), dtype=int, ndmin=1)
    score = metrics.f1_score(y_true, y_pred, average='macro')
    return 'f1', -score

def recall(preds, dtrain):
    for i in range(len(preds)):
        preds[i] = np.exp(preds[i])/np.sum(np.exp(preds[i]))
    y_pred = np.array(np.argmax(np.array(preds),axis=1), ndmin=1)
    y_true = np.array(dtrain.get_label(), dtype=int, ndmin=1)
    score = metrics.recall_score(y_true, y_pred, average='macro')
    return 'recall', -score

def min_recall(preds, dtrain):
    for i in range(len(preds)):
        preds[i] = np.exp(preds[i])/np.sum(np.exp(preds[i]))
    y_pred = np.array(np.argmax(np.array(preds),axis=1), ndmin=1)
    y_true = np.array(dtrain.get_label(), dtype=int, ndmin=1)
    score = np.min(metrics.recall_score(y_true, y_pred, average=None))
    return 'min_recall', -score

def wrecall(preds, dtrain):
    for i in range(len(preds)):
        preds[i] = np.exp(preds[i])/np.sum(np.exp(preds[i]))
    y_pred = np.array(np.argmax(np.array(preds),axis=1), ndmin=1)
    y_true = np.array(dtrain.get_label(), dtype=int, ndmin=1)
    score = metrics.recall_score(y_true, y_pred, average='weighted')
    return 'wrecall', -score

def MPAUC(preds, dtrain):
    for i in range(len(preds)):
        preds[i] = np.exp(preds[i])/np.sum(np.exp(preds[i]))
    y_pred = np.array(preds)
    y_true = np.array(dtrain.get_label(), dtype=int, ndmin=1)
    score = metrics.roc_auc_score(y_true, y_pred, multi_class='ovr') #ovo
    return 'MPAUC', -score

def MAvG(preds, dtrain):
    y_pred = np.array(np.argmax(np.array(preds),axis=1), ndmin=1)
    y_true = np.array(dtrain.get_label(), dtype=int, ndmin=1)
    recall = metrics.recall_score(y_true, y_pred, average=None)
    score = np.prod(recall)**(1/len(recall))
    return 'MAvG', -score

def IAM(preds, dtrain):
    y_pred = np.array(np.argmax(np.array(preds),axis=1), ndmin=1)
    y_true = np.array(dtrain.get_label(), dtype=int, ndmin=1)
    cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
    n_sum=0
    for i in range(len(cnf_matrix)):
        num = cnf_matrix[i,i] - max(sum(cnf_matrix[i,:])-cnf_matrix[i,i],sum(cnf_matrix[:,i])-cnf_matrix[i,i])
        den = max(sum(cnf_matrix[i,:]),sum(cnf_matrix[:,i]))
        n_sum+=num/den
    return 'IAM', -n_sum/len(cnf_matrix)

def CBA(preds, dtrain):
    y_pred = np.array(np.argmax(np.array(preds),axis=1), ndmin=1)
    y_true = np.array(dtrain.get_label(), dtype=int, ndmin=1)
    cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
    n_sum=0
    for i in range(len(cnf_matrix)):
        num = cnf_matrix[i,i]
        den = max(sum(cnf_matrix[i,:]),sum(cnf_matrix[:,i]))
        n_sum+=num/den
    return 'CBA', -n_sum/len(cnf_matrix)

def class_balance_accuracy(matrix):
    n_sum=0
    for i in range(len(matrix)):
        num = matrix[i,i]
        den = max(sum(matrix[i,:]),sum(matrix[:,i]))
        n_sum+=num/den
    return n_sum/len(matrix)

def imbalance_accuracy_metric(matrix):
    n_sum=0
    for i in range(len(matrix)):
        num = matrix[i,i] - max(sum(matrix[i,:])-matrix[i,i],sum(matrix[:,i])-matrix[i,i])
        den = max(sum(matrix[i,:]),sum(matrix[:,i]))
        n_sum+=num/den
    return n_sum/len(matrix)

def geometric_mean(y_true, y_pred):
    nclass = len(np.unique(y_true))
    class_final_index = np.argmax(y_pred,axis=1) #el index de la clase con mayor prob
    class_final_name = [list(range(nclass))[x] for x in class_final_index] # la clase con mayor prob
    recall = metrics.recall_score(y_true, class_final_name, average=None)
    return np.prod(recall)**(1/nclass)

def eval_metrics(y_true, y_pred, print_metrics=True):
    nclass = len(np.unique(y_true))
    class_final_index = np.argmax(y_pred,axis=1) #el index de la clase con mayor prob
    class_final_name = [list(range(nclass))[x] for x in class_final_index] # la clase con mayor prob
    cnf_matrix = metrics.confusion_matrix(ReverseEncoder(y_true), ReverseEncoder(class_final_name, len(np.unique(y_true))), labels=label_order(nclass))
    Metrics = {'MPAUC_OVA': metrics.roc_auc_score(y_true, y_pred, multi_class='ovr'),
               'MPAUC_OVO': metrics.roc_auc_score(y_true, y_pred, multi_class='ovo'),
               'MAUC': MAUC(y_true, y_pred),
               'precision': metrics.precision_score(y_true, class_final_name, average='macro'),
               'recall': metrics.recall_score(y_true, class_final_name, average='macro'),
               'f1': metrics.f1_score(y_true, class_final_name, average='macro'),
               'kappa': metrics.cohen_kappa_score(y_true, class_final_name),
               'CBA': class_balance_accuracy(cnf_matrix),
               'MAvG': np.prod(metrics.recall_score(y_true, class_final_name, average=None))**(1/nclass),
               'MCC': metrics.matthews_corrcoef(y_true, class_final_name),
               'min_recall': np.min(metrics.recall_score(y_true, class_final_name, average=None)),
               'Gmean': Gmean(y_true, class_final_name)}
    for key in list(Metrics.keys()):
        Metrics[key] = round(Metrics[key],3)
    if print_metrics:
        for key in list(Metrics.keys()):
            print('{}: {}'.format(key,Metrics[key]))
    return Metrics, cnf_matrix

def plot_confusion_matrix(cm, plot_name, normalize=True, title=None, cmap=plt.cm.Blues, plot=True):
    classes = label_order(len(cm))
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = np.round((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    if plot==True:
        print(cm)
    if plot==True:
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        #plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontsize = 17)
        plt.yticks(tick_marks, classes, fontsize = 17)
        #fmt = '.2f' if normalize else 'd'
        fmt =  'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "%d"%  (cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize = 16)
        plt.tight_layout()
        plt.ylabel('True label',fontsize = 18)
        plt.xlabel('Predicted label',fontsize = 18)
        plt.savefig(plot_name, bbox_inches='tight')
        #plt.close()
    return cm

def plot_confusion_matrix_perc(cm,cm_low,cm_high, classes, plot_name,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues,font=20,fig_x = 20, fig_y = 12):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm_aux = np.round((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100)
        cm_low = np.round((cm_low.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100)
        cm_high = np.round((cm_high.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100)
        cm = cm_aux
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    fig, ax = plt.subplots(figsize=(fig_x, fig_y))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize = font+1)
    plt.yticks(tick_marks, classes, fontsize = font+1)

    #fmt = '.2f' if normalize else 'd'
    fmt =  'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #plt.text(j, i, format(cm[i, j], fmt),
        if cm_high[i, j]>100: cm_high[i, j]=100
        plt.text(j, i,r"$%d^{+%d}_{-%d}$"%  (cm[i, j],cm_high[i, j]-cm[i, j],cm[i, j]-cm_low[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize = font)

    plt.tight_layout()
    plt.ylabel('True label', fontsize = font+2)
    plt.xlabel('Predicted label',fontsize = font+2)
    plt.savefig(plot_name, bbox_inches='tight')
    #plt.close()

def cnf_matrix_metrics(cnf_matrix):
    diagonal = np.diagonal(cnf_matrix)
    cnf_matrix_metrics = {'max': np.round(np.max(diagonal)),
                          'min': np.round(np.min(diagonal)),
                          'mean': np.round(np.mean(diagonal)),
                          'median': np.round(np.median(diagonal)),
                          'std': np.round(np.std(diagonal))}
    return cnf_matrix_metrics

def multi_roc_curve(y_true, y_pred, plot_macro=True, plot_classes=True, classes=None, title='ROC multiclass curves',
                     figsize=(8, 6), models_name=None, save=False, file_name=None, round=[2,2], legend_size=10):
    ohe = OneHotEncoder(sparse=True)
    y_true_ohe=ohe.fit_transform(np.array(y_true, ndmin=2).reshape(-1, 1)).toarray()
    # Compute ROC curve and ROC area for each class
    fpr = [dict() for _ in range(len(y_pred))]
    tpr = [dict() for _ in range(len(y_pred))]
    roc_auc = [dict() for _ in range(len(y_pred))]
    n_classes=len(np.unique(y_true))
    for j, pred in enumerate(y_pred):
        for i in range(n_classes):
            fpr[j][i], tpr[j][i], _ = metrics.roc_curve(y_true_ohe[:, i], pred[:, i])
            roc_auc[j][i] = metrics.auc(fpr[j][i], tpr[j][i])
    # First aggregate all false positive rates
    all_fpr = [np.unique(np.concatenate([fpr_model[i] for i in range(n_classes)])) for fpr_model in fpr]
    mean_tpr = [np.zeros_like(fpr_model) for fpr_model in all_fpr]
    # Then interpolate all ROC curves at this points
    for j, a_tpr in enumerate(all_fpr):
        for i in range(n_classes):
            mean_tpr[j] += interp(a_tpr, fpr[j][i], tpr[j][i])
    # Finally average it and compute AUC
    for j, m_tpr in enumerate(mean_tpr):
        mean_tpr[j] /= n_classes
        fpr[j]["macro"] = all_fpr[j]
        tpr[j]["macro"] = m_tpr
        roc_auc[j]["macro"] = metrics.auc(fpr[j]["macro"], tpr[j]["macro"])
    # Plot all ROC curves
    linestyles = ['dashed', 'dotted', 'dashdot']
    plt.figure(figsize=figsize, dpi=80)
    lw = 2
    if type(models_name)==type(None):
        models_name = ['model'+str(i) for i in range(len(y_pred))]
    if plot_macro:
        if round[0]==3:
            label_str = 'macro-avg {1} (AUC={0:0.3f})'
        else:
            label_str = 'macro-avg {1} (AUC={0:0.2f})'
        for i, name in enumerate(models_name):
            plt.plot(fpr[i]["macro"], tpr[i]["macro"],label=label_str.format(roc_auc[i]["macro"], name),
                     color='midnightblue', linestyle=linestyles[i], lw=lw)
    if plot_classes:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        labels = label_order(n_classes)
        if type(classes)==type(None):
            toplot_classes = range(n_classes)
        else:
            toplot_classes = classes
        if round[1]==3:
            label_str = '{0} {1} (AUC={2:0.3f})'
        else:
            label_str = '{0} {1} (AUC={2:0.2f})'
        for i, color in zip(toplot_classes, colors):
            for j, names in enumerate(models_name):
                if type(classes)!=type(None):
                    if len(classes)==1: color=colors[j]
                plt.plot(fpr[j][i], tpr[j][i], color=color, lw=1.5, linestyle=linestyles[j],
                label=label_str.format(names, labels[i], roc_auc[j][i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(fontsize=legend_size, loc="lower right")
    if save:
        if type(file_name)==type(None):
            file_name = ''
            for name in models_name: file_name+='_'+name
        save_file = 'images2/roc_'+file_name+'.png'
        plt.savefig(save_file)
    plt.show()

def pr_curve(y_true, y_pred, classes=None, title='PR multiclass curves', figsize=(8, 6), models_name=None,
             legend_size=10, legend_out=False, plot_micro=False, plot_macro=True, plot_classes=True, round=[2,2], save=False):
    ohe = OneHotEncoder(sparse=True)
    y_true_ohe=ohe.fit_transform(np.array(y_true, ndmin=2).reshape(-1, 1)).toarray()    
    precision = [dict() for _ in range(len(y_pred))]
    recall = [dict() for _ in range(len(y_pred))]
    rev_precision = [dict() for _ in range(len(y_pred))]
    rev_recall = [dict() for _ in range(len(y_pred))]
    average_precision = [dict() for _ in range(len(y_pred))]
    n_classes = len(np.unique(y_true))
    for j, pred in enumerate(y_pred):
        for i in range(n_classes):
            precision[j][i], recall[j][i], _ = metrics.precision_recall_curve(y_true_ohe[:, i], pred[:, i])
            average_precision[j][i] = metrics.average_precision_score(y_true_ohe[:, i], pred[:, i])
            rev_precision[j][i] = np.flip(precision[j][i])
            rev_recall[j][i] = np.flip(recall[j][i])
        precision[j]["micro"], recall[j]["micro"], _ = metrics.precision_recall_curve(y_true_ohe.ravel(), pred.ravel())
        average_precision[j]["micro"] = metrics.average_precision_score(y_true_ohe, pred, average="micro")
    # First aggregate all false positive rates
    all_recall = [np.unique(np.concatenate([recall_model[i] for i in range(n_classes)])) for recall_model in rev_recall]
    mean_precision = [np.zeros_like(recall_model) for recall_model in all_recall]
    # Then interpolate all ROC curves at this points
    for j, a_tpr in enumerate(all_recall):
        for i in range(n_classes):
            mean_precision[j] += interp(a_tpr, rev_recall[j][i], rev_precision[j][i])
    # Finally average it and compute AUC
    for j, m_tpr in enumerate(mean_precision):
        mean_precision[j] /= n_classes
        rev_recall[j]["macro"] = all_recall[j]
        rev_precision[j]["macro"] = m_tpr
        average_precision[j]["macro"] = metrics.auc(rev_recall[j]["macro"], rev_precision[j]["macro"])
    plt.figure(figsize=figsize, dpi=80)
    linestyles = ['dashed', 'dotted', 'dashdot']
    if type(models_name)==type(None):
        models_name = ['model'+str(i) for i in range(len(y_pred))]
    if plot_micro:
        if round[0]==3:
            label_str = 'micro-avg {1} (AUC={0:0.3f})'
        else:
            label_str = 'micro-avg {1} (AUC={0:0.2f})'
        for i, name in enumerate(models_name):
            plt.plot(recall[i]["micro"], precision[i]["micro"],label=label_str.format(average_precision[i]["micro"], name),
                     color='midnightblue', linestyle=linestyles[i], lw=2)
    if plot_macro:
        if round[0]==3:
            label_str = 'macro-avg {1} (AUC={0:0.3f})'
        else:
            label_str = 'macro-avg {1} (AUC={0:0.2f})'
        for i, name in enumerate(models_name):
            plt.plot(rev_recall[i]["macro"], rev_precision[i]["macro"],label=label_str.format(average_precision[i]["macro"], name),
                     color='midnightblue', linestyle=linestyles[i], lw=2)
    if plot_classes:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        labels = label_order(n_classes)
        if type(classes)==type(None):
            toplot_classes = range(n_classes)
        else:
            toplot_classes = classes
        if round[1]==3:
            label_str = '{0} {1} (AUC={2:0.3f})'
        else:
            label_str = '{0} {1} (AUC={2:0.2f})'
        for i, color in zip(toplot_classes, colors):
            for j, names in enumerate(models_name):
                if type(classes)!=type(None):
                    if len(classes)==1: color=colors[j]
                plt.plot(recall[j][i], precision[j][i], color=color, lw=1.5, linestyle=linestyles[j],
                label=label_str.format(names, labels[i], average_precision[j][i]))
    plt.xlabel("recall")
    plt.ylabel("precision")
    if legend_out:
        plt.legend(fontsize=legend_size, loc="best", bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(fontsize=legend_size, loc="best")
    plt.title(title)
    if save:
        file_name = ''
        for name in models_name: file_name+='_'+name
        save_file = 'images2/pr_'+file_name+'.png'
        plt.savefig(save_file)
    plt.show()

def t_student(score_1, score_2, n, n1, n2):
    diff = [y - x for y, x in zip(score_1, score_2)]
    #mean of differences
    d_bar = np.mean(diff)
    #variance of differences
    sigma2 = np.var(diff)
    #modified variance
    sigma2_mod = sigma2 * (1/n + n2/n1)
    #t_static
    t_static =  d_bar / np.sqrt(sigma2_mod)
    #p-value and plot the results 
    pvalue = ((1 - t.cdf(t_static, n-1))*200)
    return pvalue

def t_student2(score_1, score_2):
    diff = [np.abs(y - x) for y, x in zip(score_1, score_2)]
    d_bar = np.mean(diff)
    n = len(diff)
    t_value = d_bar/(np.std(diff)/np.sqrt(n))
    return t_value
    
def multi_roc_curve_dict(y_true, y_pred):
    ohe = OneHotEncoder(sparse=True)
    y_true_ohe=ohe.fit_transform(np.array(y_true, ndmin=2).reshape(-1, 1)).toarray()
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes=len(np.unique(y_true))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true_ohe[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    # Then interpolate all ROC curves at this points
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    return {'fpr':fpr, 'tpr':tpr, 'roc_auc':roc_auc}

def multi_roc_curve_nCV_dict(y_true: dict, y_pred: dict):
    n_classes = len(np.unique(y_true[0]))
    fpr, tpr = {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i] = [], []
    for i in list(y_true.keys()):
        roc_dict = multi_roc_curve_dict(y_true[i], y_pred[i])
        for key in list(fpr.keys()):
            fpr[key].append(roc_dict['fpr'][key])
            tpr[key].append(roc_dict['tpr'][key])
    #INTERPOLATE
    interp_fpr, interp_tpr, interp_roc_auc = {}, {}, {}
    #interpolate in class
    for i in range(n_classes):
        all_fpr = np.unique(np.concatenate([values for values in fpr[i]]))
        mean_tpr = np.zeros_like(all_fpr)
        for iter in range(len(fpr[i])):
            mean_tpr += interp(all_fpr, fpr[i][iter], tpr[i][iter])
        mean_tpr /= len(fpr[i])
        interp_fpr[i] = all_fpr
        interp_tpr[i] = mean_tpr
        interp_roc_auc[i] = metrics.auc(interp_fpr[i], interp_tpr[i])

    #interpolate between classes
    all_fpr = np.unique(np.concatenate([interp_fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, interp_fpr[i], interp_tpr[i])
    mean_tpr /= n_classes
    interp_fpr["macro"] = all_fpr
    interp_tpr["macro"] = mean_tpr
    interp_roc_auc["macro"] = metrics.auc(interp_fpr["macro"], interp_tpr["macro"])

    return {'fpr':interp_fpr, 'tpr':interp_tpr, 'roc_auc':interp_roc_auc}

def multi_pr_curve_dict(y_true, y_pred):
    ohe = OneHotEncoder(sparse=True)
    y_true_ohe=ohe.fit_transform(np.array(y_true, ndmin=2).reshape(-1, 1)).toarray()    
    precision = dict()
    recall = dict()
    rev_precision = dict()
    rev_recall = dict()
    average_precision = dict()
    n_classes = len(np.unique(y_true))
    for i in range(n_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_true_ohe[:, i], y_pred[:, i])
        average_precision[i] = metrics.average_precision_score(y_true_ohe[:, i], y_pred[:, i])
        rev_precision[i] = np.flip(precision[i])
        rev_recall[i] = np.flip(recall[i])
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_true_ohe.ravel(), y_pred.ravel())
    average_precision["micro"] = metrics.average_precision_score(y_true_ohe, y_pred, average="micro")
    # First aggregate all false positive rates
    all_recall = np.unique(np.concatenate([rev_recall[i] for i in range(n_classes)]))
    mean_precision = np.zeros_like(all_recall)
    # Then interpolate all ROC curves at this points
    for i in range(n_classes):
        mean_precision += interp(all_recall, rev_recall[i], rev_precision[i])
    # Finally average it and compute AUC
    mean_precision /= n_classes
    rev_recall["macro"] = all_recall
    rev_precision["macro"] = mean_precision
    average_precision["macro"] = metrics.auc(rev_recall["macro"], rev_precision["macro"])

    return {'average_precision':average_precision, 'rev_precision':rev_precision, 'rev_recall':rev_recall, 'precision':precision, 'recall':recall}

def multi_pr_curve_nCV_dict(y_true:dict, y_pred:dict):
    n_classes = len(np.unique(y_true[0]))
    precision, recall = {}, {}
    for i in range(n_classes):
        precision[i], recall[i] = [], []
    for i in list(y_true.keys()):
        pr_dict = multi_pr_curve_dict(y_true[i], y_pred[i])
        for key in list(precision.keys()):
            precision[key].append(pr_dict['precision'][key])
            recall[key].append(pr_dict['recall'][key])
    #INTERPOLATE
    interp_precision, interp_recall, interp_auc = {}, {}, {}
    #interpolate in class
    for i in range(n_classes):
        all_recall = np.unique(np.concatenate([np.flip(values) for values in recall[i]]))
        mean_precision = np.zeros_like(all_recall)
        for iter in range(len(recall[i])):
            mean_precision += interp(all_recall, np.flip(recall[i][iter]), np.flip(precision[i][iter]))
        mean_precision /= len(recall[i])
        interp_recall[i] = all_recall
        interp_precision[i] = mean_precision
        interp_auc[i] = metrics.auc(interp_recall[i], interp_precision[i])

    #interpolate between classes
    all_recall = np.unique(np.concatenate([interp_recall[i] for i in range(n_classes)]))
    mean_precision = np.zeros_like(all_recall)
    for i in range(n_classes):
        mean_precision += interp(all_recall, interp_recall[i], interp_precision[i])
    mean_precision /= n_classes
    interp_recall["macro"] = all_recall
    interp_precision["macro"] = mean_precision
    interp_auc["macro"] = metrics.auc(interp_recall["macro"], interp_precision["macro"])

    return {'recall':interp_recall, 'precision':interp_precision, 'auc':interp_auc}