import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# auc roc
def roc_metric(y_true, y_scores, path):
    '''
    :param y_true: array-like of shape = (n_samples) or (n_samples, n_outputs)
    Ground truth (correct) target values.
    :param y_scores: array-like of shape = (n_samples) or (n_samples, n_outputs)
    Estimated target values.
    :param path: if path equal 0, show ks plot; if path equal string of filepath, ks plot will
    save to filepath.
    :return: roc value
    '''
    fpr, tpr, threshold = roc_curve(y_true, y_scores)
    auc_value = auc(fpr, tpr)

    # plot
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = %0.2f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    if path == 0:
        plt.show()
    else:
        plt.savefig(path, dpi=150)
    return auc_value

# KS curve
from scipy import stats
def ks_metric(y_true, y_scores, bins, path):
    '''
    :param y_true: array-like of shape = (n_samples) or (n_samples, n_outputs)
    Ground truth (correct) target values.
    :param y_scores: array-like of shape = (n_samples) or (n_samples, n_outputs)
    Estimated target values.
    :param bins: bins of y_scores
    :param path: if path equal 0, show ks plot; if path equal string of filepath, ks plot will
    save to filepath.
    :return: ks value
    '''
    df = pd.DataFrame({'y': y_true,
                       'score': y_scores})
    cdf_data1 = df[df['y'] == 0]['score']
    cdf_data2 = df[df['y'] == 1]['score']
    cdf1 = stats.cumfreq(cdf_data1, numbins=bins)
    cdf2 = stats.cumfreq(cdf_data2, numbins=bins)
    y_0 = cdf1[0] / cdf1[0][-1]
    y_1 = cdf2[0] / cdf2[0][-1]
    cdf_data = pd.DataFrame({'y_0': y_0, 'y_1': y_1})

    # plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(cdf_data)
    ax.legend(list(cdf_data.columns))
    plt.ylabel('累积概率')
    plt.xlabel('预测得分')
    if path == 0:
        plt.show()
    else:
        plt.savefig(path, dpi=150)

    # KS值
    ks = np.max(cdf1[0] / cdf1[0][-1] - cdf2[0] / cdf2[0][-1])
    return ks

# 分类问题
# 准确率、混淆矩阵、召回率&准确率&f1 score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
def classify_metric(y_true, y_pred):
    print('The accuracy score of the model is: {}'.format(accuracy_score(y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred)
    print('The confusion_matrix of the model is:\n {}\n'.format(cm))
    if cm.shape[0] == 2 & cm.shape[1] == 2:
        tp, fn, fp, tn = cm.ravel()
        precision = tn/(fn + tn) # 标记为1的为目标样本
        recall = tn/(fp + tn) # 标记为1的为目标样本
        f1_score = 2*precision*recall/(precision + recall)
        print(' precision: {}\n recall: {}\n fi_score: {}'.format(precision, recall, f1_score))

# 连续值预测问题
from sklearn.metrics import mean_squared_error, log_loss
def continuous_metric(y_true, y_pred, self_func=False):
    print('The mse of the model is: {}'.format(mean_squared_error(y_true, y_pred)))
    print('The log_loss of the model is: {}'.format(log_loss(y_true, y_pred)))
    if self_func != False:
        return self_func(y_true, y_pred)
    
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i]+1) - math.log(y[i]+1)) ** 2 for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5
