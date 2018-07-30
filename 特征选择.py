import numpy as np
import pandas as pd

# -------------------- Filter ------------------------
# 方差选择法，返回值为特征选择后的数据
# 参数threshold为方差的阈值
from sklearn.feature_selection import VarianceThreshold
def variance_filter(ts, feature):
    '''
    :param ts: the threshold of variance
    :param feature: features of pandas' DataFranme type
    :return: results
    '''
    vt = VarianceThreshold(threshold=ts)
    vt.fit(feature)
    res = pd.DataFrame({'feature': list(feature.columns),
                        'if_retain': vt.get_support()})
    return res

# 相关系数法
from scipy.stats import pearsonr

def pearson_filter(x, y, k):
    '''
    :param X: features
    :param Y: target
    :param k: numbers of features to retain
    :return: results dataFrame with pearsonr and p_value
    '''
    pear = [pearsonr(x[i], y)[0] for i in list(x.columns)]
    p_value = [pearsonr(x[i], y)[1] for i in list(x.columns)]
    res = pd.DataFrame({'feature': list(x.columns),
                        'pear': pear,
                        'p_value': p_value})
    res = res.sort_values(by='pear', ascending=False)
    return res[:k]

# 卡方法
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
def chi2_filter(x, y, k):
    '''
    :param x: features
    :param y: target
    :param k: numbers of features to retain
    :return: results dataFrame with score and p_value
    '''
    kbest = SelectKBest(chi2, k)
    kbest.fit_transform(x, y)
    res = pd.DataFrame({'feature': list(x.columns),
                        'score': kbest.scores_,
                        'p_value': kbest.pvalues_})
    return res

# -------------------- Wrapper ------------------------
# 递归特征消除法
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
def rfe_wrapper(x, y, k):
    '''
    :param x: features
    :param y: target
    :param k: numbers of features to retain
    :return: results feture names
    '''
    rfe = RFE(estimator=LogisticRegression(), n_features_to_select=2)
    rfe.fit(x, y)
    res = pd.DataFrame({'feature': list(x.columns),
                        'if_retain': rfe.get_support()})
    return list(res['feature'].loc[res['if_retain'] == 'True'])

# -------------------- Embedded ------------------------
# L1
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

def l1_embedded(x, y, c):
    '''
    :param x: features
    :param y: target
    :return: results feture names
    '''
    lr = SelectFromModel(LogisticRegression(penalty="l1", C=c))
    lr.fit(x, y)
    res = pd.DataFrame({'feature': list(x.columns),
                        'if_retain': lr.get_support()})
    return list(res['feature'].loc[res['if_retain'] == 'True'])

# 随机森林法
from sklearn.ensemble import RandomForestClassifier

def rf_selection(x, y, n, max_depth, k):
    '''
    :param x: features
    :param y: target
    :param n: estimators of random forests
    :param max_depth: max_depth of random forests
    :param k: numbers of features to retain
    :return: results feture names
    '''
    rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth)
    rf.fit_transform(x.values, y.values)
    features_name = list(x.columns)
    importance = list(rf.feature_importances_)
    df = pd.DataFrame({'feature': features_name,
                       'importance': importance})
    df.sort_values(by='importance', ascending=False, inplace=True)
    return list(df['feature'].iloc[:k])
