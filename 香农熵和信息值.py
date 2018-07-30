import numpy as np
def calc_ent(x):
    """
        calculate shanno ent of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

def calc_condition_ent(x, y):
    """
        calculate ent H(y|x)
    """
    # calc ent(y|x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent

    return ent


def calc_ent_grap(x,y):
    """
        calculate ent grap
    """

    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap


def CalcIV(Xvar, Yvar):
    N_0 = np.sum(Yvar == 0)
    N_1 = np.sum(Yvar == 1)
    N_0_group = np.zeros(np.unique(Xvar).shape)
    N_1_group = np.zeros(np.unique(Xvar).shape)
    for i in range(len(np.unique(Xvar))):
        N_0_group[i] = Yvar[(Xvar == np.unique(Xvar)[i]) & (Yvar == 0)].count()
        N_1_group[i] = Yvar[(Xvar == np.unique(Xvar)[i]) & (Yvar == 1)].count()
    iv = np.sum((N_0_group / N_0 - N_1_group / N_1) * np.log((N_0_group / N_0) / (N_1_group / N_1)))
    return iv


def caliv_batch(df, Kvar, Yvar):
    df_Xvar = df.drop([Kvar, Yvar], axis=1)
    ivlist = []
    for col in df_Xvar.columns:
        iv = CalcIV(df[col], df[Yvar])
        ivlist.append(iv)
    names = list(df_Xvar.columns)
    iv_df = pd.DataFrame({'Var': names, 'Iv': ivlist}, columns=['Var', 'Iv'])

    return iv_df