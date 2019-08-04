# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('data/data.csv')
result_col = '是否受欢迎'
df.index = df['ID']
df


# %%
def cal_entropy(nb, title=''):

    print("------------{}------------".format(title))
    s = nb.sum()
    e = (nb/s).apply(lambda x: -x*np.log2(x)).sum()
    print(nb.rename_axis(index=[None]))
    print('sum: {}, entropy: {}'.format(s, e))
    print('------------------------\n')
    return e


def cal_entropy_D(df):
    groups = df.groupby(result_col)
    size = groups.size()
    return cal_entropy(size, 'D')


def cal_entropy_D_F(df, col):
    print('col: {}\n======================='.format(col))

    groups = df.groupby(col).groups
    X = pd.DataFrame({val: df.loc[idxs].groupby(result_col).size()
                      for (val, idxs) in groups.items()}).fillna(0)

    s = X.sum().to_frame().T.rename(index={0: 'SUM'})

    ss = s.sum().sum()
    p = s.loc['SUM'] / ss

    print(pd.concat([X, s]), '\nsum: {}\n'.format(ss))

    e = 0
    for c in X.columns:
        print('val {}: {}'.format(c, groups[c].values))
        e = e + cal_entropy(X[c]) * p[c]
    print('H(D|{}): {}\n------------------------\n\n'.format(col, e))
    return e


# %%
def cal_G(df):
    H_D = cal_entropy_D(df)
    F = ['Appearance', 'Income', 'Age', 'Profession']
    H_F = pd.DataFrame({c: [cal_entropy_D_F(df, c)] for c in F})
    G = H_D - H_F
    print('G:------------------------\n', G)
    print(G.idxmax(axis=1)[0])
    print('------------------------\n')


# %%
if __name__ == "__main__":
    print('🔶🔶🔶🔶🔶🔶🔶🔶')
    cal_G(df)
    # root -> Appearance
    # Ah: [4,5,6,10,14]
    # Good: [1,2,8,9,11]
    # Great: [3,7,12,13] -> Y
    df_Ah = df.loc[[4, 5, 6, 10, 14]]
    df_Good = df.loc[[1, 2, 8, 9, 11]]
    df_Great = df.loc[[3, 7, 12, 13]]
# %%
    pd.concat([df_Ah, df_Good, df_Great])


# %%
    # branch Ah -> Profession
    df_Ah
# %%
    print('🔶🔶🔶🔶🔶🔶🔶🔶')
    cal_G(df_Ah)
    # Steady: [4,5,10] -> Y
    # Unstable: [6,14] -> N
    df_Ah_Steady = df.loc[[4, 5, 10]]
    df_Ah_Unstable = df.loc[[6, 14]]
# %%
    pd.concat([df_Ah_Steady, df_Ah_Unstable])

# %%
    # branch Good -> Age
    df_Good

# %%
    print('🔶🔶🔶🔶🔶🔶🔶🔶')
    cal_G(df_Good)
    # Older: [1,2,8] -> N
    # Younger: [9,11] -> Y
    df_Good_Older = df.loc[[1, 2, 8]]
    df_Good_Younger = df.loc[[9, 11]]
# %%
    pd.concat([df_Good_Older, df_Good_Younger])

# %%
    # branch Great -> Y
    df_Great
# %%
    print('🔶🔶🔶🔶🔶🔶🔶🔶')
    cal_entropy_D(df_Great)

# %%
    xs = np.linspace(0.001, 1, 1000)
    ys = - xs * np.log(xs)
    plt.plot(xs, ys)
