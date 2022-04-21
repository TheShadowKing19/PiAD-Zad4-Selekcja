import numpy as np
import pandas as pd
import scipy.sparse as sp
import math
from sklearn.datasets import fetch_rcv1


def freq(x, prob=True):
    return np.unique(x), x.value_counts(prob)


def freq2(x, y, prob=True):
    df = pd.concat([x, y], axis=1)
    return np.unique(x), np.unique(y), df.value_counts(normalize=prob)


def entropy(x):
    suma = 0
    _, values = freq(x)
    for i in values:
        suma += i * math.log(i, 2)
    return -suma


def info_gain(x, y):
    suma = 0
    _, _, z = freq2(x, y)
    for i in z:
        suma += i * math.log(i, 2)
    return entropy(x) + entropy(y) - -suma

# Zad 5
def convert_to_sparse_pandas(DataFrame, exclude_columns=[]):
    DataFrame = DataFrame.copy()
    exclude_columns = set(exclude_columns)
    for (columnName, columnData) in DataFrame.iteritems():
        if columnName not in exclude_columns:
            DataFrame[columnName] = pd.SparseArray(columnData.values, fill_value=0, dtype='uint8')

    return DataFrame


if __name__ == '__main__':
    x = pd.read_csv("zoo.csv")
    print(freq(x['legs']))
    print(freq(x['legs'], prob=False))
    print("\n", freq2(x['legs'], x['type'], prob=True))
    print("\n", freq2(x['legs'], x['type'], prob=False))
    print(entropy(x['type']))
    print(info_gain(x['domestic'], x['type']))
    print(x.columns)
    print("\n\n")

    # Zad 4
    x = pd.read_csv("zoo.csv")
    for i in x.columns:
        if i != 'type':
            print(i, info_gain(x[i], x['type']))
    print("\n\n")

    # Zad 5
    data, numbers = freq(x['legs'])
    data_one_hot = pd.get_dummies(x, columns=x.columns)
    print(data_one_hot.head(), data_one_hot.dtypes)
    data_one_hot_sparse = convert_to_sparse_pandas(data_one_hot)
    print(data_one_hot_sparse.head(), data_one_hot_sparse.dtypes)
    print("\n\n")
    print(freq(data_one_hot_sparse['legs_4']))
    print(freq2(data_one_hot_sparse['eggs_True'], data_one_hot_sparse['type_bird'], prob=True))

    # Zad 6
    rcv1 = fetch_rcv1()
    df = pd.DataFrame(rcv1.data, columns=rcv1.target_names)
    print(df.head())
