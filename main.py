import numpy as np
import pandas as pd
import scipy.sparse as sp
import math


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



# def entropy(x):
#     _sum = 0
#     _, y = freq(x)
#     for p in y:
#         _sum -= p * math.log(p, 2)
#     return _sum


def info_gain(x, y):
    suma = 0
    _, _, z = freq2(x, y)
    for i in z:
        suma += i * math.log(i, 2)
    return entropy(x) + entropy(y) - -suma


if __name__ == '__main__':
    x = pd.read_csv("zoo.csv")
    print(freq2(x['legs'], x['type'], prob=True))
    print(entropy(x['type']))
    print(info_gain(x['domestic'], x['type']))
    print(x.columns)
    x = pd.read_csv("zoo.csv")
    for i in x.columns:
        if i != 'type':
            print(i, info_gain(x[i], x['type']))
#zad4
# for i in x.columns:
#     if i!='type':
#         #print("działa")
#         z=info_gain(x[i],x['type'])
#         print(i,z)