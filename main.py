import numpy as np
import pandas as pd
import pandas as ps
import scipy.sparse as sp


def freq(x, prob=True):
    values = np.unique(x)
    if prob:
        p_values = x.value_counts() / x.value_counts().sum()
        return [values, p_values]
    else:
        ni = x.value_counts()
        return [np.sort(values), ni]


if __name__ == '__main__':
    x = pd.read_csv("zoo.csv")
    [xi, ni] = freq(x['legs'], prob=False)
    print([xi, ni])
    print(np.unique(x))

