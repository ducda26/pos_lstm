import numpy as np
import pandas as pd


def read_data(data_path):
    data = pd.read_csv(data_path, sep = '\t', header=None)
    data = data.drop([2,3,5,6,7,8,9],axis=1)
    data = data.rename(columns={0: "ID", 1: "WORD", 4: "POS"})
    data["ID"] = np.nan
    data["ID"][0] = "Câu 1"
    cnt = 2
    for i, word in enumerate(data.WORD):
        if word == ".":
            data["ID"][i+1] = "Câu " + str(cnt)
            cnt = cnt + 1
    data["ID"] = data.ID.fillna(method='ffill')

    words = list(set(data["WORD"].values))
    words.append("ENDPAD")
    n_words = len(words); n_words


    return data, words, n_words



