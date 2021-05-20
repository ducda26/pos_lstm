from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.sentences import GetSentence


def text_to_token(data, words):
    get = GetSentence(data)
    sentences = get.all_sentences
    pos = list(set(data['POS']))
    max_len = max([len(s) for s in sentences])
    w_index = {w: i + 1 for i, w in enumerate(words)}
    p_index = {t: i for i, t in enumerate(pos)}

    X = [[w_index[w[0]]for w in s]for s in sentences]
    X = pad_sequences(maxlen=max_len, padding='post', sequences=X)

    Y = [[p_index[w[1]] for w in s] for s in sentences]
    Y = pad_sequences(maxlen=max_len, padding='post', sequences=Y)

    return X, Y, max_len, pos
