from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def train_split(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test
