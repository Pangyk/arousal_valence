from tensorflow import keras
from sklearn import metrics
import data_parser as dp
import tensorflow as tf
import parameter as p
import os


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def predict(x, y_a, y_v):
    model_a = keras.Sequential([
        keras.layers.Dense(p.kernel1, activation=tf.nn.softmax),
        keras.layers.Dense(p.kernel1, activation=tf.nn.softmax),
        keras.layers.Dense(p.kernel2, activation=tf.nn.softmax),
        keras.layers.Dense(p.kernel2, activation=tf.nn.softmax),
        keras.layers.Dense(2, activation=tf.nn.sigmoid)
    ])
    if os.path.isfile(p.path_a + ".index"):
        model_a.load_weights(p.path_a)

    model_v = keras.Sequential([
        keras.layers.Dense(p.kernel3, activation=tf.nn.softmax),
        keras.layers.Dense(p.kernel3, activation=tf.nn.softmax),
        keras.layers.Dense(p.kernel3, activation=tf.nn.softmax),
        keras.layers.Dense(2, activation=tf.nn.sigmoid)
    ])
    if os.path.isfile(p.path_v + ".index"):
        model_v.load_weights(p.path_v)

    pred_a = tf.argmax(model_a.predict(x, batch_size=p.total - p.train), 1)
    _a_acc = metrics.accuracy_score(y_a, pred_a)
    _a_f1 = metrics.f1_score(y_a, pred_a, pos_label=1)

    pred_v = tf.argmax(model_v.predict(x, batch_size=p.total - p.train), 1)
    _v_acc = metrics.accuracy_score(y_v, pred_v)
    _v_f1 = metrics.f1_score(y_v, pred_v, pos_label=1)

    return _a_acc, _a_f1, _v_acc, _v_f1


if __name__ == "__main__":
    features = dp.parse("resources/deep_features.txt", p.total)[p.train - p.total:]
    arousal_class = dp.parse("resources/arousal_class.txt", p.total)[p.train - p.total:] - 1
    valence_class = dp.parse("resources/valence_class.txt", p.total)[p.train - p.total:] - 1
    a_acc, a_f1, v_acc, v_f1 = predict(features, arousal_class, valence_class)
    print("Arousal result: accuracy is " + str(a_acc) + ", F1 score is " + str(a_f1))
    print("Valence result: accuracy is " + str(v_acc) + ", F1 score is " + str(v_f1))
