from tensorflow import keras
from sklearn import metrics
import data_parser as dp
import tensorflow as tf
import parameter as p
import os


def train(x, y_a, y_v, epochs):
    idx = tf.range(p.total)
    idx = tf.random.shuffle(idx)
    x_train = tf.gather(x, idx[:p.train])
    y_train_a, y_train_v = tf.gather(y_a, idx[:p.train]), tf.gather(y_v, idx[:p.train])
    x_test = tf.gather(x, idx[p.train - p.total:])
    y_test_a, y_test_v = tf.gather(y_a, idx[p.train - p.total:]), tf.gather(y_v, idx[p.train - p.total:])
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

    model_a.compile(optimizer=keras.optimizers.Adam(lr=p.lr1), loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    model_a.fit(x_train, y_train_a, epochs=epochs, batch_size=p.batch_size)
    pred_a = tf.argmax(model_a.predict(x_test, batch_size=p.total - p.train), 1)
    _a_acc = metrics.accuracy_score(y_test_a, pred_a)
    _a_f1 = metrics.f1_score(y_test_a, pred_a, pos_label=1)
    model_a.save_weights(p.path_a)

    model_v.compile(optimizer=keras.optimizers.Adam(lr=p.lr2), loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    model_v.fit(x_train, y_train_v, epochs=epochs, batch_size=p.batch_size)
    pred_v = tf.argmax(model_v.predict(x_test, batch_size=p.total - p.train), 1)
    _v_acc = metrics.accuracy_score(y_test_v, pred_v)
    _v_f1 = metrics.f1_score(y_test_v, pred_v, pos_label=1)
    model_v.save_weights(p.path_v)

    return _a_acc, _a_f1, _v_acc, _v_f1


if __name__ == "__main__":
    features = dp.parse("resources/deep_features.txt", p.total)
    arousal_class = dp.parse("resources/arousal_class.txt", p.total) - 1
    valence_class = dp.parse("resources/valence_class.txt", p.total) - 1
    avg_a_acc, avg_a_f1, avg_v_acc, avg_v_f1 = 0, 0, 0, 0
    for i in range(p.repeat):
        a_acc, a_f1, v_acc, v_f1 = train(features, arousal_class, valence_class, p.epochs)
        avg_a_acc += a_acc
        avg_a_f1 += a_f1
        avg_v_acc += v_acc
        avg_v_f1 += v_f1
    print("Arousal result: average accuracy is " + str(avg_a_acc / p.repeat) + ", average F1 score is " + str(
        avg_a_f1 / p.repeat))
    print("Valence result: average accuracy is " + str(avg_v_acc / p.repeat) + ", average F1 score is " + str(
        avg_v_f1 / p.repeat))
