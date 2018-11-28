import tensorflow as tf
import datetime
import pickle
import numpy as np

from cnn import *
from PreProcess import *
from gensim.models import Word2Vec

import matplotlib.pyplot as plt

global_loss = []
global_accuracy = []
batches = get_batch(3, 300)

def train_step(sess, global_step, cnn, optimizer, batch, label):
    feed_dict = {
        cnn.input_sentence: batch,
        cnn.label: label,
        cnn.dropout_keep_prob: 0.5
    }
    _, W, b, step, loss, accuracy, predictions = sess.run(
        [optimizer, cnn.W, cnn.b, global_step, cnn.losses, cnn.accuracy, cnn.predictions],
        feed_dict=feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {}, accuracy {}".format(time_str, step, loss, accuracy))
    if step == 227:
        final_w = sess.run(cnn.W)
        final_b = sess.run(cnn.b)
        print("w and B")
        print(final_w)
        print(final_b)

    global_loss.append(loss)
    global_accuracy.append(accuracy)

def dev_step(sess, global_step, cnn, optimizer, batch, label):
    feed_dict = {
        cnn.input_sentence: batch,
        cnn.label: label,
        cnn.dropout_keep_prob: 0.5
    }
    step, accuracy = sess.run([global_step, cnn.accuracy], feed_dict=feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, accuracy {:g}".format(time_str, step, accuracy))

def train_model():
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = Cnn(sequence_length=1500,
                    embedding_size=128,
                    filter_sizes=[1, 2, 3, 4, 5, 6, 7],
                    num_filters=150,
                    num_classes=6,
                    vocab_size=74680)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.001).minimize(cnn.losses, global_step=global_step)
            sess.run(tf.global_variables_initializer())

            test_x, test_y = pickle.load(open("corpus_test.pkl", "rb"))
            for data in batches:
                x_train, y_train = zip(*data)
                train_step(sess, global_step, cnn, optimizer, x_train, y_train)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % 30 == 0:
                    dev_step(sess, global_step, cnn, optimizer, test_x, test_y)

            # Save the variables to disk.
            saver = tf.train.Saver()
            save_path = saver.save(sess, "../states/model.ckpt")
            print("Model saved in path: %s" % save_path)

            # Output training results in graphs
            x = list(range(len(global_loss)))
            plt.plot(x, global_loss, 'r', label="loss")
            plt.xlabel("batches")
            plt.ylabel("loss")
            plt.savefig("../output/loss_modify.png")
            plt.close()

            plt.plot(x, global_accuracy, 'b', label="accuracy")
            plt.xlabel("batches")
            plt.ylabel("accuracy")
            plt.savefig("../output/accuracy.png")
            plt.close()

if __name__ == "__main__":
    train_model()
