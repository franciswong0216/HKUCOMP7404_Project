# -*- coding:utf-8 -*-
# !/usr/bin/env python

import pickle
import os
import time
import datetime
import jieba
import jieba.analyse
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
from collections import Counter

import tensorflow as tf
import numpy as np

from cnn import *
from train import *
from gensim.models import Word2Vec

import os

# label set
label_list = ['Military', 'Economy', 'Culture', 'Sports', 'Auto', 'Medicine']

# encode every word, every will encode with index
def encoder(dict_path, path):
    # Build context and labels
    all_context, all_labels, all_files = zip(*loading_corpus(dict_path))
    eval_context, eval_labels, eval_files = zip(*loading_corpus(path))

    # Building dictionary from ../corpus documents
    vocab_processor = learn.preprocessing.VocabularyProcessor(1500, min_frequency=5)

    # Assign index based on dictionary
    all_context = list(vocab_processor.fit_transform(all_context))
    eval_context = list(vocab_processor.fit_transform(eval_context))
    print("number of words :", len(vocab_processor.vocabulary_))
    pickle.dump((eval_context, eval_labels, eval_files), open("result_test.pkl", "wb"))

# get label for softmax
def soft_max_label(label):
    new_label = 6 * [0]
    index = label_list.index(label)
    new_label[index] = 1
    return new_label

# cut sentence and join
def delete_and_split(all_context, all_labels, all_File):
    new_data = []
    data = zip(all_context, all_labels, all_File)
    for context, label, filename in data:
        article = ' '.join(list(jieba.cut(context)))
        new_data.append((article, soft_max_label(label), filename))
    return new_data	
	
# load all document in corpus
def loading_corpus(path):
    allContext = []
    allLabel = []
    allFile = os.listdir(path=path)
    allFilename = []
    for file in allFile:
        label = file.split('_')[0]
        filePath = os.path.join(path, file)
        filename = file
        with open(filePath, 'r', encoding='utf-8') as fd:
            context = fd.read()
        allContext.append(context)
        allLabel.append(label)
        allFilename.append(filename)

    newData = delete_and_split(allContext, allLabel, allFilename)
    return newData	

def loadmodel(session, checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("Trained model '%s' is found.  Reloading trained session..." % ckpt_name)
        new_saver = tf.train.Saver()
        new_saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

def result_step(optimizer, global_step, batch, label, filename):
    feed_dict = {
	cnn.input_sentence: batch,
	cnn.label: label,
	cnn.dropout_keep_prob: 0.5
    }
    step, result, label, accuracy = sess.run([global_step, cnn.result, cnn.label, cnn.accuracy], feed_dict=feed_dict)

    print("-------------------------------------------- Results ---------------------------------------------")
    for i in range(len(filename)):
        print(" File - {:18s}, predicted = {:10s}, probability = {:.8%}, correct label = {}".format( filename[i], label_list[np.argmax(result[i])], np.amax(result[i]), label_list[np.argmax(label[i])]))
    print("==================================================================================================")
    print(" Overall accuracy = {:.8%}".format(accuracy))
    print("==================================================================================================")

if __name__ == "__main__":
    encoder("../corpus", "../input")

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
        
        if not loadmodel(sess, "../states"):
            print("No session found. Training model...")
            train_model()        
        else:
            print("Session is reloaded.")

        # Load evaluation data
        test_x, test_y , test_files= pickle.load(open("result_test.pkl", "rb"))
        result_step(optimizer, global_step, test_x, test_y, test_files)
