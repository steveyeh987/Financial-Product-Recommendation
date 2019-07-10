import random
import os
import sys
sys.path.append('/home/ntu_user/mac_lab/ijcai_script/')
from data_model import User, UserLast
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import hamming_loss
from data_model import User
import pickle
import collections
from collections import OrderedDict
from itertools import chain
from multiprocessing import Pool
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import time
import math
import traceback 
from math import log10

USER_DIR = '/filepool/proj/data/preprocess_cje/ijcai/user_2018/CNN_campaign_profile/'
SAVE_DIR = '/filepool/proj/data/preprocess_cje/ijcai/models/journal/results/'

product = ["credit_card", "credit_loan", "financial_management", "house_loan", "insurance"]

sequence_length = 45 
num_classes = 5 
embedding_size = 256 
#filter_sizes = [2, 3, 4]
filter_sizes = [3]
num_filters = 200
nm_epochs = 100
batch_size = 256
vocab_size = [2, 5, 114, 15, 9, 3, 33, 18, 10, 4, 18, 2, 13, 3, 4, 4, 14, 12, 3, 2, 165, 22, 7, 334]
embedd_dim = [2, 5, 64, 15, 9, 3, 16, 8, 10, 4, 8, 2, 8, 3, 4, 4, 8, 8, 3, 2, 64, 16, 7, 256]

multi_pos, pos, neg_all, v_multi_pos, v_pos, v_neg_all = read_train_data(USER_DIR)

for r in [4, 5, 6]:
    train_num_batches_per_epoch = (r + 1) * (len(multi_pos)+len(pos)) // batch_size
    valid_num_batches_per_epoch = (r + 1) * (len(v_multi_pos)+len(v_pos)) // batch_size

    tf.reset_default_graph()

    x = tf.placeholder(tf.int32, [None, sequence_length, 24])
    y = tf.placeholder(tf.float32, [None, num_classes])
    p = tf.placeholder(tf.float32, [None, 6])
    d = tf.placeholder(tf.int32, [None, sequence_length])
    weight = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    def embeddings(vocab_size, dim):
        lst = []
        for v, d in zip(vocab_size, dim):
            embedding = tf.Variable(tf.random_uniform([v, d], -1.0, 1.0))
            lst.append(embedding)
        return lst

    def masked_embeddings(x, embedding, vocab_size, index):
        mask_y = []
        for e, v, i in zip(embedding, vocab_size, index):
            y = tf.nn.embedding_lookup(e, tf.slice(x, [0, 0, i], [-1, -1, 1]))
            y = tf.squeeze(y, axis=2)
            mask_y.append(y)
        return mask_y

    def conv1d(x, W):
        return tf.nn.conv1d(x, W, stride=1, padding='SAME')
    
    class CausalConv1D(tf.layers.Conv1D):
        def __init__(self, filters,
                   kernel_size,
                   strides=1,
                   dilation_rate=1,
                   activation=None,
                   use_bias=True,
                   kernel_initializer=None,
                   bias_initializer=tf.zeros_initializer(),
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   activity_regularizer=None,
                   kernel_constraint=None,
                   bias_constraint=None,
                   trainable=True,
                   name=None,
                   **kwargs):
            super(CausalConv1D, self).__init__(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='valid',
                data_format='channels_last',
                dilation_rate=dilation_rate,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                trainable=trainable,
                name=name, **kwargs
            )
       
        def call(self, inputs):
            padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
            inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
            return super(CausalConv1D, self).call(inputs)

    def globalavgpool(x, pool_size):
        return tf.layers.average_pooling1d(inputs=x, pool_size=pool_size, strides=1, padding='valid')

    def globalmaxpool(x, pool_size):
        return tf.layers.max_pooling1d(inputs=x, pool_size=pool_size, strides=1, padding='valid')

    def neural_network_model(sequence_length, num_classes, embedding_size, filter_sizes, num_filters, keep_prob): 
        weights = {'W_fc_seq':tf.Variable(tf.truncated_normal([embedding_size, 128], stddev=0.1)),    
                   'out':tf.Variable(tf.truncated_normal([38, num_classes], stddev=0.1))}

        biases = {'b_fc_seq':tf.Variable(tf.constant(0.1, shape=[128])),
                  'out':tf.Variable(tf.constant(0.1, shape=[num_classes]))}
        
        embedding_array = embeddings(vocab_size, embedd_dim) 
        attribute_embedding_array = masked_embeddings(x, embedding_array, vocab_size, np.arange(24))
        concat = tf.concat(attribute_embedding_array, axis=2)
        concat = tf.layers.dense(concat, units=embedding_size, activation=tf.nn.relu)
        # dwell time encoding
        d_emb = tf.Variable(tf.truncated_normal([93, embedding_size],stddev=0.1))
        dwell_embedding = tf.nn.embedding_lookup(d_emb, d)
        #dense += dwell_embedding
        #dense = tf.concat([dwell_embedding, dense], axis=2)
        #dense = (1 + dwell_embedding) * dense
        dwell_embedding = tf.nn.softmax(tf.layers.dense(dwell_embedding, units=embedding_size), dim=-1)
        dense = dwell_embedding * concat
        
        for i in range(4):
            dilation_size = 2 ** i
            dense = CausalConv1D(filters=16*2**(i+1), kernel_size=3, strides=1, dilation_rate=dilation_size, activation=tf.nn.relu)(time_dense)
            dense = tf.contrib.layers.layer_norm(dense)
        dense = tf.reduce_max(dense, axis=1)

        dense_expanded = tf.expand_dims(dense, -1)
        h_concat = tf.layers.dense(concat, units=embedding_size, activation=tf.nn.relu)
        A = tf.nn.softmax(tf.transpose(tf.matmul(h_concat, dense_expanded), [0, 2, 1]), -1)
        M = tf.matmul(A, concat)
        h_flat = tf.squeeze(M, 1) + dense
 
        fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_flat, weights['W_fc_seq']), biases['b_fc_seq']))
        fc = tf.nn.dropout(fc, keep_prob)
        fc = tf.layers.dense(fc, units=32, activation=tf.nn.relu)
        fc = tf.concat([fc ,p], axis=1)
        output = tf.matmul(fc, weights['out']) + biases['out']
        
        return output, tf.squeeze(A)

    def negative_sampling(multi_pos, pos, neg_all):
        multi_pos = np.array(multi_pos)
        pos = np.array(pos)
        neg_all = np.array(neg_all)
        neg_all = neg_all[np.random.choice(np.arange(len(neg_all)), r*(len(multi_pos)+len(pos)))]
        source, target, history, source_dwell = zip(*np.concatenate([multi_pos, pos, neg_all], axis=0))
        return source, target, history, source_dwell


    def next_batch(multi_pos, pos, neg_all, batch_size, epoch):
        # Shuffle data
        source, target, history, source_dwell = negative_sampling(multi_pos, pos, neg_all)
        source = np.array(source)
        target = np.array(target)
        history = np.array(history)
        source_dwell = np.array(source_dwell)
        shuffle_indices = np.random.permutation(np.arange(len(target)))
        source = source[shuffle_indices]
        target = target[shuffle_indices]
        history = history[shuffle_indices]
        source_dwell = source_dwell[shuffle_indices]
        
        for batch_i in range(0, len(source)//batch_size):
            position_batch = []
            start_i = batch_i * batch_size
            source_batch = source[start_i:start_i + batch_size]
            target_batch = target[start_i:start_i + batch_size]
            history_batch = history[start_i:start_i + batch_size]
            #profile_batch = cust_profile[start_i:start_i + batch_size]
            source_dwell_batch = source_dwell[start_i:start_i + batch_size]

            pos_weight = 1
            """
            pos_weight = np.sum(target_batch==0)
            if pos_weight == 0:
                pos_weight = 1
                #pos_weight = (r+1)
            elif pos_weight == len(target_batch):
                pos_weight = 1
                #pos_weight = (r+1)/(r+2)
            else:
                pos_weight = pos_weight /((len(target_batch) - pos_weight) * (r+1))
                #pos_weight = 1
            """
            yield np.array(source_batch), np.array(target_batch), np.array(history_batch), np.array(source_dwell_batch), pos_weight

    def range_with_end(start, stop, step):
        return chain(range(start, stop, step), (stop,))

    def train_neural_network():
        final_pred = []
        final_soft_pred = []
        final_att = []
        scores, attentions = neural_network_model(sequence_length, num_classes, embedding_size, filter_sizes, num_filters, keep_prob)
        pred = tf.nn.sigmoid(scores)
        soft_pred = tf.nn.softmax(scores)
        #cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=scores, targets=y, pos_weight=weight))
        weights = tf.constant([1.0, 3.0, 1.0, 10.0, 2.0])
        cost = tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=y), weights))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            max_precision = 0.0
            max_epoch = 0
            epoch = 0
            for epoch in range(nm_epochs):
                training_loss = 0.0
                valid_loss = 0.0
                for epoch_x, epoch_y, epoch_p, epoch_dwell, epoch_weight in next_batch(multi_pos, pos, neg_all, batch_size, epoch):
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, p: epoch_p, d: epoch_dwell, weight: epoch_weight, keep_prob:1.0})
                    training_loss += c / train_num_batches_per_epoch

                print('Epoch {} training loss: {}'.format(str(epoch+1)+'/'+str(nm_epochs), training_loss))

                # Validation
                valid_pred = []
                sample_size = len(X_valid)
                
                for batch_start, batch_end in zip(range_with_end(0, sample_size, batch_size), range_with_end(batch_size, sample_size, batch_size)):
                    epoch_x, epoch_y, epoch_p, epoch_dwell = X_valid[batch_start:batch_end], Y_valid[batch_start:batch_end], P_valid[batch_start:batch_end], dwell_valid[batch_start:batch_end]
                    prediction = sess.run(pred, feed_dict={x: epoch_x, p: epoch_p, d: epoch_dwell, keep_prob:1.0})
                    valid_pred.append(prediction)
                score = np.concatenate(valid_pred, axis=0)
                prediction = np.round(score)
                valid_precision = f1_score(Y_valid, prediction, average='weighted')
                print("valid_precision:", precision_score(Y_valid, prediction, average='weighted'))
                print("valid_recall:", recall_score(Y_valid, prediction, average='weighted'))
                print("valid_f1:", valid_precision)
                
                if valid_precision > max_precision:
                    max_precision = valid_precision
                    max_epoch = epoch
                else:
                    if epoch - max_epoch > 1:
                        break
                
            for epoch in range(max_epoch+1):
                training_loss = 0.0
                for epoch_x, epoch_y, epoch_p, epoch_dwell, epoch_weight, in next_batch(v_multi_pos, v_pos, v_neg_all, batch_size, epoch):
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, p: epoch_p, d: epoch_dwell, weight: epoch_weight, keep_prob:1.0})
                    training_loss += c / valid_num_batches_per_epoch

                print('Epoch {} training loss: {}'.format(str(epoch+1)+'/'+str(max_epoch+1), training_loss))

            # Testing
            sample_size = len(X_test)
                
            for batch_start, batch_end in zip(range_with_end(0, sample_size, batch_size), range_with_end(batch_size, sample_size, batch_size)):
                epoch_x, epoch_p, epoch_dwell = X_test[batch_start:batch_end], P_test[batch_start:batch_end], dwell_test[batch_start:batch_end]
                prediction, soft_prediction, attention = sess.run([pred, soft_pred, attentions], feed_dict={x: epoch_x, p: epoch_p, d: epoch_dwell, keep_prob:1.0})
                final_pred.append(prediction)
                final_soft_pred.append(soft_prediction)
                final_att.append(attention)
            return final_pred, final_soft_pred, final_att

    final_pred, final_soft_pred, final_att = train_neural_network()
    scores = np.concatenate(final_pred, axis=0)
    soft_scores = np.concatenate(final_soft_pred, axis=0)
    attentions = np.concatenate(final_att, axis=0)
    prediction = np.round(scores)
    np.savez(SAVE_DIR+"results_2_ratio_{}".format(r+1), prediction=prediction, scores=scores, attentions=attentions)

    print(product)
    print("ratio: 1: {}".format(r+1))
    print("precision:", precision_score(Y_test, prediction, average=None))
    print("recall:", recall_score(Y_test, prediction, average=None))
    print("f1-score:", f1_score(Y_test, prediction, average=None))
    print("micro f1-score:", f1_score(Y_test, prediction, average='micro'))
    print("macro f1-score:", f1_score(Y_test, prediction, average='macro'))
    print("weighted f1-score:", f1_score(Y_test, prediction, average='weighted'))
    print("auc-score:", roc_auc_score(Y_test, scores, average=None))
    print("micro auc-score:", roc_auc_score(Y_test, scores, average='micro'))
    print("macro auc-score:", roc_auc_score(Y_test, scores, average='macro'))
    print("weighted auc-score:", roc_auc_score(Y_test, scores, average='weighted'))
    print("hamming_loss:", hamming_loss(Y_test, prediction))

    idx = np.argsort(scores, axis=0)[::-1]
    sorted_Y_test = np.array([Y_test[:, i][idx[:, i]] for i in range(5)]).T
    print("precision@100:", precision_score(sorted_Y_test[:100, :], np.ones((100, 5)), average=None))
    print("precision@1000:", precision_score(sorted_Y_test[:1000, :], np.ones((1000, 5)), average=None))

    idx = np.sum(Y_test, 1) > 0
    print("samples auc-score:", roc_auc_score(Y_test[idx], scores[idx], average='samples'))
    print("samples average precision:", average_precision_score(Y_test[idx], soft_scores[idx], average='samples'))
    print("average precision:", average_precision_score(Y_test, soft_scores, average=None))
    print("miro average precision:", average_precision_score(Y_test, soft_scores, average='micro'))
    print("macro average precision:", average_precision_score(Y_test, soft_scores, average='macro'))
    print("weighted average precision:", average_precision_score(Y_test, soft_scores, average='weighted'))
    
    #np.savez_compressed(USER_DIR+"results/attention_{}_ratio_{}".format(product[i],r+1), X_test=X_test[idx], Y_test=Y_test[:, i][idx], scores=scores[idx], attention=attentions[idx])