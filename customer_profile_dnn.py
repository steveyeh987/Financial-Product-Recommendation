import pandas as pd
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import time
import os
import sys
import pickle
from utils import myEval, read_X_chg_y_train_test_time, timeEval, topEval, should_stop

neg_prop = str(sys.argv[1])
train_type = str(sys.argv[2])
rand_num = str(sys.argv[3])
product_name = str(sys.argv[4])

include_flag_cnt = str(sys.argv[5])=='1' # use 1/0 in cli arg to control
include_chg = str(sys.argv[6])=='1'
include_seq_len = str(sys.argv[7])=='1'
include_last_y = str(sys.argv[8])=='1'

DIR = '/filepool/proj/data/preprocess_cje/ijcai/user_2018/wilson/1_'+neg_prop+'/'+train_type+'/'+rand_num+'/'

prods = ['creditcard', 'credit_loan', 'deposit', 'financial_management', 'house_loan', 'insurance'] # order matters!

EVAL_ONLY = False
## Hyperparameters
MODEL = 'DNN' # DNN or CNN
BATCH_SIZE = 128
KEEP_PROB = 0.8
POS_WEIGHT = 2.0
LEARNING_RATE = 0.0025
PRINT_FREQ = 1
N_EARLY_STOPPING = 3
N_EPOCH = 18
GD_START_EPOCH = 6
GD_START_BS = 32
INC_BS_START_EPOCH = 10
INC_BS_SIZE = 32
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_file_name = product_name+'_'+MODEL+'_model_'+rand_num+'.npz'

# load data
train_X, train_X_chg, train_X_last_y, train_y, test_X, test_X_chg, test_X_last_y, test_y, test_time_X, test_time_X_chg, test_time_X_last_y, test_time_y = read_X_chg_y_train_test_time(DIR, product_name)
X_cols = pickle.load(open(DIR+product_name+'_X_cols.pkl', 'rb'))

# remove some unnecessary columns (the reference classes for dummy variables)
N_col_ind = [i for i, c in enumerate(X_cols) if c.endswith('_flag_N') or c.endswith('_code_0.0') or c.endswith('_ind_N') or c=='gender_type_code_F']
ind = [i for i in range(len(X_cols)) if i not in N_col_ind]
train_X, test_X, test_time_X = train_X[:,ind], test_X[:,ind], test_time_X[:,ind]
X_cols = [c for i,c in enumerate(X_cols) if i not in N_col_ind]


if not include_flag_cnt:
    N_col_ind = [i for i, c in enumerate(X_cols) if ('_flag_' in c) or c.endswith('_cnt')]
    ind = [i for i in range(len(X_cols)) if i not in N_col_ind]
    train_X, test_X, test_time_X = train_X[:,ind], test_X[:,ind], test_time_X[:,ind]
    X_cols = [c for i,c in enumerate(X_cols) if i not in N_col_ind]
else:
    print('include flag and cnt')

if True:
    print('be aware: include chg')
    train_X = np.concatenate([train_X_chg, train_X], axis = 1)
    test_X = np.concatenate([test_X_chg, test_X], axis = 1)
    test_time_X = np.concatenate([test_time_X_chg, test_time_X], axis = 1)
    X_cols = ['asset_chg_'+str(i) for i in range(3)] + ['liability_chg_'+str(i) for i in range(3)] + X_cols
    
if not include_seq_len:
    N_col_ind = [i for i, c in enumerate(X_cols) if c.startswith('seq_len_')]
    ind = [i for i in range(len(X_cols)) if i not in N_col_ind]
    train_X, test_X, test_time_X = train_X[:,ind], test_X[:,ind], test_time_X[:,ind]
    X_cols = [c for i,c in enumerate(X_cols) if i not in N_col_ind]
else:
    print('include seq_len')

if include_last_y:
    print('include last_y')
    train_X = np.concatenate([train_X_last_y, train_X], axis = 1)
    test_X = np.concatenate([test_X_last_y, test_X], axis = 1)
    test_time_X = np.concatenate([test_time_X_last_y, test_time_X], axis = 1)
    X_cols = ['last_y_'+s for s in ['creditcard', 'credit_loan', 'deposit', 'financial_management', 'house_loan', 'insurance']]+X_cols


zip_col_ind = [i for i, c in enumerate(X_cols) if c.startswith('contact_zip')]
occu_col_ind = [i for i, c in enumerate(X_cols) if c.startswith('occupation_desc')]
inv_col_ind = [i for i, c in enumerate(X_cols) if c.startswith('invest_tolerance_code')]
edu_col_ind = [i for i, c in enumerate(X_cols) if c.startswith('education_code')]
mari_col_ind = [i for i, c in enumerate(X_cols) if c.startswith('marital_status_code')]
N_col_ind = [i for i, c in enumerate(X_cols) if c.endswith('_flag_N') or c.endswith('_code_0.0') or c.endswith('ind_N') or c=='gender_type_code_F'] # should be removed
prof_col_ind = [i for i in range(len(X_cols)) if i not in (zip_col_ind+occu_col_ind+inv_col_ind+edu_col_ind+mari_col_ind+N_col_ind)]
#print([X_cols[i] for i in prof_col_ind])
def slice_input(X):
    # prof_train_X, zip_train_X, occu_train_X, inv_train_X, edu_train_X, mari_train_X = slice_input(train_X)
    return X[:, prof_col_ind], X[:, zip_col_ind], X[:, occu_col_ind], X[:, inv_col_ind], X[:, edu_col_ind], X[:, mari_col_ind]

## train / valid / test split
train_X, valid_X, train_X_chg, valid_X_chg, train_y, valid_y = train_test_split(train_X, train_X_chg, train_y, test_size=0.1, random_state=42)
print('train size: {} ; valid size: {}, test size: {}'.format(len(train_X), len(valid_X), len(test_X)))
print('using - {} - model'.format(MODEL))
print('shape of train_X: {}, shape of train_y: {}'.format(train_X.shape, train_y.shape))
print('prop of pos in train_y: {}'.format(round(np.mean(train_y), 4)))
assert((len(train_X)==len(train_y)) and (len(test_X)==len(test_y)))

if MODEL=='DNN':
    # Model construction
    tf.set_random_seed(42)
    
    prof_X = tf.placeholder(tf.float32, shape=(None, len(prof_col_ind)))
    occu_X = tf.placeholder(tf.float32, shape=(None, len(occu_col_ind)))
    zip_X = tf.placeholder(tf.float32, shape=(None, len(zip_col_ind)))
    inv_X = tf.placeholder(tf.float32, shape=(None, len(inv_col_ind)))
    edu_X = tf.placeholder(tf.float32, shape=(None, len(edu_col_ind))) 
    mari_X = tf.placeholder(tf.float32, shape=(None, len(mari_col_ind))) 
    chg_X = tf.placeholder(tf.float32, shape=(None, train_X_chg.shape[1]))
    y_ = tf.placeholder(tf.int64, shape=(None, ))

    zip_net = tl.layers.InputLayer(zip_X, name='zip_input')
    #zip_net = tl.layers.DenseLayer(zip_net, 4, b_init=None, act=tf.identity, name='zip_embed')
    occu_net = tl.layers.InputLayer(occu_X, name='occu_input')
    occu_net = tl.layers.DenseLayer(occu_net, 8, b_init=None, act=tf.identity, name='occu_embed')
    inv_net = tl.layers.InputLayer(inv_X, name='inv_input')
    #inv_net = tl.layers.DenseLayer(inv_net, 3, b_init=None, act=tf.identity, name='inv_embed')
    edu_net = tl.layers.InputLayer(edu_X, name='edu_input')
    #edu_net = tl.layers.DenseLayer(edu_net, 3, b_init=None, act=tf.identity, name='edu_embed')
    mari_net = tl.layers.InputLayer(mari_X, name='mari_input')
    #mari_net = tl.layers.DenseLayer(mari_net, 3, b_init=None, act=tf.identity, name='mari_embed')
    prof_input_net = tl.layers.InputLayer(prof_X, name='prof_input')

    chg_net = tl.layers.InputLayer(chg_X, name='chg_input')
    #chg_net = tl.layers.DenseLayer(chg_net, 64, act=lambda x : tl.act.lrelu(x, 0.1), W_init=tf.contrib.layers.xavier_initializer(), name='chg_relu1')
    #chg_net = tl.layers.DenseLayer(chg_net, 8, act=lambda x : tl.act.lrelu(x, 0.1), W_init=tf.contrib.layers.xavier_initializer(), name='chg_relu2') 

    #prof_deep_net = tl.layers.DropoutLayer(prof_input_net, keep=0.8, name='prof_drop1')
    #prof_deep_net = tl.layers.DenseLayer(prof_input_net, 128, tf.nn.relu, W_init=tf.contrib.layers.xavier_initializer(), name='prof_relu1')
    #prof_deep_net = tl.layers.DropoutLayer(prof_deep_net, keep=KEEP_PROB, name='prof_drop2')
    #prof_deep_net = tl.layers.DenseLayer(prof_deep_net, 64, tf.nn.relu, name='prof_relu2')
    
    #concat_net = tl.layers.ConcatLayer(layer = [prof_input_net, prof_deep_net, chg_net, zip_net, occu_net, inv_net, edu_net, mari_net], name ='concat_layer')
    concat_net = tl.layers.ConcatLayer(layer = [prof_input_net, chg_net, zip_net, occu_net, inv_net, edu_net, mari_net], name ='concat_layer')
    net = tl.layers.DenseLayer(concat_net, 64, tf.nn.relu, W_init=tf.contrib.layers.xavier_initializer(), name='relu1')
    net = tl.layers.DropoutLayer(net, keep=KEEP_PROB, name='drop1')
    net = tl.layers.DenseLayer(net, 32, tf.nn.relu, W_init=tf.contrib.layers.xavier_initializer(), name='relu2')
    net = tl.layers.DropoutLayer(net, keep=KEEP_PROB, name='drop2')
    #net = tl.layers.DenseLayer(net, 32, tf.nn.relu, W_init=tf.contrib.layers.xavier_initializer(), name='relu3')
    #net = tl.layers.DropoutLayer(net, keep=KEEP_PROB, name='drop3')
    #net = tl.layers.DenseLayer(net, 16, tf.nn.relu, W_init=tf.contrib.layers.xavier_initializer(), name='relu4')
    net = tl.layers.DenseLayer(net, n_units=2, act=tf.identity, name='output')

    # define cost function and metric for multilabel classification
    y = net.outputs
    
    cost = tl.cost.cross_entropy(y, y_, name='cross-entropy')
    
    targ = tf.cast(tf.concat([tf.expand_dims(1-y_, 1), tf.expand_dims(y_, 1)], axis = 1), tf.float32)
    class_weights = tf.constant([[1.0, POS_WEIGHT]])
    weights = tf.reduce_sum(class_weights * targ, axis=1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=targ, logits=y)
    weighted_losses = unweighted_losses * weights
    cost = tf.reduce_mean(weighted_losses)
    
    l2 = 0
    #for param_num in [0, 2, 11]:
    #    p = net.all_params[param_num]
    #    print('l2 reg: {}'.format(p))
    #    l2 = l2 + tf.contrib.layers.l2_regularizer(0.001)(p)
    cost = cost + l2
    model_prediction = tf.argmax(y, 1)
    correct_prediction = tf.equal(model_prediction, y_ )
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # define the optimizer
    train_params = net.all_params
    adam_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost, var_list=train_params)
    gd_train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost, var_list=train_params)

sess = tf.InteractiveSession()
# initialize all variables in the session
#tl.layers.initialize_global_variables(sess)
sess.run(tf.global_variables_initializer())
# print network information
net.print_params(False)
#net.print_layers()

if EVAL_ONLY: # evaluation mode
    load_params = tl.files.load_npz(name=DIR+model_file_name)
    tl.files.assign_params(sess, load_params, net)
else:       # training mode
    # train the network
    #tl.utils.fit(sess, net, train_op, cost, train_X, train_y, X, y_, acc=acc, batch_size=BATCH_SIZE, \
    #        n_epoch=N_EPOCH, PRINT_FREQ=1, X_val=valid_X, y_val=valid_y, eval_train=True)
    val_loss_list = []
    for epoch in range(N_EPOCH):
        start_time = time.time()
        if epoch+1 < GD_START_EPOCH:
            bs = BATCH_SIZE
        else:
            if epoch + 1 < INC_BS_START_EPOCH:
                bs = GD_START_BS
            else:
                bs = GD_START_BS + INC_BS_SIZE * (epoch + 2 - INC_BS_START_EPOCH)
        for train_X_a, train_y_a in tl.iterate.minibatches(np.hstack((train_X, train_X_chg)), train_y, bs, shuffle=True):
            train_X_a, train_X_chg_a = train_X_a[:,0:-train_X_chg.shape[1]], train_X_a[:,-train_X_chg.shape[1]:]
            prof_train_X_a, zip_train_X_a, occu_train_X_a, inv_train_X_a, edu_train_X_a, mari_train_X_a = slice_input(train_X_a)
            feed_dict = {prof_X: prof_train_X_a, zip_X: zip_train_X_a, occu_X: occu_train_X_a, 
                    inv_X: inv_train_X_a, edu_X: edu_train_X_a, mari_X: mari_train_X_a,
                    chg_X: train_X_chg_a, y_: train_y_a}
            feed_dict.update(net.all_drop)  # enable noise layers
            if epoch+1 < GD_START_EPOCH:
                sess.run(adam_train_op, feed_dict=feed_dict)
            else:
                sess.run(gd_train_op, feed_dict=feed_dict)
        if epoch + 1 == 1 or (epoch + 1) % PRINT_FREQ == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, N_EPOCH, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for train_X_a, train_y_a in tl.iterate.minibatches(np.hstack((train_X, train_X_chg)), train_y, 2048, shuffle=True):
                train_X_a, train_X_chg_a = train_X_a[:,0:-train_X_chg.shape[1]], train_X_a[:,-train_X_chg.shape[1]:]
                prof_train_X_a, zip_train_X_a, occu_train_X_a, inv_train_X_a, edu_train_X_a, mari_train_X_a = slice_input(train_X_a)
                dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
                feed_dict = {prof_X: prof_train_X_a, zip_X: zip_train_X_a, occu_X: occu_train_X_a, 
                        inv_X: inv_train_X_a, edu_X: edu_train_X_a, mari_X: mari_train_X_a,
                        chg_X: train_X_chg_a, y_: train_y_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += ac
                n_batch += 1
            if epoch+1 < GD_START_EPOCH: 
                print('adam, batch size: {}'.format(bs))
            else:
                print('gd, batch size: {}'.format(bs))
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for valid_X_a, valid_y_a in tl.iterate.minibatches(np.hstack((valid_X, valid_X_chg)), valid_y, 1024, shuffle=True):
                valid_X_a, valid_X_chg_a = valid_X_a[:,0:-valid_X_chg.shape[1]], valid_X_a[:,-valid_X_chg.shape[1]:]
                prof_valid_X_a, zip_valid_X_a, occu_valid_X_a, inv_valid_X_a, edu_valid_X_a, mari_valid_X_a = slice_input(valid_X_a)
                feed_dict = {prof_X: prof_valid_X_a, zip_X: zip_valid_X_a, occu_X: occu_valid_X_a,
                        inv_X: inv_valid_X_a, edu_X: edu_valid_X_a, mari_X: mari_valid_X_a,
                        chg_X: valid_X_chg_a, y_: valid_y_a}
                dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err
                val_acc += ac
                n_batch += 1
            val_loss_list += [(val_loss / n_batch)]
            print("                             val loss: %f" % (val_loss / n_batch))                    
            print("                             val acc: %f" % (val_acc / n_batch))
            if should_stop(val_loss_list, times=N_EARLY_STOPPING):
                print('early stopping')
                break

    ## save the network to .npz file
    tl.files.save_npz(net.all_params, name=DIR+model_file_name, sess = sess)
print('all val loss: ')
print(val_loss_list)

# evaluation
pred_raw, predictions, label_y = None, None, None
test_loss, test_acc, n_batch = 0, 0, 0
for test_X_a, test_y_a in tl.iterate.minibatches(np.hstack((test_X, test_X_chg)), test_y, 1024, shuffle=True):
    test_X_a, test_X_chg_a = test_X_a[:,0:-test_X_chg.shape[1]], test_X_a[:,-test_X_chg.shape[1]:]
    prof_test_X_a, zip_test_X_a, occu_test_X_a, inv_test_X_a, edu_test_X_a, mari_test_X_a = slice_input(test_X_a)
    feed_dict = {prof_X: prof_test_X_a, zip_X: zip_test_X_a, occu_X: occu_test_X_a,
            inv_X: inv_test_X_a, edu_X: edu_test_X_a, mari_X: mari_test_X_a,
            chg_X: test_X_chg_a, y_: test_y_a}
    dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
    feed_dict.update(dp_dict)
    err, ac = sess.run([cost, acc], feed_dict=feed_dict)
    test_loss += err
    test_acc += ac
    n_batch += 1
    pred_raw_a, predictions_a = sess.run([y, model_prediction], feed_dict=feed_dict)
    pred_raw = pred_raw_a if pred_raw is None else np.concatenate([pred_raw, pred_raw_a])
    predictions = predictions_a if predictions is None else np.concatenate([predictions, predictions_a])
    label_y = test_y_a if label_y is None else np.concatenate([label_y, test_y_a])
    
print('quick check: test_y: {}/{}, label_y: {}/{}'.format(np.sum(test_y), len(test_y), np.sum(label_y),len(label_y)))
# precision and recall for each class
proba = pred_raw[:,1]
print('evaluating:  - {} -'.format(product_name))
myEval(y_pred=predictions, y_true=label_y, y_score=proba)
print()
topEval(proba, label_y, 100)
print()
topEval(proba, label_y, 1000)
print()
topEval(proba, label_y, 5000)
print()
print(pred_raw[105:112,:])