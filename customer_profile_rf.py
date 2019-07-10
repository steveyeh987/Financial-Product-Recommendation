import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from utils import myEval, read_X_chg_y_train_test_time, timeEval, topEval
import sys

neg_prop = str(sys.argv[1])
train_type = str(sys.argv[2])
rand_num = str(sys.argv[3])
product_name = str(sys.argv[4])
grid_search_on = False

include_flag_cnt = str(sys.argv[5])=='1' # use 1/0 in cli arg to control
include_chg = str(sys.argv[6])=='1'
include_seq_len = str(sys.argv[7])=='1'
include_last_y = str(sys.argv[8])=='1'

no_clip=False
if no_clip:
    print('warning: no_clip=True')

N_CORES=12
if not grid_search_on:
    N_CORES=20

DIR = '/filepool/proj/data/preprocess_cje/ijcai/user_2018/wilson/1_'+neg_prop+'/'+train_type+'/'+rand_num+'/'

# load data
train_X, train_X_chg, train_X_last_y, train_y, test_X, test_X_chg, test_X_last_y, test_y, test_time_X, test_time_X_chg, test_time_X_last_y, test_time_y = read_X_chg_y_train_test_time(DIR, product_name, no_clip=no_clip)
if no_clip:
    X_cols = pickle.load(open(DIR+'no_clip_'+product_name+'_X_cols.pkl', 'rb'))
else:
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

if include_chg:
    print('include chg')
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



## train / valid / test split
#train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=1/100000, random_state=42)
#print('train size: {} ; valid size: {}, test size: {}'.format(len(train_X), len(valid_X), len(test_X)))
print('train size: {}, test size: {}'.format(len(train_X), len(test_X)))
print('shape of train_X: {}, shape of train_y: {}'.format(train_X.shape, train_y.shape))
assert((len(train_X)==len(train_y)) and (len(test_X)==len(test_y)))
assert len(X_cols)==train_X.shape[1]

#random forest
param_grid = {
             'n_estimators': [300, 400],
             'max_features': ['sqrt'],
             'max_depth': [10, None],
             'min_samples_split': [5, 10],
             'min_samples_leaf': [5, 10],
             'bootstrap': [True, False]
             }

## base model
print('random forest')
#base_rf = RandomForestClassifier(n_estimators=200,oob_score=True, random_state=42, n_jobs = N_CORES)
base_rf = RandomForestClassifier(n_estimators=400, max_features='sqrt', max_depth=None, min_samples_split=10, min_samples_leaf=5,oob_score=True, random_state=42, n_jobs = N_CORES)
base_rf.fit(train_X, train_y)


print('evaluating:  - {} -'.format(product_name))
print(' ~ base model ~ ')
base_predictions = base_rf.predict(test_X)
base_proba = base_rf.predict_proba(test_X)
myEval(y_pred=base_predictions, y_true=test_y, y_score=base_proba)
print()
topEval(base_proba, test_y, 100)
print()
topEval(base_proba, test_y, 1000)
print()
topEval(base_proba, test_y, 5000)
print()
print('{} users are kept from training for test_time_eval'.format(int(len(test_time_y)/12)))
#print('test_time evaluation:')
#test_time_predictions = base_rf.predict(test_time_X)
#timeEval(test_time_predictions, test_time_y)

# check feat. importance
if not grid_search_on:
    imp = base_rf.feature_importances_
    assert len(X_cols)==len(imp)
    imp_order = np.argsort(imp)[::-1]
    tmp = pd.DataFrame({'feature': np.array(X_cols)[imp_order], 'importance': imp[imp_order]})
    tmp.importance = tmp.importance.apply(lambda x: str(int(x*10000)/100.0)+ ' %')
    print(tmp.head(50 if len(X_cols) >=50 else len(X_cols)))

## grid search
if grid_search_on:
    rf = RandomForestClassifier()
    rf_grid = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 4, verbose=1, n_jobs = N_CORES)
    rf_grid.fit(train_X, train_y)
    best_rf = rf_grid.best_estimator_
    print()
    print(' ~ tuned model ~ ')
    print('best param:')
    print(rf_grid.best_params_)
    best_predictions = best_rf.predict(test_X)
    best_proba = best_rf.predict_proba(test_X)
    myEval(y_pred=best_predictions, y_true=test_y, y_score=best_proba)
    print()
    topEval(best_proba, test_y, 1000)
    print()
    print('test_time evaluation:')
    test_time_predictions = best_rf.predict(test_time_X)
    timeEval(test_time_predictions, test_time_y)
