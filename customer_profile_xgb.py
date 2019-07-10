import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from utils import myEval, read_X_chg_y_train_test_time, timeEval, topEval
import xgboost as xgb
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

N_CORES=20
if not grid_search_on: # when testing, n_cores could be higher
    N_CORES=25

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
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=1/100, random_state=42)
print('train size: {} ; valid size: {}, test size: {}'.format(len(train_X), len(valid_X), len(test_X)))
print('shape of train_X: {}, shape of train_y: {}'.format(train_X.shape, train_y.shape))
assert((len(train_X)==len(train_y)) and (len(test_X)==len(test_y)))

# xgb
param_grid = {
             'n_estimators': [300, 400],
             'max_depth': [5, 10, 15],
             'min_child_weight': [10, 20, 30],
             'scale_pos_weight': [1]
             }

## base model
print('XGBoost')
#base_xgb = xgb.XGBClassifier(n_estimators=200, n_jobs=N_CORES) 
base_xgb = xgb.XGBClassifier(n_estimators=400, min_child_weight=20, max_depth=5, n_jobs=N_CORES) 
#base_xgb = xgb.XGBClassifier(n_estimators=400, min_child_weight=20, max_depth=10, scale_pos_weight=1.0, n_jobs=N_CORES) 
base_xgb.fit(train_X, train_y)
print('evaluating:  - {} -'.format(product_name))
print(' ~ base model ~ ')
base_predictions = base_xgb.predict(test_X)
base_proba = base_xgb.predict_proba(test_X)
myEval(y_pred=base_predictions, y_true=test_y, y_score=base_proba)
print()
topEval(base_proba, test_y, 100)
print()
topEval(base_proba, test_y, 1000)
print()
topEval(base_proba, test_y, 5000)
print()
print('test_time evaluation:')
test_time_predictions = base_xgb.predict(test_time_X)
timeEval(test_time_predictions, test_time_y)

# check feat. importance
if not grid_search_on:
    print()
    imp = base_xgb.feature_importances_
    assert len(X_cols)==len(imp)
    imp_order = np.argsort(imp)[::-1]
    tmp = pd.DataFrame({'feature': np.array(X_cols)[imp_order], 'importance': imp[imp_order]})
    tmp.importance = tmp.importance.apply(lambda x: str(int(x*10000)/100.0)+ ' %')
    print(tmp.head(50 if len(X_cols) >=50 else len(X_cols)))


## grid search
if grid_search_on:
    def expand_grid(dictionary):
        return pd.DataFrame([row for row in product(*dictionary.values())], 
                                                columns=dictionary.keys())
    for i, row in expand_grid(param_grid).iterrows():
        xgb_grid = xgb.XGBClassifier(n_estimators=row['n_estimators'], min_child_weight=row['min_child_weight'], max_depth=row['max_depth'], scale_pos_weight=row['scale_pos_weight'], n_jobs=N_CORES) 
        #xgb = xgb.XGBClassifier()
        #xgb_grid = GridSearchCV(estimator = xgb, param_grid = param_grid, cv = 3, verbose=1, n_jobs = N_CORES)
        xgb_grid.fit(train_X, train_y)
        #best_xgb = xgb_grid.best_estimator_
        print()
        #print(' ~ tuned model ~ ')
        #print('best param:')
        #print(xgb_grid.best_params_)
        print(row)
        best_predictions = xgb_grid.predict(test_X)
        best_proba = xgb_grid.predict_proba(test_X)
        myEval(y_pred=best_predictions, y_true=test_y, y_score=best_proba)
        print()
        topEval(best_proba, test_y, 1000)