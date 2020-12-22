import cupy as cp
import pandas as pd
from xgb_clf import XGBClassifier
from cuml.metrics import roc_auc_score
from config import PATH
from utils import run
from fe import run_fe, get_target, count_encode, target_encode, get_default_feas


def get_x_y(df):
    y = df['answered_correctly'].values.astype('float32')
    df = df.drop(['answered_correctly'], axis=1)

    print(df.columns)
    X = df.values.astype('float32')
    cols = [i for i in df.columns]
    #X = cp.ascontiguousarray(X)
    return X, y, df, cols


def xgb(path):
    FOLD = 0 
    
    train1,valid1 = run_fe(get_default_feas, path, 'default', FOLD)
    train2,valid2 = run_fe(target_encode, path, 'tgt', FOLD)
    train3,valid3 = run_fe(get_target, path, 'y', FOLD)
    train4,valid4 = run_fe(count_encode, path, 'count_encode', FOLD)
    
    train = pd.concat([train1, train2, train3, 
        train4
        ], axis=1)
    valid = pd.concat([valid1, valid2, valid3, 
        valid4
        ], axis=1)

    params = {'n_estimators': 1000,
              'eta': 0.1,
              'early_stopping_rounds': 10,
              'max_depth': 5, 
              'colsample_bytree': 0.5,
              'subsample': 0.5,
              'verbosity': 1,
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'validation_fraction': 0,
             }

    clf = XGBClassifier(**params)


    print('X, y = get_x_y(train)')
    X, y, train, cols = get_x_y(train)
    del train
    print('Xt, yt = get_x_y(valid)')
    Xt, yt, valid, cols = get_x_y(valid)
    del valid

    clf.fit(X, y, Xt, yt)
    clf.clf.save_model(f'{path}/cache/xgb.json')

    yp = clf.predict_proba(Xt)
    print(cols)
    return roc_auc_score(yt, yp)

if __name__ == '__main__':
    run(__file__, True, xgb, PATH)
