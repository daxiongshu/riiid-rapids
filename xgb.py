import cudf as gd
import os
import random
import numba
import cupy as cp
from xgb_clf import XGBClassifier
from cuml.preprocessing import TargetEncoder
from tqdm import tqdm
from cuml.preprocessing.model_selection import train_test_split 
from cuml.metrics import roc_auc_score
from config import PATH
from utils import run

def clean(df):
    cols = ['viretual_time_stamp', 'user_answer']
    df = df.drop(cols, axis=1)

    mask = df['content_type_id'] == 0
    df = df[mask]
    df = df.drop('content_type_id', axis=1)
    
    for i in df.columns:
        if i == 'prior_question_had_explanation':
            df[i] = df[i].astype('float32')
        df[i] = df[i].fillna(-999) 
    return df

def get_x_y(df):
    y = df['answered_correctly'].values.astype('float32')
    df = df.drop(['row_id', 'answered_correctly'], axis=1)

    X = df.values.astype('float32')
    X = cp.ascontiguousarray(X)
    return X, y, df

def xgb(path):
    FOLD = 0
    train = gd.read_parquet(f'{path}/cache/train_{FOLD}.parquet')
    valid = gd.read_parquet(f'{path}/cache/valid_{FOLD}.parquet')

    train = clean(train)
    valid = clean(valid)

    id_cols = [i for i in train.columns if i.endswith('_id') and i!='row_id']
    tgt = {}

    for i in tqdm(id_cols):
        encoder = TargetEncoder()
        train[i] = encoder.fit_transform(train[i], train['answered_correctly'])
        valid[i] = encoder.transform(valid[i])
        tgt[i] = encoder.encode_all.drop(['__TARGET___x', '__TARGET___y'], axis=1)
        del encoder

    params = {'n_estimators': 100,
              'eta': 0.1,
              'max_depth': 10,
              'verbosity': 1,
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'validation_fraction': 0,
             }

    clf = XGBClassifier(**params)


    print('X, y = get_x_y(train)')
    X, y, train = get_x_y(train)
    del train
    print('Xt, yt = get_x_y(valid)')
    Xt, yt, valid = get_x_y(valid)
    del valid

    clf.fit(X, y, Xt, yt)
    clf.clf.save_model(f'{path}/cache/xgb.json')

    for i in tgt:
        tgt[i].to_parquet(f'{path}/cache/tgt_{i}.parquet')


    yp = clf.predict_proba(Xt)
    return roc_auc_score(yt, yp)

if __name__ == '__main__':
    run(__file__, True, xgb, PATH)
