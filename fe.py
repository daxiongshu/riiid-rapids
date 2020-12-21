import cudf as gd
import pandas as pd
import os
import random
import numba
from time import time
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
    df['idx'] = cp.arange(df.shape[0])
    return df

def get_x_y(df):
    y = df['answered_correctly'].values.astype('float32')
    df = df.drop(['row_id', 'answered_correctly'], axis=1)

    print(df.columns)
    X = df.values.astype('float32')
    X = cp.ascontiguousarray(X)
    return X, y, df

def merge_question(df, dq):
    df = df.merge(dq, left_on='content_id', right_on='question_id', how='left')
    df = df.drop('question_id', axis=1)
    return df

def get_train_valid(path, FOLD):
    train = gd.read_parquet(f'{path}/cache/train_{FOLD}.parquet')
    valid = gd.read_parquet(f'{path}/cache/valid_{FOLD}.parquet')

    train = clean(train)
    valid = clean(valid)
    
    return train,valid

def get_sorted_df(df, cols):
    df = df[cols+['idx']]
    df = df.sort_values('idx').drop('idx', axis=1)
    return df

def get_default_feas(path, tag, FOLD):
    
    train,valid = get_train_valid(path, FOLD)
    ycol = 'answered_correctly'
    cols = [i for i in train.columns if not i.endswith('_id') and i!=ycol]
    print(cols)
    
    train = get_sorted_df(train, cols)
    valid = get_sorted_df(valid, cols)
    return train, valid

def count_encode(path, tag, FOLD):
    
    train,valid = get_train_valid(path, FOLD)
    
    ycol = 'answered_correctly'
    id_cols = [i for i in train.columns if i.endswith('_id') and i!='row_id']
    print('id_cols', id_cols)
    tgt = {}

    for i in tqdm(id_cols):
        dg = train.groupby(i).agg({ycol:'count'})
        dg = dg.reset_index()
        dg.columns = [i, f'count_{i}']
        train = train.merge(dg, on=i, how='left')
        valid = valid.merge(dg, on=i, how='left')
        dg.to_parquet(f'{path}/cache/count_{i}_{FOLD}.parquet')
    
    id_cols = [f'count_{i}' for i in id_cols]
    train = get_sorted_df(train, id_cols)
    valid = get_sorted_df(valid, id_cols)
    
    return train, valid

def target_encode(path, tag, FOLD):
    
    train,valid = get_train_valid(path, FOLD)
    
    ycol = 'answered_correctly'
    id_cols = [i for i in train.columns if i.endswith('_id') and i!='row_id']
    print('id_cols', id_cols)
    tgt = {}

    for i in tqdm(id_cols):
        encoder = TargetEncoder()
        train[i] = encoder.fit_transform(train[i], train['answered_correctly'])
        valid[i] = encoder.transform(valid[i])
        tgt[i] = encoder.encode_all.drop(['__TARGET___x', '__TARGET___y'], axis=1)
        del encoder
        
    train = get_sorted_df(train, id_cols)
    valid = get_sorted_df(valid, id_cols)
    
    for i in tgt:
        tgt[i].to_parquet(f'{path}/cache/tgt_{i}_{FOLD}.parquet')
    
    return train, valid

def get_target(path, tag, FOLD):
    col = 'answered_correctly'
    train,valid = get_train_valid(path, FOLD)
    return train[[col]], valid[[col]]

def run_fe(func, path, tag, FOLD):
    start = time()
    print(f'run fe {tag} ...')  
    name = f"{path}/cache/fe_train_{tag}_{FOLD}.parquet"
    if os.path.exists(name):
        print(f'{name} already exists')
        tr = pd.read_parquet(name)
        va = pd.read_parquet(name.replace('train', 'valid'))
    else:
        tr,va = func(path, tag, FOLD)
        tr.to_parquet(name)
        va.to_parquet(name.replace('train', 'valid'))
    duration = time() - start
    print(f'run fe {tag} done! Time:{duration: .1f} seconds')
    return tr,va

if __name__ == '__main__':
    run(__file__, True, xgb, PATH)

