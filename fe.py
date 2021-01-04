import cudf as gd
import pandas as pd
import numpy as np
import os
import random
from numba import cuda
from time import time
import cupy as cp
from cuml.preprocessing import TargetEncoder
from tqdm import tqdm
from config import PATH,FOLD
import sys
import glob
from utils import run

def clean(df, drop=True):
    cols = ['viretual_time_stamp', 'user_answer']
    df = df.drop(cols, axis=1)
    
    if drop:
        df = drop_lecture(df)
    
    for i in df.columns:
        if df[i].dtype == 'bool':
            df[i] = df[i].astype('float32')
        #df[i] = df[i].fillna(-999) 
    df['idx'] = cp.arange(df.shape[0])
    return df

def drop_lecture(df):
    mask = df['content_type_id'] == 0
    df = df[mask]
    df = df.drop('content_type_id', axis=1)
    return df

def get_x_y(df, bad=[]):
    y = df['answered_correctly'].values.astype('float32')
    bad = [i for i in bad if i in df.columns]
    df = df.drop(bad+['answered_correctly'], axis=1)

    print(df.columns)
    X = df.values.astype('float64')
    #X = cp.ascontiguousarray(X)
    return X, y, df

def get_data(fillna=None, dropids=True, norm=False):
    train, valid = load_feas()
    bad = ['timestamp']# + [i for i in train.columns if i.startswith('time')]# or i.startswith('prior') or i.startswith('count')]
    if dropids:
        bad = bad + ['user_id', 'content_id', 'part']
    ms = {}
    if norm:
        for col in train.columns:
            if col not in ['user_id', 'content_id', 'part', 'answered_correctly']:
                if train[col].max()>100 and train[col].min()>=0:
                    print(col, train[col].min(), train[col].max())
                    train[col] = np.log1p(train[col])
                    valid[col] = np.log1p(valid[col])
                    print(col, train[col].min(), train[col].max(), valid[col].min(), valid[col].max())
                mean,std = train[col].mean(), train[col].std()
                ms[col] = (mean, std)
                train[col] = (train[col] - mean)/std
                valid[col] = (valid[col] - mean)/std
    if fillna is not None:
        train = train.fillna(fillna)
        valid = valid.fillna(fillna)
    X, y, train = get_x_y(train, bad)
    cols1 = train.columns.values.tolist()
    del train
    Xt, yt, valid = get_x_y(valid, bad)
    cols2 = valid.columns.values.tolist()
    del valid
    assert cols1 == cols2
    print(cols1)
    print(ms)
    return X,y,Xt,yt,cols1

def merge_question(df, dq):
    df = df.merge(dq, left_on='content_id', right_on='question_id', how='left')
    df = df.drop('question_id', axis=1)
    df = df.sort_values('idx')
    return df

def get_train_valid(path, FOLD, drop=True, question=False):
    train = gd.read_parquet(f'{path}/cache/train_{FOLD}.parquet')
    valid = gd.read_parquet(f'{path}/cache/valid_{FOLD}.parquet')

    train = clean(train, drop=False)
    valid = clean(valid, drop=True)
    
    # get most recent samples globally
    
    N = int(valid.shape[0]*1.5)
    base = train[:-N]
    train = train[-N:]
    
    if drop:
        base = drop_lecture(base)
    train = drop_lecture(train) # no online lecture features for now
    
    if question:
        dq = gd.read_csv(f'{path}/questions.csv')
        base = merge_question(base, dq)
        train = merge_question(train, dq)
        valid = merge_question(valid, dq)
    return base,train,valid

def get_sorted_df(df, cols):
    print(cols)
    df = df[cols+['idx']]
    df = df.sort_values('idx').drop('idx', axis=1)
    return df

def ids(path, tag, FOLD):
    _,train,valid = get_train_valid(path, FOLD, question=True)

    ycol = 'answered_correctly'
    cols = ['user_id', 'content_id', 'part']
    print(cols)

    train['user_id'] = train['user_id']%N
    valid['user_id'] = valid['user_id']%N

    train = get_sorted_df(train, cols)
    valid = get_sorted_df(valid, cols)
    return train, valid

def default(path, tag, FOLD):
    
    _,train,valid = get_train_valid(path, FOLD)
    
    ycol = 'answered_correctly'
    cols = [i for i in train.columns if not i.endswith('_id') and i not in [ycol,'idx']]
    #cols = [i for i in train.columns if  i!=ycol]
    print(cols)
    
    train = get_sorted_df(train, cols)
    valid = get_sorted_df(valid, cols)
    return train, valid

def prev_y(path, tag, FOLD):
    
    _,train,valid = get_train_valid(path, FOLD)
    cols = []

    for tag in ['rolling_row']:#,'rolling_row_tags']:
        tr = gd.read_csv(f'{path}/cache/train_{FOLD}_{tag}.csv')
        va = gd.read_csv(f'{path}/cache/valid_{FOLD}_{tag}.csv')
    
        train = train.merge(tr, on='row_id', how='left')
        valid = valid.merge(va, on='row_id', how='left')
    
        cols = cols + [i for i in tr.columns if i!='row_id']
    
    train = get_sorted_df(train, cols)
    valid = get_sorted_df(valid, cols)
    return train, valid

def tocsv(path, tag, FOLD):
    _,train,valid = get_train_valid(path, FOLD, question=True)
    name = f'{path}/cache/train_{FOLD}.csv'
    train.to_csv(name, index=False)
    name = f'{path}/cache/valid_{FOLD}.csv'
    valid.to_csv(name, index=False)

def time_diff(path, tag, FOLD):
    
    base,train,valid = get_train_valid(path, FOLD)
    
    def get_time_diff(user_id, timestamp, time_diff, time_diff2):
        N = len(user_id)
        for i in range(cuda.threadIdx.x, len(user_id), cuda.blockDim.x):
            if i == 0:
                time_diff[i] = -1
            else:
                time_diff[i] = timestamp[i] - timestamp[i-1]
                
            if i<2:
                time_diff2[i] = -1
            else:
                time_diff2[i] = timestamp[i] - timestamp[i-2]

    cols = ['row_id', 'user_id', 'timestamp']                
    tr = train[cols].drop_duplicates(subset=['user_id', 'timestamp'], keep='last')
    print(tr.shape, train.shape)

    tr = tr.groupby('user_id', 
                          as_index=False).apply_grouped(get_time_diff,incols=['user_id','timestamp'],
                                  outcols={'time_diff': 'int64',
                                           'time_diff2': 'int64'},
                                  tpb=32)
    for col in ['time_diff', 'time_diff2']:
        mask = tr[col] == -1
        tr.loc[mask, col] = None

    tr = tr.drop(['user_id', 'timestamp'], axis=1)
    train = train.merge(tr, on='row_id', how='left')
    
    va = valid[cols].drop_duplicates(subset=['user_id', 'timestamp'], keep='last')
    print(va.shape, valid.shape)
    va = va.groupby('user_id', 
                          as_index=False).apply_grouped(get_time_diff,incols=['user_id','timestamp'],
                                  outcols={'time_diff': 'int64',
                                           'time_diff2': 'int64'},
                                  tpb=32)
    for col in ['time_diff', 'time_diff2']:
        mask = va[col] == -1
        va.loc[mask, col] = None

    va = va.drop(['user_id', 'timestamp'], axis=1)
    valid = valid.merge(va, on='row_id', how='left')
    
    cols = ['time_diff', 'time_diff2']
    train = get_sorted_df(train, cols)
    valid = get_sorted_df(valid, cols)
    
    return train, valid

def encode(path, tag, FOLD, func, cols, question=False):
    
    base,train,valid = get_train_valid(path, FOLD, question=question)
    
    ycol = 'answered_correctly'
    print('cols', cols)
    tgt = {}

    for i in tqdm(cols):
        dg = base.groupby(i).agg({ycol:func})
        dg = dg.reset_index()
        dg.columns = [i, f'{tag}_{i}']
        train = train.merge(dg, on=i, how='left')
        valid = valid.merge(dg, on=i, how='left')
        dg.to_parquet(f'{path}/cache/{tag}_{i}_{FOLD}.parquet')
    
    cols = [f'{tag}_{i}' for i in cols]
    train = get_sorted_df(train, cols)
    valid = get_sorted_df(valid, cols)
    
    return train, valid

def count_encode(path, tag, FOLD):
    return encode(path, tag, FOLD, func='count', cols=['content_id', 'task_container_id'])

def target_encode(path, tag, FOLD):
    return encode(path, tag, FOLD, func='mean', cols=['user_id', 'content_id'])

def count_lecture(path, tag, FOLD):
    
    base,train,valid = get_train_valid(path, FOLD, drop=False)
    
    col = 'content_type_id'
    ycol = 'count_lecture'
    
    dg = base.groupby('user_id').agg({col:'sum'})
    dg = dg.reset_index()
    dg.columns = ['user_id', ycol]
    
    train = train.merge(dg, on='user_id', how='left')
    valid = valid.merge(dg, on='user_id', how='left')
    dg.to_parquet(f'{path}/cache/{ycol}_{FOLD}.parquet')
    
    cols = [ycol]
    train = get_sorted_df(train, cols)
    valid = get_sorted_df(valid, cols)
    
    return train, valid

def question_target_encode(path, tag, FOLD):
    return encode(path, tag, FOLD, func='mean', cols=['part'], question=True)

def question_count_encode(path, tag, FOLD):
    return encode(path, tag, FOLD, func='count', question=True)

def target(path, tag, FOLD):
    col = 'answered_correctly'
    _,train,valid = get_train_valid(path, FOLD)
    return train[[col]], valid[[col]]

def run_fe(func, path, tag, FOLD):
    start = time()
    print(f'run fe {tag} ...')  
    name = f"{path}/cache/fe_train_{tag}_{FOLD}.parquet"
    if os.path.exists(name):
        print(f'{name} already exists')
    else:
        tr,va = func(path, tag, FOLD)
        tr.to_parquet(name)
        va.to_parquet(name.replace('train', 'valid'))
    duration = time() - start
    print(f'run fe {tag} done! Time:{duration: .1f} seconds')

def load_feas():
    path = f"{PATH}/cache"
    names = glob.glob(f"{path}/fe_train*")
    tname = f"{path}/fe_train_ids_0.parquet"
    names = [i for i in names if i!=tname] + [i for i in names if i==tname]
    for i in names:
        print(i)
    trs = [pd.read_parquet(name) for name in names]
    vas = [pd.read_parquet(name.replace('train', 'valid')) for name in names]
    print([len(i) for i in vas])
    tr = pd.concat(trs, axis=1)
    va = pd.concat(vas, axis=1)
    return tr,va

if __name__ == '__main__':
    func = sys.argv[1]
    path = PATH
    if func == 'tocsv':
        run(__file__, False, tocsv, path, func, FOLD)
    else:
        run_fe(eval(func), path, func, FOLD)
