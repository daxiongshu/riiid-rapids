import cudf as gd
import pandas as pd
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

def clean(df, drop=True):
    cols = ['viretual_time_stamp', 'user_answer']
    df = df.drop(cols, axis=1)
    
    if drop:
        df = drop_lecture(df)
    
    for i in df.columns:
        if df[i].dtype == 'bool':
            df[i] = df[i].astype('float32')
        df[i] = df[i].fillna(-999) 
    df['idx'] = cp.arange(df.shape[0])
    return df

def drop_lecture(df):
    mask = df['content_type_id'] == 0
    df = df[mask]
    df = df.drop('content_type_id', axis=1)
    return df

def get_x_y(df, bad=[]):
    y = df['answered_correctly'].values.astype('float32')
    df = df.drop(bad+['answered_correctly'], axis=1)

    print(df.columns)
    X = df.values.astype('float64')
    #X = cp.ascontiguousarray(X)
    return X, y, df

def get_data():
    train, valid = load_feas()
    bad = ['timestamp']
    X, y, train = get_x_y(train, bad)
    cols1 = train.columns.values.tolist()
    del train
    Xt, yt, valid = get_x_y(valid, bad)
    cols2 = valid.columns.values.tolist()
    del valid
    assert cols1 == cols2
    print(cols1)
    return X,y,Xt,yt,cols1

def merge_question(df, dq):
    df = df.merge(dq, left_on='content_id', right_on='question_id', how='left')
    df = df.drop('question_id', axis=1)
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
    df = df[cols+['idx']]
    df = df.sort_values('idx').drop('idx', axis=1)
    return df

def default(path, tag, FOLD):
    
    _,train,valid = get_train_valid(path, FOLD)
    
    ycol = 'answered_correctly'
    cols = [i for i in train.columns if not i.endswith('_id') and i!=ycol]
    #cols = [i for i in train.columns if  i!=ycol]
    print(cols)
    
    train = get_sorted_df(train, cols)
    valid = get_sorted_df(valid, cols)
    return train, valid

def user_order(path, tag, FOLD):
    
    base,train,valid = get_train_valid(path, FOLD)
    
    def get_order_in_group(user_id, order):
        N = len(user_id)
        for i in range(cuda.threadIdx.x, len(user_id), cuda.blockDim.x):
            order[i] = i - N

    tmp = gd.concat([base[['user_id','row_id']], train[['user_id','row_id']], valid[['user_id','row_id']]], axis=0)
    tmp = tmp.groupby('user_id', 
                          as_index=False).apply_grouped(get_order_in_group,incols=['user_id'],
                                  outcols={'order': 'int32'},
                                  tpb=32)

    train = train.merge(tmp[['row_id', 'order']], on='row_id', how='left')
    valid = valid.merge(tmp[['row_id', 'order']], on='row_id', how='left')
    
    cols = ['order']
    train = get_sorted_df(train, cols)
    valid = get_sorted_df(valid, cols)
    
    return train, valid

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
    
    tr = tr.drop(['user_id', 'timestamp'], axis=1)
    train = train.merge(tr, on='row_id', how='left')
    
    va = valid[cols].drop_duplicates(subset=['user_id', 'timestamp'], keep='last')
    print(va.shape, valid.shape)
    va = va.groupby('user_id', 
                          as_index=False).apply_grouped(get_time_diff,incols=['user_id','timestamp'],
                                  outcols={'time_diff': 'int64',
                                           'time_diff2': 'int64'},
                                  tpb=32)
    va = va.drop(['user_id', 'timestamp'], axis=1)
    valid = valid.merge(va, on='row_id', how='left')
    
    cols = ['time_diff', 'time_diff2']
    train = get_sorted_df(train, cols)
    valid = get_sorted_df(valid, cols)
    
    return train, valid

def encode(path, tag, FOLD, func, question=False):
    
    base,train,valid = get_train_valid(path, FOLD, question=question)
    
    ycol = 'answered_correctly'
    if question:
        id_cols = ['bundle_id', 'part']
    else:
        id_cols = [i for i in train.columns if i.endswith('_id') and i!='row_id']
    print('id_cols', id_cols)
    tgt = {}

    for i in tqdm(id_cols):
        dg = base.groupby(i).agg({ycol:func})
        dg = dg.reset_index()
        dg.columns = [i, f'{tag}_{i}']
        train = train.merge(dg, on=i, how='left')
        valid = valid.merge(dg, on=i, how='left')
        dg.to_parquet(f'{path}/cache/{tag}_{i}_{FOLD}.parquet')
    
    id_cols = [f'{tag}_{i}' for i in id_cols]
    train = get_sorted_df(train, id_cols)
    valid = get_sorted_df(valid, id_cols)
    
    return train, valid

def count_encode(path, tag, FOLD):
    return encode(path, tag, FOLD, func='count')

def target_encode(path, tag, FOLD):
    return encode(path, tag, FOLD, func='mean')

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
    return encode(path, tag, FOLD, func='mean', question=True)

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
    print(names)
    trs = [pd.read_parquet(name) for name in names]
    vas = [pd.read_parquet(name.replace('train', 'valid')) for name in names]
    print([len(i) for i in vas])
    tr = pd.concat(trs, axis=1)
    va = pd.concat(vas, axis=1)
    return tr,va

if __name__ == '__main__':
    func = sys.argv[1]
    path = PATH
    run_fe(eval(func), path, func, FOLD)
