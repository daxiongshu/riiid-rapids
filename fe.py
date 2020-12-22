import cudf as gd
import pandas as pd
import os
import random
from numba import cuda
from time import time
import cupy as cp
from xgb_clf import XGBClassifier
from cuml.preprocessing import TargetEncoder
from tqdm import tqdm
from config import PATH

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
    
    # get most recent samples globally
    
    N = valid.shape[0]
    base = train[:-N]
    train = train[-N:]
    print(base.shape, train.shape, valid.shape)
    
    
    # get per user most recent 
    """
    def get_order_in_group(user_id, order):
        N = len(user_id)
        for i in range(cuda.threadIdx.x, len(user_id), cuda.blockDim.x):
            order[i] = i - N

    train = train.groupby('user_id', 
                          as_index=False).apply_grouped(get_order_in_group,incols=['user_id'],
                                  outcols={'order': 'int32'},
                                  tpb=32)
    mask = train['order']>-200
    base = train[~mask]
    train = train[mask]
    base = base.drop('order', axis=1)
    train = train.drop('order', axis=1)
    print(base.shape, train.shape, valid.shape)
    train = train.sort_values('idx')
    """
    return base,train,valid

def get_sorted_df(df, cols):
    df = df[cols+['idx']]
    df = df.sort_values('idx').drop('idx', axis=1)
    return df

def get_default_feas(path, tag, FOLD):
    
    _,train,valid = get_train_valid(path, FOLD)
    ycol = 'answered_correctly'
    cols = [i for i in train.columns if not i.endswith('_id') and i!=ycol]
    #cols = [i for i in train.columns if  i!=ycol]
    print(cols)
    
    train = get_sorted_df(train, cols)
    valid = get_sorted_df(valid, cols)
    return train, valid

def count_encode(path, tag, FOLD):
    
    base,train,valid = get_train_valid(path, FOLD)
    
    ycol = 'answered_correctly'
    id_cols = [i for i in train.columns if i.endswith('_id') and i!='row_id']
    print('id_cols', id_cols)
    tgt = {}

    for i in tqdm(id_cols):
        dg = base.groupby(i).agg({ycol:'count'})
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
    
    base,train,valid = get_train_valid(path, FOLD)
    
    ycol = 'answered_correctly'
    id_cols = [i for i in train.columns if i.endswith('_id') and i!='row_id']
    print('id_cols', id_cols)
    
    out_cols = []
    for i in tqdm(id_cols):
        dg = base.groupby(i).agg({ycol:'mean'})
        dg = dg.reset_index()
        dg.columns = [i, f'tgt_{i}']
        train = train.merge(dg, on=i, how='left')
        valid = valid.merge(dg, on=i, how='left')
        dg.to_parquet(f'{path}/cache/tgt_{i}_{FOLD}.parquet')
        out_cols.append(f'tgt_{i}')
    
    train = get_sorted_df(train, out_cols)
    valid = get_sorted_df(valid, out_cols)
    
    return train, valid

def get_target(path, tag, FOLD):
    col = 'answered_correctly'
    _,train,valid = get_train_valid(path, FOLD)
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

