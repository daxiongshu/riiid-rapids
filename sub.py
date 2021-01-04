import pandas as pd
import xgboost as xgb
from time import time
from collections import defaultdict
import json
from config import PATH


path = f'{PATH}/cache'
bst = xgb.Booster()
bst.load_model(f'{path}/xgb.json')

FOLD = 0
tgt = {}
tag = 'target_encode'
for i in ['user_id', 'content_id']:
    tgt[i] = pd.read_parquet(f'{path}/{tag}_{i}_{FOLD}.parquet')
    
ce = {}
tag = 'count_encode'
for i in ['content_id', 'task_container_id']:
    ce[i] = pd.read_parquet(f'{path}/{tag}_{i}_{FOLD}.parquet')
    
qtgt = {}
tag = 'question_target_encode'
for i in ['part']:
    qtgt[i] = pd.read_parquet(f'{path}/{tag}_{i}_{FOLD}.parquet')
    
dt = pd.DataFrame()
dt['user_id'] = [-1]
dt['prevtime'] = [0]
dt['time_diff'] = [0]

dq = pd.read_csv(f'{PATH}/questions.csv')
dq = dq.drop(['tags','bundle_id'], axis=1)

ucols=['content_id', 'part']
y_prev_mean = {i:{} for i in ['user_id']+ucols}
alpha = 0.01
mean = 0.66

with open((f'{path}/rolling_0.json')) as f:
    y_mean = json.load(f)
    
prev = {i:[] for i in ['user_id']+ucols}
prev['is_question'] = []

def merge_question(df, dq):
    df = df.merge(dq, left_on='content_id', right_on='question_id', how='left')
    df = df.drop('question_id', axis=1)
    return df

def time_diff(df, dt):
    ucols = ['user_id', 'timestamp']
    df = df.merge(dt, on='user_id', how='left')
    
    delta = df['timestamp'] - df['prevtime']
    df['time_diff2'] = delta + df['time_diff']
    df['time_diff'] = delta
    
    mask = df.content_type_id == 0
    tmp = df[mask][['user_id', 'timestamp', 'time_diff']]
    tmp.columns = ['user_id', 'prevtime', 'time_diff']
    dt = pd.concat([dt, tmp], axis=0)
    dt = dt.drop_duplicates(subset=['user_id'], keep='last')
    del tmp
    
    df['time_diff'] = df['time_diff'].fillna(-1)
    df['time_diff2'] = df['time_diff2'].fillna(-1)
    
    return df, dt

def encode(df, dg, col, tag):
    tcol =  f'{tag}_{col}'
    df = df.merge(dg, on=col, how='left')
    return df

def tostr(x):
    if x is None:
        return ''
    return str(x)

def fe(df, dt, tgt, ce, prev):
    
    for i in df.columns:
        if i == 'prior_question_had_explanation':
            df[i] = df[i].astype('float32')
        df[i] = df[i].fillna(-999)
    
    df = merge_question(df, dq)
    
    #start = time()
    prev_y = eval(df["prior_group_answers_correct"].iloc[0])
    
    qs = prev['is_question']
    #print('non question', sum(qs))
    
    for col in ['user_id']+ucols:
        v = prev[col]
        for u,y,q in zip(v, prev_y, qs):
            if q!=0:
                continue
            if u not in y_mean[col]:
                y_mean[col][u] = mean*(1-alpha) + alpha*y
            else:
                y_mean[col][u] = y_mean[col][u]*(1-alpha) + alpha*y
    
    users = df['user_id'].values.tolist()
    users = [tostr(i) for i in users]
    prev = {}
    for col in ucols:
        l = df[col].values.tolist()
        prev[col] = ['%s-%s'%(i,tostr(j)) for i,j in zip(users,l)]   
    prev['user_id'] = users
    prev['is_question'] = df['content_id'].values.tolist()
    
    for col in ['user_id']+ucols:
        df[f'y_rolling_{col}'] = [y_mean[col][i] if i in y_mean[col] else None for i in prev[col]]
        
    #duration = time() - start
    #print(f'rolling: {duration:.3f} seconds')
    #rs = [i for i in df.columns if i.startswith('y_rolling')]+['row_id']
    #print(df[rs])
    
    
    
    #start = time()
    df,dt = time_diff(df, dt)
    #duration = time() - start
    #print(f'time_diff: {duration:.3f} seconds')
    #print(df[['user_id', 'timestamp', 'prevtime', 'time_diff', 'time_diff2']])
        
    for col,dg in ce.items():
        df = encode(df, dg, col, 'count_encode')
    
    for col,dg in tgt.items():
        df = encode(df, dg, col, 'target_encode')
        
    for col,dg in qtgt.items():
        df = encode(df, dg, col, 'question_target_encode')
   
    cols = ['question_target_encode_part', 'prior_question_elapsed_time', 
            'prior_question_had_explanation', 'y_rolling_user_id', 
            'y_rolling_content_id', 'y_rolling_part', 'target_encode_user_id',
            'target_encode_content_id', 'time_diff', 'time_diff2', 
            'count_encode_content_id', 'count_encode_task_container_id']
    
    df = df[cols]
    return df.values, dt, prev


BEST_ITER = 100

for (test_df, sample_prediction_df) in iter_test:
    Xt,dt,prev = fe(test_df, dt, tgt, ce, prev)
    dtest = xgb.DMatrix(data=Xt)
    yp = bst.predict(dtest)#,ntree_limit=BEST_ITER)
    test_df['answered_correctly'] = yp
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])