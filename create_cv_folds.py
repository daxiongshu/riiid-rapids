import cudf as gd
import os
import random
import numba
import cupy
from tqdm import tqdm
from utils import run
from config import PATH

# - row_id 0 101230331 int64
# - timestamp 0 87425772049 int64
# - user_id 115 2147482888 int64
# - content_id 0 32736 int64
# - content_type_id 0 1 int64
# - task_container_id 0 9999 int64
# - user_answer -1 3 int64
# - answered_correctly -1 1 int64
# - prior_question_elapsed_time 0.0 300000.0 float64
# - prior_question_had_explanation False True bool


def create_folds(path, FOLDS=4):
    out = f'{path}/cache/train_{FOLDS-1}.parquet'
    if os.path.exists(out):
        print(out, 'exists')
        return

    if not os.path.exists(f'{path}/cache'):
        os.mkdir(f'{path}/cache')
        
    dtype = {'row_id':'int32', 
             'timestamp':'int64',
             'user_id':'int64',
             'content_id':'int32',
             'content_type_id':'bool',
             'task_container_id':'int16',
             'user_answer':'int32',
             'answered_correctly':'int32',
             'prior_question_elapsed_time':'float32',
             'prior_question_had_explanation':'boolean',
            }
    train = gd.read_csv(f'{path}/train.csv', dtype=dtype)

    print(train.dtypes)
    print(train.head())

    for i in train.columns:
        print(i, train[i].min(), train[i].max(), train[i].isnull().sum())


    max_timestamp_u = train[['user_id','timestamp']].groupby(['user_id']).agg(['max']).reset_index()
    max_timestamp_u.columns = ['user_id', 'max_time_stamp']
    MAX_TIME_STAMP = max_timestamp_u.max_time_stamp.max()
    max_timestamp_u.head()


    max_timestamp_u['rand_time_stamp'] = MAX_TIME_STAMP - max_timestamp_u['max_time_stamp']
    max_timestamp_u['rand_time_stamp'] = cupy.random.rand(max_timestamp_u.shape[0])*max_timestamp_u['rand_time_stamp'] 
    max_timestamp_u['rand_time_stamp'] = max_timestamp_u['rand_time_stamp'].astype('int64')
    train = train.merge(max_timestamp_u, on='user_id', how='left')
    train['viretual_time_stamp'] = train.timestamp + train['rand_time_stamp']
    train = train.drop(['rand_time_stamp', 'max_time_stamp'], axis=1)

    train = train.sort_values(['viretual_time_stamp', 'row_id']).reset_index(drop=True)

    def count_new(train, valid, col):
        valid_unique = valid[col].unique()
        train_unique = train[col].unique()
        mask = valid_unique.isin(train_unique)
        return valid_unique.shape[0] - mask.sum()


    val_size = 2500000
    for cv in range(FOLDS):
        valid = train[-val_size:]
        train = train[:-val_size]
        # check new users and new contents
        new_users = count_new(train, valid, col='user_id')
        valid_question = valid[valid.content_type_id == 0]
        train_question = train[train.content_type_id == 0]
        new_contents = count_new(train_question, valid_question, col='content_id')
        print(f'cv{cv} {train_question.answered_correctly.mean():.3f} {valid_question.answered_correctly.mean():.3f} {new_users} {new_contents}')
        train.to_parquet(f'{path}/cache/train_{cv}.parquet')
        valid.to_parquet(f'{path}/cache/valid_{cv}.parquet')
        del valid_question
        del train_question
        del valid
        
if __name__ == '__main__':
    run(__file__, False, create_folds, PATH)
