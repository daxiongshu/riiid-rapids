from csv import DictReader
from time import time
import os
import sys
from utils import run
from collections import defaultdict
import json
from config import PATH,FOLD
TAG = 'rolling_time'

def get_prev_y(name, y_mean=None, 
               y_current=None,
               ucols=['content_id', 'part']
        ):
    
    out = name.replace('.csv', f'_{TAG}.csv')
    if os.path.exists(out):
        print(out,'exists')
        #return

    alpha = 0.01
    mean = 0.66

    if y_mean is None:
        y_mean = {i:{} for i in ['user_id']+ucols}
        y_current = {i:{} for i in ['user_id']+ucols}
    lasttime = {i:defaultdict(str) for i in ['user_id']+ucols}
    
    fo = open(out, 'w')
    head = ','.join(['row_id']+['y_%s_%s'%(TAG,i) for i in ['user_id']+ucols])+'\n'
    fo.write(head)
    
    with open(name, 'r') as f:
        for c,row in enumerate(DictReader(f)):
            
            t,y = row['timestamp'], row['answered_correctly']
            line = [row['row_id']]
            
            for col in ucols:
                row[col] = '%s-%s'%(row['user_id'], row[col])
                
            for col in ['user_id']+ucols:
                u = row[col]
                y_mean[col], y_current[col], lasttime[col], res = update(y_mean[col], y_current[col], lasttime[col], u, t, y, mean, alpha)
                line.append(res)
            line = ",".join(line)+'\n'
            fo.write(line)
            if c%100000 == 0:
                print(c)
    fo.close()
    return y_mean, y_current

def get_mean(s):
    return sum(s)/len(s)

def update(y_mean, y_current, lasttime, u, t, y, mean, alpha):
    res = ''
    if t != lasttime[u]:
        if u in y_current and len(y_current[u]):
            ym = get_mean(y_current[u])
            y_mean = update_ymean(y_mean, u, ym, mean, alpha)
            res = '%.4f'%y_mean[u]
        y_current[u] = []
    y_current[u].append(int(y))  
    lasttime[u] = t
    
    return y_mean, y_current, lasttime, res

def update_ymean(y_mean, u, y, mean, alpha):
    if u not in y_mean:
        y_mean[u] = mean*(1-alpha) + alpha*y
    else:
        y_mean[u] = y_mean[u]*(1-alpha) + alpha*y
    return y_mean
        
if __name__ == '__main__':
    #y = xgb(PATH)
    #name = sys.argv[1]
    #run(__file__, False, get_prev_y, name)
    start = time()
    name = f'{PATH}/cache/train_{FOLD}.csv'
    y_mean, y_current = get_prev_y(name)

    name = f'{PATH}/cache/valid_{FOLD}.csv'
    y_mean, y_current = get_prev_y(name, y_mean, y_current)

    with open(f'{PATH}/cache/{TAG}_{FOLD}.json', 'w') as f:
        json.dump(y_mean, f)
        
    duration = time() - start
    print(f'All done. {duration:.1f} seconds')
