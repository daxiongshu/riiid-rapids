from csv import DictReader
from time import time
import os
import sys
from utils import run
from collections import defaultdict
import json
from config import PATH,FOLD
TAG = 'rolling_row_tags'

def get_prev_y(name, y_mean=None, 
        ):
    
    out = name.replace('.csv', f'_{TAG}.csv')
    if os.path.exists(out):
        print(out,'exists')
        #return

    alpha = 0.01
    mean = 0.66

    if y_mean is None:
        y_mean = {}
    lasttime = defaultdict(str)
    y_prev_mean = {}
    
    fo = open(out, 'w')
    func = ['mean','min','max']
    NF = len(func)
    head = ','.join(['row_id']+['y_%s_%s'%(TAG,i) for i in func])+'\n'
    fo.write(head)
    
    with open(name, 'r') as f:
        for c,row in enumerate(DictReader(f)):
            
            t,y = row['timestamp'], row['answered_correctly']
            line = [row['row_id']]
            
            us = []
            for tg in row['tags'].split():
                u = '%s-%s'%(row['user_id'], tg)
                y_mean, y_prev_mean, lasttime = update(y_mean, y_prev_mean, lasttime, u, t, y, mean, alpha)
                if u in y_prev_mean:
                    us.append(y_prev_mean[u])
            if len(us) == 0:
                line = line + ['']*NF
            else:
                line = line + ['%.4f'%(mean_(us)),'%.4f'%(min(us)), '%.4f'%(max(us))]

            line = ",".join(line)+'\n'
            fo.write(line)
            if c%100000 == 0:
                print(c)
    fo.close()
    return y_mean

def mean_(x):
    return sum(x)/len(x)

def update(y_mean, y_prev_mean, lasttime, u, t, y, mean, alpha):
    
    if t != lasttime[u]:
        if u in y_mean:
            y_prev_mean[u] = y_mean[u]
    lasttime[u] = t
    if u not in y_mean:
        y_mean[u] = mean*(1-alpha) + alpha*int(y)
    else:
        y_mean[u] = y_mean[u]*(1-alpha) + alpha*int(y)
    return y_mean, y_prev_mean, lasttime

if __name__ == '__main__':
    #y = xgb(PATH)
    #name = sys.argv[1]
    #run(__file__, False, get_prev_y, name)
    start = time()
    name = f'{PATH}/cache/train_{FOLD}.csv'
    y_mean = get_prev_y(name)

    name = f'{PATH}/cache/valid_{FOLD}.csv'
    y_mean = get_prev_y(name, y_mean)

    with open(f'{PATH}/cache/{TAG}_{FOLD}.json', 'w') as f:
        json.dump(y_mean, f)
        
    duration = time() - start
    print(f'All done. {duration:.1f} seconds')
