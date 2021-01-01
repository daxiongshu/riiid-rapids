from csv import DictReader
from time import time
import os
import sys
from utils import run
from collections import defaultdict

def get_prev_y(name):
    
    out = name.replace('.csv', '_prev_y.csv')
    if os.path.exists(out):
        print(out,'exists')
        return
    
    y_count = defaultdict(int)
    y_sum = defaultdict(int)
    y_mean = {}
    lasttime = defaultdict(str)
    
    fo = open(out, 'w')
    fo.write('row_id,y_cum\n')
    
    with open(name, 'r') as f:
        for c,row in enumerate(DictReader(f)):
            u,t,y = row['user_id'], row['timestamp'], row['answered_correctly']
            if t != lasttime[u]:
                try:
                    y_mean[u] = '%.4f'%(y_sum[u]/y_count[u])
                except:
                    y_mean[u] = ''
            lasttime[u] = t
            y_count[u] += 1
            y_sum[u] += int(y)
            
            line = "%s,%s\n"%(row['row_id'],y_mean[u])
            fo.write(line)
            if c%100000 == 0:
                print(c)
    fo.close()

if __name__ == '__main__':
    #y = xgb(PATH)
    name = sys.argv[1]
    run(__file__, False, get_prev_y, name)
