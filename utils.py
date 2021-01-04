import os
from datetime import datetime
from time import time
from config import DEVICE, BACKUP, LOG, INFO
import sys

def backup(filename, duration, score):
    name = make_name()
    bpath = f"{BACKUP}/{name}"
    if os.path.exists(bpath):
        msg = f"{bpath} already exists. Please manually backup."
        return
    mkdir(bpath)
    write_log(LOG, name, duration, score, f"{filename} {INFO}")
    cmd = f"cp *.py run.sh run.log {bpath}/"
    os.system(cmd)
    cmd = f"mv *.pth {bpath}/"
    os.system(cmd)
    print(f"{bpath} backup done")

def mkdir(bpath):
    if not os.path.exists(BACKUP):
        os.mkdir(BACKUP)
    os.mkdir(bpath)

def write_log(log_name, run_name, duration, score, info):
    if not os.path.exists(log_name):
        with open(log_name, 'w') as f:
            header = 'run_name,duration,score,info\n'
            f.write(header)

    with open(log_name, 'a') as f:
        line = ','.join([run_name, f"{duration:.1f}", f"{score:.4f}", info])
        f.write(line+'\n')

def get_clock_time():
    clock = "{}".format(datetime.now()).replace(' ','-').replace(':','-').split('.')[0]
    return clock

def make_name():
    clock = get_clock_time()
    return f"{clock}-{DEVICE}"

def run(filename, do_backup, func, *args):
    start = time()
    tag = f'python {filename}'
    print(tag, '...')
    score = func(*args)
    if score is not None and type(score) == float:
        print(f"score: {score:.4f}")
    duration = time() - start
    if do_backup and score is not None:
        backup(filename, duration, score)
    print(f'{tag} done! Time:{duration: .1f} seconds')
    return score

def parquet2csv(name):
    import pandas as pd
    out = name.replace('.parquet', '.csv')
    if os.path.exists(out):
        print(out, 'exists')
        return
    df = pd.read_parquet(name)
    print(df.shape, out)
    df.to_csv(out, index=False)

if __name__ == '__main__':
    name = sys.argv[1]
    parquet2csv(name)
