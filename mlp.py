import cupy as cp
import pandas as pd
from cuml.metrics import roc_auc_score
from config import PATH,FOLD
from utils import run
from fe import get_data
from MLP_cuml import MLPClassifier 

def mlp(path):
    
    X,y,Xt,yt,cols = get_data(fillna=0, norm=True, dropids=True)
    X = cp.asarray(X)
    y = cp.asarray(y)
    Xt = cp.asarray(Xt)
    yt = cp.asarray(yt)
    params = {
                 'learning_rate_init':0.005,
                 'verbose': True,
                 'hidden_layer_sizes': (64,64,64),
                 'max_iter':100,
                 'batch_size':4096,
                 'shuffle':True,
                 'alpha':0.000,
                 'model_path':f'{path}/cache/mlp.pth',
             }

    clf = MLPClassifier(**params)
    clf.fit(X, y, Xt, yt)

    yp = clf.predict_proba(Xt)[:,1]
    score = roc_auc_score(yt, yp)
    print('AUC: %.4f'%score)

    cp.save('mlp.npy', yp)
    #return yp
    print(cols)
    return score

if __name__ == '__main__':
    #y = xgb(PATH)
    run(__file__, True, mlp, PATH)
