import cupy as cp
import pandas as pd
from xgb_clf import XGBClassifier
from cuml.metrics import roc_auc_score
from config import PATH,FOLD
from utils import run
from fe import get_data

def xgb(path):
    
    X,y,Xt,yt,cols = get_data()
    params = {'n_estimators': 100,
              'eta': 0.1,
              'early_stopping_rounds': 10,
              'max_depth': 7, 
              'colsample_bytree': 0.7,
              'subsample': 0.5,
              'verbosity': 1,
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'validation_fraction': 0,
             }

    clf = XGBClassifier(**params)
    clf.fit(X, y, Xt, yt)
    clf.clf.save_model(f'{path}/cache/xgb.json')

    yp = clf.predict_proba(Xt)
    #return yp
    print(cols)
    return roc_auc_score(yt, yp)

if __name__ == '__main__':
    #y = xgb(PATH)
    run(__file__, True, xgb, PATH)
