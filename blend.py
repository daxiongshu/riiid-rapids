import cupy as cp
from cuml.metrics import roc_auc_score

y1 = cp.load('backup/2021-01-03-20-58-05-DGX-S/mlp.npy')
y2 = cp.load('backup/2021-01-03-21-39-45-DGX-S/mlp.npy')
y3 = cp.load('backup/2021-01-03-21-56-52-DGX-S/mlp.npy')
yt = cp.load('yt.npy')
y1 = (y1+y2+y3)/3

y2 = cp.load('xgb_va.npy')
print('AUC: %.4f'%roc_auc_score(yt,y1))

y1 = y1.argsort().argsort()/y1.shape[0]
y2 = y2.argsort().argsort()/y2.shape[0]

for i in range(11):
    w = i*0.1
    yp = y1*w + y2*(1-w)
    print(i, 'AUC: %.4f'%roc_auc_score(yt,yp))
