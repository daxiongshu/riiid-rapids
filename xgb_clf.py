import xgboost as xgb
import cupy as cp
from cuml.preprocessing.model_selection import train_test_split 

def get_default_params():
    return  {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'tree_method': 'gpu_hist',
            'verbosity': 0,
            'early_stopping':False,
            'validation_fraction':0,
            'early_stopping_rounds':10,
        }

def print_shape(*x):
    for i in x:
        print(i.shape, end=' ')
    print()

class XGBbase:
    
    def __init__(self, **params):
        self.params = get_default_params()
        self.params.update(params)
        
    def fit(self, X, y, Xt=None, yt=None):
        
        test_size = self.params['validation_fraction']
        num_boost_round = self.params['n_estimators']
        
        #ext_params = ['early_stopping', 'early_stopping_rounds', 
        #              'n_estimators', 'silent', 'validation_fraction', 'verbose']
        
        if test_size >0:
            X, Xt, y, yt = train_test_split(X, y, test_size = test_size)
        if Xt is not None:
            dtrain = xgb.DMatrix(data=X, label=y)
            dvalid = xgb.DMatrix(data=Xt, label=yt)
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')] 
            early_stopping_rounds = self.params['early_stopping_rounds']
            self.clf = xgb.train(self.params, dtrain=dtrain,
                num_boost_round=num_boost_round,evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=self.params.get('verbose_eval', 10))
        else:
            dtrain = xgb.DMatrix(data=X, label=y)
            self.clf = xgb.train(self.params, dtrain=dtrain,
                num_boost_round=num_boost_round)

        return self
    
    def predict(self, X):
        self.clf.set_param({'predictor': 'gpu_predictor'})
        dtest = xgb.DMatrix(data=X)
        yp = self.clf.predict(dtest)
        yp = cp.asarray(yp)
        return yp
    
class XGBClassifier(XGBbase):
    
    def __init__(self, **params):
        super().__init__(**params)
        
        
    def predict_proba(self, X):
        return super().predict(X)
    
    def predict(self, X):
        yp = super().predict(X)
        if len(yp.shape) == 2:
            yp = cp.argmax(yp, axis=1)
        else:
            yp = yp>0.5
        return yp
    
class XGBRegressor(XGBbase):
    
    def __init__(self, **params):
        params['objective'] = 'reg:squarederror'
        params['eval_metric'] = 'rmse'
        super().__init__(**params)
