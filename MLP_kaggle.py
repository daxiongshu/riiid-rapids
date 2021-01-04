import torch
import numpy as np


class MLP(torch.nn.Module):
    
    def __init__(self, N, Hs, C, **params):
        
        super(MLP, self).__init__()
        self.params = params
        
        if hasattr(Hs, '__iter__'):
            Hs = list(Hs)
        else:
            Hs = [Hs]
        Hs = [N] + Hs + [C]
        
        layers = []
        for c,H in enumerate(Hs[:-1]):
            layer = torch.nn.Linear(in_features=H, out_features=Hs[c+1])
            layers.append(layer)
        self.layers = torch.nn.ModuleList(layers)
        self.dp = torch.nn.Dropout(p=0.2)
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = activate(x, self.params['activation'])
            #x = self.dp(x)
        layer = self.layers[-1]
        return layer(x)
    
def activate(x, act):
    if act == 'identity':
        return x
    elif act == 'relu':
        return torch.nn.functional.relu(x)
    elif act == 'tanh':
        return torch.nn.functional.tanh(x)
    elif act == 'sigmoid':
        return torch.nn.functional.sigmoid(x)
    else:
        assert 0, "unknown activation "+act


def get_default_params():
   return {
       'hidden_layer_sizes': (100, ), 
       'norm':False,
       'activation':'relu',
       'solver':'adam', 
       'alpha':0.0001,
       'batch_size':'auto', 
       'learning_rate':'constant', 
       'learning_rate_init':0.001, 
       'max_iter':200, 
       'shuffle':True, 
       'random_state':None, 
       'tol':0.0001, 
       'verbose':False,
       'momentum':0.9, 
       'nesterovs_momentum':True, 
       'early_stopping':False, 
       'validation_fraction':0.1, 
       'beta_1':0.9, 
       'beta_2':0.999, 
       'epsilon':1e-08, 
       'n_iter_no_change':10, 
       'max_fun':15000,
   }

class MLPbase:
    
    def __init__(self, **params):
        self.params = get_default_params()
        self.params.update(params)
        self.task = None
        
    def load_model(self, N, C):
        model = MLP(N, self.params['hidden_layer_sizes'], C, **self.params)
        model.load_state_dict(torch.load(self.params['model_path'], map_location=torch.device('cpu') ))
        self.model = model
        return model
    
    
class MLPClassifier(MLPbase):
    
    def __init__(self, **params):
        super().__init__(**params)
        self.task = 'classification'
        
    def predict_proba(self, X):
        yp = self.model(X).detach().numpy()
        yp = softmax(yp)
        return yp
    
def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    s = ex.sum(axis=1, keepdims=True)
    res =  ex/s
    eps = 1e-5
    return np.clip(res, eps, 1-eps)