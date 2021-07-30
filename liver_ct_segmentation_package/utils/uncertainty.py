import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def set_dropout(model, drop_rate=0.5):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def standard_prediction(model, X):
    model = model.eval()
    outputs = model(Variable(X))
    pred = torch.argmax(outputs, dim=1).float()
    
    return pred #.numpy()

def predict_dist(model, X, T=1000):
    
    model = model.train()
    y1 = []
    y2 = []
    for _ in range(T):
        print("T: " + str(_))
        _y1 = model(Variable(X))
        _y2 = F.softmax(_y1, dim=1)

        del _y1
        torch.cuda.empty_cache()

        #print("softmax shape:" + str(_y2.shape))

        #_y1 = _y1.squeeze(0) #remove batch dim
        _y2 = _y2.squeeze(0)

        #y1.append(_y1) #.data.numpy())
        y2.append(_y2) #.data.numpy())

    #y1 = torch.stack(y1)
    y2 = torch.stack(y2)  
    #print("y2 shape:" + str(y2.shape))

    return y2

def monte_carlo_dropout_proc(model, x, T=1000, dropout_rate=0.5):

    standard_pred = standard_prediction(model, x)
    #print("standard_pred shape:" + str(standard_pred.shape))

    set_dropout(model, drop_rate=dropout_rate)
    
    softmax_dist = predict_dist(model, x, T)
    
    #pred_var = torch.var(softmax_dist, 0)
    pred_std = torch.std(softmax_dist, dim=0)
    del softmax_dist
    torch.cuda.empty_cache()
    
    #print("pred std shape:" + str(pred_std.shape))

    pred_std = pred_std.gather(0, standard_pred.long()).squeeze(0)
    #print("pred std gather shape:" + str(pred_std.shape))

    model = model.eval()

    return pred_std


