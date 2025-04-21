import torch
from torch_geometric.data import Batch

import xfno

class LossFNO:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, data):
        data['pre'] = self.model(data['param'], data['coord'])
        data['res'] = data['pre'] - data['label']
        data['res'] = ((data['mask']*data['res']**2).sum(dim=tuple(range(1, data['pre'].dim()))))**0.5

        data['loss'] = data['res'].sum()
        return data['loss']

class LossXFNO:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

        self.conv = xfno.model.FixedConv2D()
        self.conv.to(self.device)

    def __call__(self, data):
        data['pre']= self.model(data['param'], data['coord'])
        data['pre'] = data['pre'] * data['mask']
        data['pre_nb'] = self.conv(data['pre'])
        
        data['res'] = (data['weight']*data['pre_nb']).sum(1,keepdims=True) - data['right']
        data['res'] = ((data['mask']*data['res']**2).sum(dim=tuple(range(1, data['pre'].dim()))))**0.5

        data['loss'] = data['res'].sum()
        return data['loss']

class LossDAFNO:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, data):
        data['pre'] = self.model(data['param'], data['coord'], data['schar'])
        data['res'] = data['pre'] - data['label']
        data['res'] = ((data['mask']*data['res']**2).sum(dim=tuple(range(1, data['pre'].dim()))))**0.5

        data['loss'] = data['res'].sum()
        return data['loss']

class LossCT:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, data):
        data['c'] = self.model(data['x'])
        data['res_c'] = data['c'] - data['y']
        data['loss_c'] = ((data['mask']!=-1) * data['res_c']**2).mean()
        return data['loss_c']**0.5

class LossGINO:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, data):
        pred = self.model(data)
        res = pred - data.y

        shape = [len(data.ptr)-1, int(res.shape[0]/(len(data.ptr)-1))]
        res = (((data.mask*res).reshape(shape)**2).sum(-1))**0.5

        loss = res.sum()
        return loss

class RelativeL2Error():
    def __init__(self, model):
        self.model = model
    
    def __call__(self, data):
        data['pred'] = self.model(data['param'], data['coord'])
        data['pred'] = data['mask']*data['pred']
        
        sum_dim = tuple(range(1, data['pred'].dim()))
        numerator = (data['mask']*(data['pred']-data['label'])**2).sum(sum_dim)
        denominator = (data['mask']*data['label']**2).sum(sum_dim)
        
        error = ((numerator/denominator)**0.5).mean()
        return error

class RelativeL2ErrorDAFNO():
    def __init__(self, model):
        self.model = model
    
    def __call__(self, data):
        data['pred'] = self.model(data['param'], data['coord'], data['schar'])
        data['pred'] = data['mask']*data['pred']
        
        sum_dim = tuple(range(1, data['pred'].dim()))
        numerator = (data['mask']*(data['pred']-data['label'])**2).sum(sum_dim)
        denominator = (data['mask']*data['label']**2).sum(sum_dim)
        
        error = ((numerator/denominator)**0.5).mean()
        return error

class RelativeL2ErrorGINO:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, data):
        pred = self.model(data)
        res = pred - data.y
        
        shape = [len(data.ptr)-1, int(res.shape[0]/(len(data.ptr)-1))]
        numerator = ((data.mask*res**2).reshape(shape)).sum(-1)
        denominator = ((data.mask*data.y**2).reshape(shape)).sum(-1)

        error = ((numerator/denominator)**0.5).mean()
        return error