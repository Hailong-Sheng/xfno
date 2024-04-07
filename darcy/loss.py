import torch
import network

class RelativeError:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, data):
        data['pre'] = self.model(data['param'])
        data['res'] = data['pre'] - data['label']
        data['err'] = ((data['mask']*data['res']**2).sum() / 
                       (data['mask']*data['label']**2).sum())**0.5
        return data['err']

class Loss:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, data):
        data['pre'] = self.model(data['param'])
        data['res'] = data['pre'] - data['label']
        data['loss'] = (data['mask']*data['res']**2).sum()
        return data['loss']

class LossXFNO:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

        self.conv = network.FixedConv2D()
        self.conv.to(self.device)

    def __call__(self, data):
        data['pre']= self.model(data['param'])
        data['pre'] = data['pre'] * data['mask']
        data['pre_nb'] = self.conv(data['pre'])
        
        data['res'] = (data['wei_u']*data['pre_nb']).sum(1,keepdims=True) - data['r']
        data['loss'] = (data['res']**2).sum()
        return data['loss']
        