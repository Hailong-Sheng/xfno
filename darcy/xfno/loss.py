import torch
from torch_geometric.data import Batch

import xfno

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

        self.conv = xfno.model.FixedConv2D()
        self.conv.to(self.device)

    def __call__(self, data):
        data['pre']= self.model(data['param'])
        data['pre'] = data['pre'] * data['mask']
        data['pre_nb'] = self.conv(data['pre'])
        
        data['res'] = (data['wei_u']*data['pre_nb']).sum(1,keepdims=True) - data['r']
        data['loss'] = (data['res']**2).sum()
        return data['loss']

class LossCT:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, data):
        data['c'] = self.model(data['x'])
        data['res_c'] = data['c'] - data['y']
        data['loss_c'] = ((data['mask']!=-1) * data['res_c']**2).mean()
        return data['loss_c']**0.5

class LossGeoPINO:
    def __init__(self, nx, hx, model, beta):
        self.nx = nx
        self.hx = hx
        self.model = model
        self.beta = beta
        
    def __call__(self, data):
        data['u'] = self.model(data['param'])
        data['uy0'] = torch.zeros(data['param'].shape[0],1,self.nx[0]+1,self.nx[1]+1).to(data['param'].device)
        data['uy1'] = torch.zeros(data['param'].shape[0],1,self.nx[0]+1,self.nx[1]+1).to(data['param'].device)
        data['uy0y0'] = torch.zeros(data['param'].shape[0],1,self.nx[0]+1,self.nx[1]+1).to(data['param'].device)
        data['uy0y1'] = torch.zeros(data['param'].shape[0],1,self.nx[0]+1,self.nx[1]+1).to(data['param'].device)
        data['uy1y0'] = torch.zeros(data['param'].shape[0],1,self.nx[0]+1,self.nx[1]+1).to(data['param'].device)
        data['uy1y1'] = torch.zeros(data['param'].shape[0],1,self.nx[0]+1,self.nx[1]+1).to(data['param'].device)
        data['uy0'][:,:,1:self.nx[0],:] = (
            data['u'][:,:,2:self.nx[0]+1,:] - data['u'][:,:,0:self.nx[0]-1,:]) / (2*self.hx[0])
        data['uy1'][:,:,:,1:self.nx[1]] = (
            data['u'][:,:,:,2:self.nx[1]+1] - data['u'][:,:,:,0:self.nx[1]-1]) / (2*self.hx[1])
        data['uy0y0'][:,:,1:self.nx[0],:] = (
            data['u'][:,:,2:self.nx[0]+1,:] - 2*data['u'][:,:,1:self.nx[0]+0,:] + 
            data['u'][:,:,0:self.nx[0]-1,:]) / (self.hx[0]**2)
        data['uy0y1'][:,:,:,1:self.nx[1]] = (
            data['uy0'][:,:,:,2:self.nx[1]+1] - data['uy0'][:,:,:,0:self.nx[1]-1]) / (2*self.hx[1])
        data['uy1y0'][:,:,1:self.nx[0],:] = (
            data['uy1'][:,:,2:self.nx[0]+1,:] - data['uy1'][:,:,0:self.nx[0]-1,:]) / (2*self.hx[0])
        data['uy1y1'][:,:,:,1:self.nx[1]] = (
            data['u'][:,:,:,2:self.nx[1]+1] - 2*data['u'][:,:,:,1:self.nx[1]+0] + 
            data['u'][:,:,:,0:self.nx[1]-1]) / (self.hx[1]**2)

        data['ux0'] = data['uy0']*data['c0x0'] + data['uy1']*data['c1x0']
        data['ux1'] = data['uy0']*data['c0x1'] + data['uy1']*data['c1x1']
        data['ux0x0'] = (data['uy0y0']*(data['c0x0'])**2 + data['uy0y1']*data['c1x0']*data['c0x0'] + 
                         data['uy0']*data['c0x0x0'] + 
                         data['uy1y1']*(data['c1x0'])**2 + data['uy1y0']*data['c0x0']*data['c1x0'] + 
                         data['uy1']*data['c1x0x0'])
        data['ux1x1'] = (data['uy0y0']*(data['c0x1'])**2 + data['uy0y1']*data['c1x1']*data['c0x1'] + 
                         data['uy0']*data['c0x1x1'] + 
                         data['uy1y1']*(data['c1x1'])**2 + data['uy1y0']*data['c0x1']*data['c1x1'] + 
                         data['uy1']*data['c1x1x1'])

        data['res_in'] = -(data['ax0']*data['ux0'] + data['a']*data['ux0x0'] + 
                           data['ax1']*data['ux1'] + data['a']*data['ux1x1'])
        data['res_bd'] = data['u']
        
        loss = (data['res_in']**2 * (data['mask']==1)).mean() + \
               self.beta * (data['res_bd']**2 * (data['mask']==0)).mean()
        return loss**0.5

class LossGINO:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, data):
        x = self.model(data)
        loss = (data.mask*(x-data.y)**2).sum()
        return loss

class RelativeError:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, dataset):
        pre = self.model(dataset.param)
        res = pre - dataset.label
        err = ((dataset.mask*res**2).sum() / 
               (dataset.mask*dataset.label**2).sum())**0.5
        return err

class RelativeErrorGINO:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, dataset):
        batch = Batch.from_data_list(dataset.graph)
        batch = batch.to(dataset.device)
        pre = self.model(batch)
        res = pre - batch.y
        err = ((batch.mask*res**2).sum() / 
               (batch.mask*batch.y**2).sum())**0.5
        return err