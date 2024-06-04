import torch

class Loss4XFNO():
    """ loss function for XFNO """
    def __init__(self, model, conv):
        self.model = model
        self.conv = conv

    def __call__(self, data):
        ''''''
        data['u'] = self.model(data['parm'])
        data['u0'] = data['u'][:,0:1,...]
        data['u1'] = data['u'][:,1:2,...]
        data['p'] = data['u'][:,2:3,...]
        
        data['u0'] = data['u0'] * data['mask']
        data['u1'] = data['u1'] * data['mask']
        data['p'] = data['p'] * data['mask']
        
        data['u0_nb '] = self.conv(data['u0']) + data['v0']
        data['u1_nb '] = self.conv(data['u1']) + data['v1']
        data['p_nb '] = self.conv(data['p']) + data['v2']
        
        data['wei0_u0'] = (data['wei0_u0_diff'] + 
            data['wei0_u0_conv_u0']*data['u0'] + data['wei0_u0_conv_u1']*data['u1'])
        data['wei1_u1'] = (data['wei1_u1_diff'] + 
            data['wei1_u1_conv_u0']*data['u0'] + data['wei1_u1_conv_u1']*data['u1'])

        data['res0'] = (data['wei0_u0'] * data['u0_nb '] + 
                        data['wei0_p'] * data['p_nb ']).sum(1,keepdims=True) - data['r0']
        data['res1'] = (data['wei1_u1'] * data['u1_nb '] + 
                        data['wei1_p'] * data['p_nb ']).sum(1,keepdims=True) - data['r1']
        
        c0 = data['wei0_u0'][:,4:5,:,:] * data['mask'] + (1-data['mask'])
        c1 = data['wei1_u1'][:,4:5,:,:] * data['mask'] + (1-data['mask'])
        data['wei0_u0_nb'] = -data['wei0_u0'] / c0
        data['wei0_p_nb']  = -data['wei0_p']  / c0
        data['r0_nb']      =  data['r0']      / c0
        data['wei1_u1_nb'] = -data['wei1_u1'] / c1
        data['wei1_p_nb']  = -data['wei1_p']  / c1
        data['r1_nb']      =  data['r1']      / c1
        data['wei0_u0_nb'][:,4,:,:] = 0
        data['wei1_u1_nb'][:,4,:,:] = 0
        
        data['u0_intp'] = (data['wei0_u0_nb'] * data['u0_nb '] + 
                           data['wei0_p_nb'] * data['p_nb ']).sum(1,keepdims=True) + data['r0_nb']
        data['u1_intp'] = (data['wei1_u1_nb'] * data['u1_nb '] + 
                           data['wei1_p_nb'] * data['p_nb ']).sum(1,keepdims=True) + data['r1_nb']

        data['res2'] = (data['wei2_u0'] * (self.conv(data['u0_intp']) + data['v0']) + 
                        data['wei2_u1'] * (self.conv(data['u1_intp']) + data['v1'])
                       ).sum(1,keepdims=True) - data['r2']
        
        loss = ((data['mask']*data['res0']**2).sum() + 
                (data['mask']*data['res1']**2).sum() + 
                (data['mask']*data['res2']**2).sum())
        return loss**0.5

class Error4XFNO():
    """ relative l-2 error """
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        
    def relative_error(self, u, ua):
        return (((u-ua)**2).sum() / ((ua**2).sum()+1e-16)) ** 0.5
    
    def __call__(self):
        self.dataset.u = self.model(self.dataset.parm)
        self.dataset.u0 = self.dataset.u[:,0:1,...]
        self.dataset.u1 = self.dataset.u[:,1:2,...]
        self.dataset.p = self.dataset.u[:,2:3,...]
        
        error_u0 = self.relative_error(self.dataset.u0*self.dataset.mask,
                                       self.dataset.u0a*self.dataset.mask)
        error_u1 = self.relative_error(self.dataset.u1*self.dataset.mask,
                                       self.dataset.u1a*self.dataset.mask)
        error_p = self.relative_error(self.dataset.p*self.dataset.mask,
                                      self.dataset.pa*self.dataset.mask)
        return error_u0.detach().cpu(), error_u1.detach().cpu(), error_p.detach().cpu()