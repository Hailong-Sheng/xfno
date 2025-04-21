import xfno

class LossFNO():
    def __init__(self, model):
        self.model = model
    
    def __call__(self, data):
        data['pred'] = self.model(data['param'], data['coord'])
        data['pred'] = data['mask']*data['pred']
        
        data['res'] = data['pred'] - data['label']
        sum_dim = tuple(range(1, data['pred'].dim()))
        data['res'] = ((data['mask']*data['res']**2).sum(dim=sum_dim))**0.5

        data['loss'] = data['res'].sum()
        return data['loss']

class LossXFNO():
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
        self.conv = xfno.FixedConv2D()
        self.conv.to(self.device)

    def __call__(self, data):
        data['pred'] = self.model(data['param'], data['coord'])
        data['pred'] = data['mask']*data['pred']
        data['pred_nb'] = self.conv(data['pred']) + data['bd_val']
        
        data['res'] = (data['weight']*data['pred_nb']).sum(1,keepdims=True) - data['right']
        sum_dim = tuple(range(1, data['pred'].dim()))
        data['res'] = ((data['mask']*data['res']**2).sum(dim=sum_dim))**0.5

        data['loss'] = data['res'].sum()
        return data['loss']

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