import os
import torch
import numpy as np
import time

import xfno

def generate_parm(center_bounds, u0_inlet_bounds, parm_size, load_cache):
    if load_cache:
        center = torch.load('./cache/center.pt')
        u0_inlet = torch.load('./cache/u0_inlet.pt')
        print(center.shape)
    else:
        dim = 2
        center = center_bounds[0,:] + torch.rand(parm_size,dim) * (
            center_bounds[1,:]-center_bounds[0,:])
        u0_inlet = u0_inlet_bounds[0] + torch.rand(parm_size) * (
            u0_inlet_bounds[1]-u0_inlet_bounds[0])
        print(center.shape)

        os.makedirs('./cache', exist_ok=True)
        torch.save(center, './cache/center.pt')
        torch.save(u0_inlet, './cache/u0_inlet.pt')

    return center, u0_inlet

def main():
    # configurations
    config = xfno.utils.load_config('./config/xfno.yaml')
    dtype = xfno.utils.set_torch_dtype(config.dtype)

    dim = 2
    bounds = torch.tensor(config.geo.bounds).reshape(dim,2)
    center_bounds = torch.tensor(config.param.center_bounds).reshape(dim,2)
    u0_inlet_bounds = torch.tensor(config.param.u0_inlet_bounds)
    center, u0_inlet = generate_parm(center_bounds, u0_inlet_bounds, 
                                     config.param.size, config.load_cache)
    
    # geometry and mesh
    nx = torch.tensor(config.geo.nx).int()
    geo = xfno.geometry.Geometry(bounds, center, config.geo.radius)
    msh = xfno.mesh.Mesh(geo, bounds, nx)
    
    # dataset
    tr_dataset = xfno.dataset.TrainSet4XFNO(geo, msh, config.param.re, u0_inlet, dtype,
                                            config.device, config.load_cache)
    tr_dataset.to(config.device)
    te_dataset = xfno.dataset.TestSet4XFNO(config.data.name, config.data.dirt,
                                           dtype, config.device)
    
    # dataloader
    tr_dataloader = xfno.dataset.DataLoader(tr_dataset, config.train.batch_size)
    
    # model
    model = xfno.model.FNO2d(config.model.modes1, config.model.modes2,
                           config.model.width).to(config.device)
    conv = xfno.model.FixedConv2D().to(config.device)
    print(f'amount of parameters: {xfno.model.count_params(model)}')
    
    # loss and error
    loss = xfno.loss.Loss4XFNO(model, conv)
    error = xfno.loss.Error4XFNO(te_dataset, model)
    
    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)

    # train
    trainer = xfno.train.Trainer(tr_dataloader, model, loss, error, optimizer, scheduler, 
                                 config.train.epochs_o, config.train.epochs_i)
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    print('train time: %.2f' %(elapsed))

if __name__=='__main__':
    main()