import torch
import numpy as np
import time

import xfno

def main():
    config = xfno.utils.load_config('./config/gino.yaml')
    
    # geometry
    dim = 2
    bounds = torch.tensor(config.geo.bounds).reshape(dim,2)
    center = torch.tensor(config.geo.center)
    geo = xfno.geometry.Geometry(bounds, center, config.geo.radius)
    
    # mesh
    nx = torch.tensor(config.mesh.nx).int()
    msh_car = xfno.mesh.MeshCartesian(geo, bounds, nx)
    msh_non = xfno.mesh.MeshNonCartesian(geo, bounds, nx)
    
    # dataset
    train_dataset = xfno.dataset.Dataset('train', geo, msh_car, config.data.param_size, 
                                         config.data.load_cache, device=config.device)
    train_dataset_gino = xfno.dataset.DatasetGINO('train', train_dataset, 
        geo, msh_non, config.data.param_size, device=config.device)
    
    valid_dataset = xfno.dataset.Dataset('valid', geo, msh_car, config.data.param_size, 
                                         config.data.load_cache, device=config.device)
    valid_dataset_gino = xfno.dataset.DatasetGINO('valid', valid_dataset, 
        geo, msh_non, config.data.param_size, device=config.device)
    
    train_dataloader = xfno.dataset.PyGDataLoader(train_dataset_gino, config.data.batch_size)
    valid_dataloader = xfno.dataset.PyGDataLoader(valid_dataset_gino, batch_size=50)
    
    # encoder
    width = 32
    ker_width = 100
    depth = 1
    edge_features = 6
    node_features = 1
    encoder = xfno.model.KernelNN(width, ker_width, depth, edge_features, in_width=node_features).to(config.device)
    
    # process
    fno = xfno.model.FNO2d(in_channel=1, out_channel=1, width=32, 
                           mode1=12, mode2=12, padding=8, layer_num=4).to(config.device)
    
    # decoder
    decoder = xfno.model.KernelNN(width, ker_width, depth, edge_features, in_width=node_features).to(config.device)
    
    # model
    input_scale = [train_dataset.param.mean(), train_dataset.param.std()]
    model = xfno.model.GINO(encoder, fno, decoder, input_scale,
                            train_dataset_gino.mesh.cen_size, 
                            train_dataset_gino.mesh.cen_size, 
                            train_dataset_gino.mesh.nx)
    
    # loss
    loss = xfno.loss.LossGINO(model=model)
    error = xfno.loss.RelativeErrorGINO(model=model)

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    gamma = config.train.decay_rate**(1.0/config.train.decay_step)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    trainer = xfno.train.Trainer(train_dataloader=train_dataloader, 
                            valid_dataloader=valid_dataloader, 
                            model=model, loss=loss, error=error,
                            optimizer=optimizer, scheduler=scheduler,
                            epoch_num=config.train.epoch_num)
    trainer.train()
    '''
    #np.savetxt('x.txt', x.cpu().detach())
    #np.savetxt('edge_index.txt', edge_index.cpu().detach())
    
    # evaluate
    param = valid_dataset_gino.graph[0]
    param = param.to(config.device)
    model = model.to(config.device)

    t0 = time.time()
    pred = model(param)
    t1 = time.time()
    print(f'time: {t1-t0}s')
    '''
if __name__=='__main__':
    main()