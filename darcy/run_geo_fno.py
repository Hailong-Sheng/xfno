import torch
import numpy as np

import xfno

def main():
    config = xfno.utils.load_config('./config/geo_fno.yaml')
    
    # geometry
    dim = 2
    bounds = torch.tensor(config.geo.bounds).reshape(dim,2)
    center = torch.tensor(config.geo.center)
    geo = xfno.geometry.Geometry(bounds, center, config.geo.radius)
    
    # mesh
    nx = torch.tensor(config.mesh.nx).int()
    mesh_car = xfno.mesh.MeshCartesian(geo, bounds, nx)
    mesh_non = xfno.mesh.MeshNonCartesian(geo, bounds, nx)
    
    # dataset
    train_dataset = xfno.dataset.Dataset('train', geo, mesh_car, config.data.param_size, 
                                         config.data.load_cache, device=config.device)
    train_dataset_geo_fno = xfno.dataset.DatasetGeoFNO('train', train_dataset, 
        geo, mesh_non, config.data.param_size, device=config.device)
    
    valid_dataset = xfno.dataset.Dataset('valid', geo, mesh_car, config.data.param_size, 
                                         config.data.load_cache, device=config.device)
    valid_dataset_geo_fno = xfno.dataset.DatasetGeoFNO('valid', valid_dataset, 
        geo, mesh_non, config.data.param_size, device=config.device)
    
    train_dataloader = xfno.dataset.DataLoader(train_dataset_geo_fno, config.data.batch_size)
    
    # model
    input_scale = [train_dataset.param.mean(), train_dataset.param.std()]
    model = xfno.model.FNO2d(config.model.in_channel, config.model.out_channel,
                             config.model.width, config.model.mode1, config.model.mode2,
                             config.model.padding, config.model.layer_num,
                             input_scale=input_scale)

    # loss
    loss = xfno.loss.Loss(model=model)
    error = xfno.loss.RelativeError(model=model)
    
    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        step_size=config.train.step_size, gamma=config.train.gamma)

    trainer = xfno.train.Trainer(train_dataloader=train_dataloader, 
                                 valid_dataset=valid_dataset_geo_fno, 
                                 model=model, loss=loss, error=error,
                                 optimizer=optimizer, scheduler=scheduler,
                                 epoch_num=config.train.epoch_num)
    trainer.train()

if __name__ == "__main__":
    main()