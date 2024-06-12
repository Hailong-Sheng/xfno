import torch
import numpy as np

import xfno

def main():
    config = xfno.utils.load_config('./config/geo_pino.yaml')
    
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
    train_dataset_geo_pino = xfno.dataset.DatasetGeoPINO('train', train_dataset, 
        geo, mesh_non, config.data.param_size, device=config.device)
    
    valid_dataset = xfno.dataset.Dataset('valid', geo, mesh_car, config.data.param_size, 
                                    config.data.load_cache, device=config.device)
    valid_dataset_geo_pino = xfno.dataset.DatasetGeoPINO('valid', valid_dataset, 
        geo, mesh_non, config.data.param_size, device=config.device)
    
    train_dataloader = xfno.dataset.get_dataloader(train_dataset_geo_pino, config.data.batch_size)
    valid_dataloader = xfno.dataset.get_dataloader(valid_dataset_geo_pino, config.data.batch_size)
    
    # model
    input_scale = [train_dataset.param.mean(), train_dataset.param.std()]
    model_u = xfno.model.FNO2d(config.model.in_channel, config.model.out_channel,
                               config.model.width, config.model.mode1, config.model.mode2,
                               config.model.padding, config.model.layer_num,
                               input_scale=input_scale).to(config.device)
    model_c = xfno.model.FCNet([2,32,32,32,2], bounds).to(config.device)

    # loss
    loss_c = xfno.loss.LossCT(model_c)
    loss_u = xfno.loss.LossGeoPINO(train_dataset_geo_pino.mesh_non.nx,
                                   train_dataset_geo_pino.mesh_non.hx, model_u, 100)
    error_u = xfno.loss.RelativeError(model_u)

    # train
    optimizer_u = torch.optim.Adam(model_u.parameters(), lr=config.train.lr)
    optimizer_c = torch.optim.Adam(model_c.parameters(), lr=config.train.lr)
    scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=100, gamma=0.98)
    scheduler_c = torch.optim.lr_scheduler.StepLR(optimizer_c, step_size=100, gamma=0.98)

    trainer_c = xfno.train.TrainerCT(train_dataloader, model_c, loss_c, optimizer_c, scheduler_c, 
                                     epoch_num=500, device=config.device)
    trainer_c.train()
    train_dataset_geo_pino.cordinate_transformation(model_c)

    trainer_u = xfno.train.Trainer(train_dataloader=train_dataloader, 
                                   valid_dataloader=valid_dataloader, 
                                   model=model_u, loss=loss_u, error=error_u,
                                   optimizer=optimizer_u, scheduler=scheduler_u,
                                   epoch_num=config.train.epoch_num)
    trainer_u.train()

if __name__ == '__main__':
    main()