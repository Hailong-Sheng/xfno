import torch
import numpy as np

import xfno

def main():
    config = xfno.utils.load_config('./config/xfno.yaml')
    
    # geometry
    dim = 2
    bounds = torch.tensor(config.geo.bounds).reshape(dim,2)
    center = torch.tensor(config.geo.center)
    geo = xfno.geometry.Geometry(bounds, center, config.geo.radius)
    
    # mesh
    nx = torch.tensor(config.mesh.nx).int()
    mesh = xfno.mesh.MeshCartesian(geo, bounds, nx)
    
    # dataset
    train_dataset = xfno.dataset.Dataset('train', geo, mesh, config.data.param_size, 
                                         config.data.load_cache, device=config.device)
    valid_dataset = xfno.dataset.Dataset('valid', geo, mesh, config.data.param_size, 
                                         config.data.load_cache, device=config.device)
    
    train_dataloader = xfno.dataset.DataLoader(train_dataset, config.data.batch_size)
    
    # model
    input_scale = [train_dataset.param.mean(), train_dataset.param.std()]
    model = xfno.model.FNO2d(config.model.input_channel, config.model.output_channel,
        config.model.width, config.model.mode1_num, config.model.mode2_num,
        config.model.padding, config.model.layer_num, input_scale=input_scale)
    
    # loss
    loss = xfno.loss.LossXFNO(model=model, device=config.device)
    error = xfno.loss.RelativeError(model=model)
    
    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        step_size=config.train.step_size, gamma=config.train.gamma)
    
    trainer = xfno.train.Trainer(train_dataloader=train_dataloader, 
                                 valid_dataset=valid_dataset, 
                                 model=model, loss=loss, error=error,
                                 optimizer=optimizer, scheduler=scheduler,
                                 epoch_num=config.train.epoch_num,
                                 ckpt_name=config.ckpt.name, ckpt_dirt=config.ckpt.dirt,
                                 result_dirt=config.output.dirt, device=config.device)
    trainer.train()
    
    # save result
    valid_dataset.pred = model(valid_dataset.param)
    param = valid_dataset.param.reshape(valid_dataset.param_size,valid_dataset.mesh.c_size)
    label = valid_dataset.label.reshape(valid_dataset.param_size,valid_dataset.mesh.c_size)
    pred = valid_dataset.pred.reshape(valid_dataset.param_size,valid_dataset.mesh.c_size)
    mask = valid_dataset.mask.reshape(1,valid_dataset.mesh.c_size)
    np.savetxt(f'{config.output.dirt}/param.txt',param.cpu().detach().numpy())
    np.savetxt(f'{config.output.dirt}/label.txt',label.cpu().detach().numpy())
    np.savetxt(f'{config.output.dirt}/pred.txt',pred.cpu().detach().numpy())
    np.savetxt(f'{config.output.dirt}/mask.txt',mask.cpu().detach().numpy())

if __name__ == "__main__":
    main()