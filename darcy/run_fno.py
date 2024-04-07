import torch
import numpy as np

import geometry
import mesh
import dataset
import network
import loss
import train
import utils

def main():
    config = utils.load_config('./config/fno.yaml')

    # geometry
    dim = 2
    bounds = torch.tensor(config.geo.bounds).reshape(dim,2)
    center = torch.tensor(config.geo.center)
    geo = geometry.Geometry(bounds, center, config.geo.radius)

    # mesh
    nx = torch.tensor(config.mesh.nx).int()
    msh = mesh.Mesh(geo, bounds, nx)

    # dataset
    train_dataset = dataset.Dataset('train', msh, config.data.param_size, 
                                    config.data.load_cache, device=config.device)
    val_dataset = dataset.Dataset('val', msh, config.data.param_size, 
                                  config.data.load_cache, device=config.device)
    
    train_dataloader = dataset.get_dataloader(train_dataset, config.data.batch_size)
    val_dataloader = dataset.get_dataloader(val_dataset, batch_size=50)
    
    # model
    input_scale = [train_dataset.param.mean(), train_dataset.param.std()]
    fno = network.FNO2d(config.model.in_channel, config.model.out_channel,
                        config.model.width, config.model.mode1, config.model.mode2,
                        config.model.padding, config.model.layer_num,
                        input_scale=input_scale)

    # loss
    loss_fn = loss.Loss(model=fno)
    error_fn = loss.RelativeError(model=fno)
    
    # train
    optimizer = torch.optim.Adam(fno.parameters(), lr=config.train.lr)
    gamma = config.train.decay_rate**(1.0/config.train.decay_step)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    trainer = train.Trainer(train_dataloader=train_dataloader, 
                            val_dataloader=val_dataloader, 
                            model=fno, loss=loss_fn, error=error_fn,
                            optimizer=optimizer, scheduler=scheduler,
                            epoch_num=config.train.epoch_num)
    trainer.train()
    
    # evaluate
    '''
    val_parm = torch.tensor(val_parm).to(config.device)
    val_targ = torch.tensor(val_targ).to(config.device)
    val_mask = torch.tensor(val_mask).to(config.device)
    val_pred = fno(val_parm)
    val_error = ((val_mask*(val_pred-val_targ)**2).sum() / (val_mask*val_targ**2).sum()) **0.5
    print(val_error)
    '''
if __name__ == "__main__":
    main()
