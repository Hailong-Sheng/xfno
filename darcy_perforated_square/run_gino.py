import torch
import numpy as np

import xfno

def main():
    config = xfno.utils.load_config('./config/gino.yaml')
    
    # geometry
    dim = 2
    bounds = np.array(config.geo.bounds).reshape(dim,2)
    center = np.array(config.geo.center)
    geo = xfno.geometry.Geometry(bounds, center, config.geo.radius)
    
    # mesh
    nx = np.array(config.mesh.nx, np.int32)
    mesh_car = xfno.mesh.MeshCartesian(geo, bounds, nx)
    mesh_non = xfno.mesh.MeshNonCartesian(geo, bounds, nx)
    
    # dataset
    param_size = config.data.train_size + config.data.valid_size
    dataset = xfno.Dataset(geo, mesh_car, param_size, config.data.load_cache)
    
    train_idx = range(config.data.train_size)
    train_dataset = xfno.dataset.DatasetGINO(dataset, train_idx, mesh_non,
                                               device=config.device)

    valid_idx = range(config.data.train_size, param_size)
    valid_dataset = xfno.dataset.DatasetGINO(dataset, valid_idx, mesh_non, 
                                               device=config.device)
    
    # dataloader
    train_dataloader = xfno.dataset.PyGDataLoader(train_dataset, config.data.batch_size)
    valid_dataloader = xfno.dataset.PyGDataLoader(valid_dataset, config.data.batch_size)
    
    # normalizer
    input_normalizer = xfno.Normalizer(train_dataset.param)
    output_normalizer = xfno.Normalizer(train_dataset.label)

    # encoder
    encoder = xfno.GNO(config.model.node_input_dim, config.model.edge_input_dim,
                       config.model.node_output_dim,
                       config.model.width, config.model.edge_hidden_dim,
                       config.model.gno_layer_num).to(config.device)
    
    # process
    fno = xfno.FNO2d(config.model.input_channel, config.model.output_channel,
        config.model.width, config.model.mode1_num, config.model.mode2_num,
        config.model.padding, config.model.fno_layer_num)
    
    # decoder
    decoder = xfno.GNO(config.model.node_input_dim, config.model.edge_input_dim,
                       config.model.node_output_dim,
                       config.model.width, config.model.edge_hidden_dim,
                       config.model.gno_layer_num).to(config.device)
    
    # model
    model = xfno.GINO(encoder, fno, decoder, train_dataset.mesh_non.cen_size, 
                      train_dataset.mesh_non.cen_size, train_dataset.mesh_non.nx,
                      input_normalizer=input_normalizer, output_normalizer=output_normalizer)
    
    # loss
    loss = xfno.LossGINO(model)
    error = xfno.RelativeL2ErrorGINO(model)

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        step_size=config.train.step_size, gamma=config.train.gamma)
    
    trainer = xfno.Trainer(train_dataloader, valid_dataloader, 
                           model, loss, error, optimizer, scheduler,
                           ckpt_name=config.ckpt.name, ckpt_dirt=config.ckpt.dirt,
                           result_dirt=config.output.dirt, device=config.device)
    trainer.train(config.train.epoch_num)

if __name__ == '__main__':
    main()