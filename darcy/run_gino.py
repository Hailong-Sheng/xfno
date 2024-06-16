import torch
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
    mesh_car = xfno.mesh.MeshCartesian(geo, bounds, nx)
    mesh_non = xfno.mesh.MeshNonCartesian(geo, bounds, nx)
    
    # dataset
    train_dataset = xfno.dataset.Dataset('train', geo, mesh_car, config.data.param_size, 
                                         config.data.load_cache, device=config.device)
    train_dataset_gino = xfno.dataset.DatasetGINO('train', train_dataset, 
        geo, mesh_non, config.data.param_size, device=config.device)
    
    valid_dataset = xfno.dataset.Dataset('valid', geo, mesh_car, config.data.param_size, 
                                         config.data.load_cache, device=config.device)
    valid_dataset_gino = xfno.dataset.DatasetGINO('valid', valid_dataset, 
        geo, mesh_non, config.data.param_size, device=config.device)
    
    train_dataloader = xfno.dataset.PyGDataLoader(train_dataset_gino, config.data.batch_size)
    
    # encoder
    encoder = xfno.model.GNO(config.model.node_input_dim, config.model.edge_input_dim,
                             config.model.node_output_dim,
                             config.model.width, config.model.edge_hidden_dim,
                             config.model.gno_layer_num).to(config.device)
    
    # process
    fno = xfno.model.FNO2d(config.model.input_channel, config.model.output_channel,
                           config.model.width, config.model.mode1_num, config.model.mode2_num,
                           config.model.padding, config.model.fno_layer_num)
    
    # decoder
    decoder = xfno.model.GNO(config.model.node_input_dim, config.model.edge_input_dim,
                             config.model.node_output_dim,
                             config.model.width, config.model.edge_hidden_dim,
                             config.model.gno_layer_num).to(config.device)
    
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        step_size=config.train.step_size, gamma=config.train.gamma)

    trainer = xfno.train.Trainer(train_dataloader=train_dataloader, 
                                 valid_dataset=valid_dataset_gino, 
                                 model=model, loss=loss, error=error,
                                 optimizer=optimizer, scheduler=scheduler,
                                 epoch_num=config.train.epoch_num,
                                 ckpt_name=config.ckpt.name, ckpt_dirt=config.ckpt.dirt,
                                 result_dirt=config.output.dirt, device=config.device)
    trainer.train()

if __name__ == '__main__':
    main()