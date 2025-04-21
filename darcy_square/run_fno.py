import torch
import numpy as np

import xfno

def main():
    # configuration
    config = xfno.utils.load_config('./config/fno.yaml')

    # set random seed
    xfno.set_random_seed()

    # dataset
    dim = 2
    bounds = np.array(config.geo.bounds).reshape(dim,2)
    geo = xfno.Geometry(bounds)
    mesh = xfno.Mesh(geo, config.mesh.nx)
    dataset = xfno.Dataset(geo, mesh, config.data.train_size+config.data.valid_size,
                           config.data.load_cache)

    train_idx = range(config.data.train_size)
    train_dataset = xfno.DatasetFNO(dataset, train_idx, device=config.device)

    valid_idx = range(config.data.train_size,config.data.train_size+config.data.valid_size)
    valid_dataset = xfno.DatasetFNO(dataset, valid_idx, device=config.device)

    # dataloader
    train_dataloader = xfno.DataLoader(train_dataset, config.data.batch_size)
    valid_dataloader = xfno.DataLoader(valid_dataset, config.data.batch_size, shuffle=False)

    # normalizer
    input_normalizer = xfno.Normalizer(train_dataset.param)
    output_normalizer = xfno.Normalizer(train_dataset.label)

    # model
    model = xfno.model.FNO2d(config.model.input_channel, config.model.output_channel,
        config.model.width, config.model.mode1_num, config.model.mode2_num,
        config.model.padding, config.model.layer_num,
        input_normalizer=input_normalizer, output_normalizer=output_normalizer)
    model.to(config.device)
    print(xfno.count_params(model))

    # loss
    loss = xfno.LossFNO(model)
    error = xfno.RelativeL2Error(model)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        step_size=config.train.step_size, gamma=config.train.gamma)

    trainer = xfno.Trainer(train_dataloader, valid_dataloader, model,
                           loss, error, optimizer, scheduler,
                           ckpt_name=config.ckpt.name, ckpt_dirt=config.ckpt.dirt)
    trainer.train(config.train.epoch_num)

if __name__ == "__main__":
    main()