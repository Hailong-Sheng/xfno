import os
import torch
import numpy as np
import time
from datetime import datetime

class Trainer():
    def __init__(self, train_dataloader, valid_dataloader, model, loss, error,
                 optimizer, scheduler, print_freq: int=10,
                 ckpt_freq: int=10, ckpt_name: str='checkpoint',
                 ckpt_dirt: str='checkpoint', result_dirt: str='./result',
                 device: str='cuda'):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.model = model
        self.loss = loss
        self.error = error
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.print_freq = print_freq
        self.ckpt_freq = ckpt_freq
        self.ckpt_name = ckpt_name
        self.ckpt_dirt = ckpt_dirt
        self.result_dirt = result_dirt
        self.device = device

        self.model = self.model.to(self.device)

    def train(self, epoch_num):
        error_history = []
        start_time = time.time()

        for epoch in range(epoch_num):
            self.model.train()

            # forward and backward propagation
            loss_sum, sample_sum = 0., 0
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                loss = self.loss(data)
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.detach().cpu()
                batch_size = data['param'].shape[0] if 'param' in data else len(data.ptr)-1
                sample_sum += batch_size
            
            self.scheduler.step()
            # print(self.optimizer.param_groups[0]['lr'])
            
            # save checkpoint
            if epoch % self.ckpt_freq == 0:
                os.makedirs(self.ckpt_dirt, exist_ok=True)
                torch.save(self.model.state_dict(), f'{self.ckpt_dirt}/{self.ckpt_name}.pth')
            
            # print
            if epoch % self.print_freq == 0:
                loss_mean = loss_sum / sample_sum

                self.model.eval()
                error_sum, sample_sum = 0., 0
                for data in self.valid_dataloader:
                    error = self.error(data)
                    batch_size = data['param'].shape[0] if 'param' in data else len(data.ptr)-1
                    sample_sum += batch_size
                    error_sum += error.detach().cpu() * batch_size
                error_mean = error_sum / sample_sum
                error_history.append(error_mean)
                
                info = (f'epoch: {epoch:6d} | loss: {loss_mean:10.3e} | ' + 
                        f'error: {error_mean:10.3e}')
                if epoch >= 0:
                    info += f', time: {time.time()-start_time:10.3e} s'
                print(info)

                start_time = time.time()
            
        os.makedirs(self.result_dirt, exist_ok=True)
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        np.savetxt(f'{self.result_dirt}/error_history_{self.ckpt_name}_{current_datetime}.txt',
                   np.array(error_history))

class TrainerCT():
    def __init__(self, train_dataloader, model, loss, optimizer, scheduler, 
                 epoch_num: int=1000, print_freq: int=10, device: str='cuda'):
        self.train_dataloader = train_dataloader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch_num = epoch_num
        self.print_freq = print_freq
        self.device = device

        self.model = self.model.to(self.device)

    def train(self):
        st = time.time()
        for epoch in range(self.epoch_num):
            
            # forward and backward propagation
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                loss = self.loss(data)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            
            # print loss stats
            if epoch % self.print_freq == 0:
                info = (f'epoch: {epoch:10d} | loss: {loss.cpu().detach():10.3e}')
                if epoch >= 0:
                    info += f' | time: {time.time()-st:10.3e} s'
                    print(info)
                    st = time.time()