import os
import torch
import numpy as np
import time
from datetime import datetime

class Trainer():
    def __init__(self, train_dataloader, valid_dataset, model, loss, error,
                 optimizer, scheduler, epoch_num: int=1000, print_freq: int=10,
                 ckpt_freq: int=10, ckpt_name: str='checkpoint',
                 ckpt_dirt: str='checkpoint', result_dirt: str='./result',
                 device: str='cuda'):
        self.train_dataloader = train_dataloader
        self.valid_dataset = valid_dataset
        self.model = model
        self.loss = loss
        self.error = error
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch_num = epoch_num
        self.print_freq = print_freq
        self.ckpt_freq = ckpt_freq
        self.ckpt_name = ckpt_name
        self.ckpt_dirt = ckpt_dirt
        self.result_dirt = result_dirt
        self.device = device

        self.model = self.model.to(self.device)

    def train(self):
        error_history = []
        st = time.time()

        for epoch in range(self.epoch_num):
            for data in self.train_dataloader:
                
                # forward and backward propagation
                self.optimizer.zero_grad()
                loss = self.loss(data)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            
            # save checkpoint
            if epoch % self.ckpt_freq == 0:
                os.makedirs(self.ckpt_dirt, exist_ok=True)
                torch.save(self.model.state_dict(), f'{self.ckpt_dirt}/{self.ckpt_name}.pth')
            
            # print
            if epoch % self.print_freq == 0:
                error = self.error(self.valid_dataset)
                error_history.append(torch.unsqueeze(error,dim=0).cpu().detach())

                info = (f'epoch: {epoch:10d} | loss: {loss.cpu().detach():10.3e} | ' + 
                        f'error: {error.cpu().detach():10.3e}')
                if epoch >= 0:
                    info += f', time: {time.time()-st:10.3e} s'
                print(info)

                st = time.time()
            
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