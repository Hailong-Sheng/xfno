import os
import torch
import numpy as np
import time

class Trainer():
    def __init__(self, train_dataloader, valid_dataloader, model, loss, error,
                 optimizer, scheduler, epoch_num: int=1000, print_freq: int=10,
                 device: str='cuda'):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.model = model
        self.loss = loss
        self.error = error
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch_num = epoch_num
        self.print_freq = print_freq
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
            
            # print loss stats
            if epoch % self.print_freq == 0:
                for data in self.valid_dataloader:
                    error = self.error(data)
                    error_history.append(torch.unsqueeze(error,dim=0).cpu().detach())

                    info = (f'epoch: {epoch:10d} | loss: {loss.cpu().detach():10.3e} | ' + 
                            f'error: {error.cpu().detach():10.3e}')
                    if epoch >= 0:
                        info += f', time: {time.time()-st:10.3e} s'
                    print(info)
                    
                    st = time.time()
                    break
            
        os.makedirs('result', exist_ok=True)
        np.savetxt('result/error_history.txt', np.array(error_history))

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