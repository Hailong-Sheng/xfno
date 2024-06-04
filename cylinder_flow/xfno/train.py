import os
import torch
import numpy as np
import time

class Trainer():
    """ trainer """
    def __init__(self, train_dataloader, model, loss, error,
                 optimizer, scheduler, epochs_o, epochs_i):
        self.train_dataloader = train_dataloader
        self.model = model
        self.loss = loss
        self.error = error
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs_o = epochs_o
        self.epochs_i = epochs_i

    def train(self):
        # record checkpoint
        os.makedirs('./checkpoint/train', exist_ok=True)
        torch.save(self.model.state_dict(), 'checkpoint/train/checkpoint_xfno.pth')
        
        # evaluate error
        loss_history = torch.zeros(self.epochs_o+1)
        error_u0_history = torch.zeros(self.epochs_o+1)
        error_u1_history = torch.zeros(self.epochs_o+1)
        error_p_history = torch.zeros(self.epochs_o+1)    
        error_u0_history[0], error_u1_history[0], error_p_history[0] = self.error()
        print('epoch: %d, loss: %.3e, error_u0: %.3e, error_u1: %.3e, error_p: %.3e'
            %(0, loss_history[0], error_u0_history[0], error_u1_history[0], error_p_history[0]))
        
        # train loop
        for it_o in range(self.epochs_o):
            
            start_time = time.time()

            # forward and backward propagation
            for it_i in range(self.epochs_i):
                for data in self.train_dataloader:
                    self.optimizer.zero_grad()
                    loss = self.loss(data)
                    loss.backward()
                    self.optimizer.step()
            
            # decay learning rate
            self.scheduler.step()
            
            # record checkpoint
            torch.save(self.model.state_dict(), './checkpoint/train/checkpoint_xfno.pth')
            
            # evaluate loss and error
            loss_history[it_o+1] = loss.detach().cpu()
            error_u0_history[it_o+1], error_u1_history[it_o+1], error_p_history[it_o+1] = self.error()
            
            # print
            elapsed = time.time() - start_time
            print('epoch: %d, loss: %.3e, error_u0: %.3e, error_u1: %.3e, error_p: %.3e, time: %.2f'
                %((it_o+1)*self.epochs_i, loss_history[it_o+1], error_u0_history[it_o+1],
                    error_u1_history[it_o+1], error_p_history[it_o+1], elapsed))
            print(self.optimizer.state_dict()['param_groups'][0]['lr'])
        
        # record loss and error history
        os.makedirs('./result', exist_ok=True)
        np.savetxt('./result/loss_history.txt', loss_history)
        np.savetxt('./result/error_u0_history.txt', error_u0_history)
        np.savetxt('./result/error_u1_history.txt', error_u1_history)
        np.savetxt('./result/error_p_history.txt', error_p_history)