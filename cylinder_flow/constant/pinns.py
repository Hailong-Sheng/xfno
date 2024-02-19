import os
import argparse
import torch
import numpy as np
import time
import pandas as pd

import network

class InSet():
    """ data structure of training set on the interior of computational domain
    Args:
        bounds: spatial bounds of computational domain
        nx: number of data points in spatial dimension
        re: renold number
        device: device storing data
    """
    def __init__(self, bounds, center, radius, nx, re, device):
        self.bounds = bounds
        self.center = center
        self.radius = radius
        self.nx = nx
        self.re = re
        self.device = device

        self.nu = 1/self.re

        self.dim = self.bounds.shape[0]

        self.sample()

    def sample(self):
        self.size = self.nx[0]*self.nx[1]
        self.x = self.bounds[:,0] + torch.rand(self.size,self.dim) * (
            self.bounds[:,1]-self.bounds[:,0])
        idx = (((self.x-self.center)**2).sum(-1))**0.5 > self.radius
        self.x = self.x[idx,:]

        self.size = self.x.shape[0]
        self.x.requires_grad = True
        self.grad_outputs = torch.ones(self.size,1)
        
        self.r0 = torch.zeros(self.size,1)
        self.r1 = torch.zeros(self.size,1)
        self.r2 = torch.zeros(self.size,1)
        
        self.x = self.x.to(self.device)
        self.grad_outputs = self.grad_outputs.to(self.device)
        self.r0 = self.r0.to(self.device)
        self.r1 = self.r1.to(self.device)
        self.r2 = self.r2.to(self.device)

class BdSet():
    """ data struture of training set on the domain boundary
    Args:
        bounds: spatial bounds of computational domain
        nx: number of data points in spatial dimension
        device: device storing data    
    """
    def __init__(self, bounds, center, radius, nx, device):
        self.bounds = bounds
        self.center = center
        self.radius = radius
        self.nx = nx
        self.device = device

        self.dim = self.bounds.shape[0]
        
        """ volecity """
        self.x_u = torch.zeros(0,self.dim)
        size = 2*self.nx[0] + self.nx[1]
        x = torch.zeros(size,self.dim)
        hx = (self.bounds[:,1]-self.bounds[:,0])/self.nx
        m = 0
        for j in range(self.nx[1]):
            x[m,0] = self.bounds[0,0]
            x[m,1] = self.bounds[1,0] + (j+0.5)*hx[1]
            m += 1
        for i in range(self.nx[0]):
            x[m,0] = self.bounds[0,0] + (i+0.5)*hx[0]
            x[m,1] = self.bounds[1,0]
            m += 1
        for i in range(self.nx[0]):
            x[m,0] = self.bounds[0,0] + (i+0.5)*hx[0]
            x[m,1] = self.bounds[1,1]
            m += 1
        self.x_u = torch.cat((self.x_u, x))

        nth = int(2*np.pi*self.radius/hx[0])
        hth = 2*np.pi/nth
        x = torch.zeros(nth,self.dim)
        for m in range(nth):
            x[m,0] = self.center[0] + self.radius*np.cos(m*hth)
            x[m,1] = self.center[1] + self.radius*np.sin(m*hth)
        self.x_u = torch.cat((self.x_u, x))

        self.size_u = self.x_u.shape[0]
        self.r0 = torch.zeros(self.size_u,1)
        self.r1 = torch.zeros(self.size_u,1)
        
        idx = (self.x_u[:,0]-self.bounds[0,0]).abs()<1e-4
        self.r0[idx,:] = 1.0
        
        """ pressure """
        self.size_p = self.nx[1]
        self.x_p = torch.zeros(self.size_p,self.dim)
        hx = (self.bounds[:,1]-self.bounds[:,0])/self.nx
        m = 0
        for j in range(self.nx[1]):
            self.x_p[m,0] = self.bounds[0,1]
            self.x_p[m,1] = self.bounds[1,0] + (j+0.5)*hx[1]
            m += 1
        self.r2 = torch.zeros(self.size_p,1)
        
        self.x_u = self.x_u.to(self.device)
        self.x_p = self.x_p.to(self.device)
        self.r0 = self.r0.to(self.device)
        self.r1 = self.r1.to(self.device)
        self.r2 = self.r2.to(self.device)

class IntSet():
    """ data struture of training set on the domain boundary
    Args:
        bounds: spatial bounds of computational domain
        nx: number of data points in spatial dimension
        device: device storing data
    """
    def __init__(self, bounds, center, radius, nx, device):
        self.bounds = bounds
        self.center = center
        self.radius = radius
        self.nx = nx
        self.device = device

        self.dim = self.bounds.shape[0]
        
        self.int_num = 2
        self.int_x = torch.tensor([0.5,1.0])
        self.size = self.int_num*self.nx[1]
        self.hx = (self.bounds[:,1]-self.bounds[:,0])/self.nx
        self.x = torch.zeros(self.int_num,self.nx[1],self.dim)
        m = 0
        for i in range(self.int_num):
            for j in range(self.nx[1]):
                self.x[i,j,0] = self.int_x[i]
                self.x[i,j,1] = self.bounds[1,0] + (j+0.5)*self.hx[1]
                m += 1
        
        self.loc = (((self.x-self.center)**2).sum())**0.5 > self.radius
        self.r0 = torch.ones(self.int_num)
        self.r1 = torch.zeros(self.int_num)

        self.x = self.x.to(self.device)
        self.loc = self.loc.to(self.device)
        self.r0 = self.r0.to(self.device)
        self.r1 = self.r1.to(self.device)

class TeSet():
    def __init__(self, file_name, dtype, device):
        self.dtype = dtype
        self.device = device
        
        data = pd.read_csv(file_name, header=None)
        data = np.array(data)
        data = torch.tensor(data, dtype=self.dtype)
        
        self.x = data[:,0:2]
        self.u0a = data[:,4:5]
        self.u1a = data[:,5:6]
        self.pa = data[:,6:7]
        self.mask = data[:,7:8]

        self.x = self.x.to(self.device)
        self.u0a = self.u0a.to(self.device)
        self.u1a = self.u1a.to(self.device)
        self.pa = self.pa.to(self.device)
        self.mask = self.mask.to(self.device)

# ----------------------------------------------------------------------------------------------------
class Net(torch.nn.Module):
    """ ResNet
    Args:
        layers: number of units in layers
        bounds: spatial bounds

    functions
        forward: forward propagation
    """
    def __init__(self, layers, bounds):
        super(Net, self).__init__()
        self.layers = layers
        self.layers_hid_num = len(layers)-2
        self.bounds = bounds

        fc = []
        for i in range(self.layers_hid_num):
            fc.append(torch.nn.Linear(self.layers[i],self.layers[i+1]))
            fc.append(torch.nn.Linear(self.layers[i+1],self.layers[i+1]))
        fc.append(torch.nn.Linear(self.layers[-2],self.layers[-1]))
        self.fc = torch.nn.Sequential(*fc)

    def forward(self, x):        
        self.bounds = self.bounds.to(x.device)
        h = (x-self.bounds[:,0])/(self.bounds[:,1]-self.bounds[:,0])
        x = 2*h - 1.0
        for i in range(self.layers_hid_num):
            h = torch.sin(self.fc[2*i](x))
            h = torch.sin(self.fc[2*i+1](h))
            tmp = torch.zeros(x.shape[0],self.layers[i+1]-self.layers[i]).to(x.device)
            x = h + torch.cat((x,tmp),1)
        return self.fc[-1](x)

# ----------------------------------------------------------------------------------------------------
def loss_func(Net, InSet, BdSet, IntSet, beta):
    """ loss function for training network

    parameters
    Net: network
    InSet: training set on the interior of domain
    BdSet: training set on the boundary of domain
    beta: penalty coefficient

    returns
    loss: value of loss function
    """
    InSet.u = Net(InSet.x)
    InSet.u0 = InSet.u[:,0:1]
    InSet.u1 = InSet.u[:,1:2]
    InSet.p = InSet.u[:,2:3]
    
    InSet.u0x, = torch.autograd.grad(InSet.u0, InSet.x,
                                     create_graph=True, retain_graph=True,
                                     grad_outputs=InSet.grad_outputs)
    InSet.u0x0 = InSet.u0x[:,0:1]
    InSet.u0x1 = InSet.u0x[:,1:2]
    InSet.u0x0x, = torch.autograd.grad(InSet.u0x0, InSet.x,
                                       create_graph=True, retain_graph=True,
                                       grad_outputs=InSet.grad_outputs)
    InSet.u0x0x0 = InSet.u0x0x[:,0:1]
    InSet.u0x1x, = torch.autograd.grad(InSet.u0x1, InSet.x,
                                       create_graph=True, retain_graph=True,
                                       grad_outputs=InSet.grad_outputs)
    InSet.u0x1x1 = InSet.u0x1x[:,1:2]
    
    InSet.u1x, = torch.autograd.grad(InSet.u1, InSet.x,
                                     create_graph=True, retain_graph=True,
                                     grad_outputs=InSet.grad_outputs)
    InSet.u1x0 = InSet.u1x[:,0:1]
    InSet.u1x1 = InSet.u1x[:,1:2]
    InSet.u1x0x, = torch.autograd.grad(InSet.u1x0, InSet.x,
                                       create_graph=True, retain_graph=True,
                                       grad_outputs=InSet.grad_outputs)
    InSet.u1x0x0 = InSet.u1x0x[:,0:1]
    InSet.u1x1x, = torch.autograd.grad(InSet.u1x1, InSet.x,
                                       create_graph=True, retain_graph=True,
                                       grad_outputs=InSet.grad_outputs)
    InSet.u1x1x1 = InSet.u1x1x[:,1:2]
    
    InSet.px, = torch.autograd.grad(InSet.p, InSet.x,
                                    create_graph=True, retain_graph=True,
                                    grad_outputs=InSet.grad_outputs)
    InSet.px0 = InSet.px[:,0:1]
    InSet.px1 = InSet.px[:,1:2]
    
    InSet.res0 = -InSet.nu * (InSet.u0x0x0 + InSet.u0x1x1) + \
                 InSet.u0*InSet.u0x0 + InSet.u1*InSet.u0x1 + \
                 InSet.px0
    InSet.res1 = -InSet.nu * (InSet.u1x0x0 + InSet.u1x1x1) + \
                 InSet.u0*InSet.u1x0 + InSet.u1*InSet.u1x1 + \
                 InSet.px1
    InSet.res2 = InSet.u0x0 + InSet.u1x1
    
    BdSet.u = Net(BdSet.x_u)
    BdSet.u0 = BdSet.u[:,0:1]
    BdSet.u1 = BdSet.u[:,1:2]
    BdSet.res0 = BdSet.u0 - BdSet.r0
    BdSet.res1 = BdSet.u1 - BdSet.r1

    BdSet.p = Net(BdSet.x_p)[:,2:3]
    BdSet.res2 = BdSet.p - BdSet.r2

    tmp_x = IntSet.x.reshape(IntSet.size,IntSet.dim)
    tmp_u = Net(tmp_x)
    IntSet.u0 = tmp_u[:,0:1].reshape(IntSet.int_num,IntSet.nx[1])
    IntSet.u1 = tmp_u[:,1:2].reshape(IntSet.int_num,IntSet.nx[1])
    
    IntSet.res0 = (IntSet.loc*IntSet.u0).mean(1) - IntSet.r0
    IntSet.res1 = (IntSet.loc*IntSet.u1).mean(1) - IntSet.r1

    loss = (InSet.res0**2).mean() + beta*(BdSet.res0**2).mean() + \
           (InSet.res1**2).mean() + beta*(BdSet.res1**2).mean() + \
           (InSet.res2**2).mean() + beta*(BdSet.res2**2).mean() + \
           beta*(IntSet.res0**2).mean() + beta*(IntSet.res1**2).mean()
    return loss**0.5

def relative_error(u, ua):
    """ Evaluate l-2 relative error
    
    parameters
    u: approximate solution
    ua: true solution
    
    returns
    l-2 relative error
    """
    return (((u-ua)**2).sum() / ((ua**2).sum()+1e-16)) ** 0.5

# ----------------------------------------------------------------------------------------------------
def train(Net, InSet, BdSet, IntSet, beta, Optim, optim_type, epochs_i):
    """ Train neural network

    parameters
    Net: network
    InSet: training set on the interior of domain
    BdSet: training set on the boundary of domain
    Optim: optimizer for training network
    optim_type: type of optimizer
    epochs_i: number of inner iterations

    returns
    loss: value of loss function
    """

    """ Record the optimal loss """
    loss0 = loss_func(Net, InSet, BdSet, IntSet, beta)
    
    """ Forward and backward propagation """
    if optim_type=='adam':
        for it_i in range(epochs_i):
            Optim.zero_grad()
            loss = loss_func(Net, InSet, BdSet, IntSet, beta)
            loss.backward()
            Optim.step()
    
    if optim_type=='lbfgs':
        def closure():
            Optim.zero_grad()
            loss = loss_func(Net, InSet, BdSet, IntSet, beta)
            loss.backward()
            return loss
        Optim.step(closure)
        
    """ Record the optimal parameters """
    loss = loss_func(Net, InSet, BdSet, IntSet, beta)
    if loss < loss0:
        torch.save(Net.state_dict(), f'./checkpoint/checkpoint_pinns_{beta}.pth')
    return loss.data

def solve(Net, InSet, BdSet, IntSet, TeSet, beta,
          Optim, Sched, optim_type, epochs_o, epochs_i):
    """ Train neural network

    parameters
    Net: network
    InSet: training set on the interior of domain
    BdSet: training set on the boundary of domain
    TeSet: test set
    Optim: optimizer for training network
    optim_type: type of optimizer
    epochs_o: number of outer iterations
    epochs_i: number of inner iterations
    """
    print('Train Neural Network')
    
    """ Record the optimal parameters """
    os.makedirs('./checkpoint', exist_ok=True)
    torch.save(Net.state_dict(), f'./checkpoint/checkpoint_pinns_{beta}.pth')
    
    """ Evaluate error """
    loss_history = torch.zeros(epochs_o+1).to(InSet.device)
    loss_history[0] = loss_func(Net, InSet, BdSet, IntSet, beta).data
    
    error_u0_history = torch.zeros(epochs_o+1).to(InSet.device)
    error_u1_history = torch.zeros(epochs_o+1).to(InSet.device)
    error_p_history = torch.zeros(epochs_o+1).to(InSet.device)
    TeSet.u = Net(TeSet.x)
    TeSet.u0 = TeSet.u[:,0:1]
    TeSet.u1 = TeSet.u[:,1:2]
    TeSet.p = TeSet.u[:,2:3]
    error_u0_history[0] = relative_error(TeSet.u0*TeSet.mask, TeSet.u0a*TeSet.mask).data
    error_u1_history[0] = relative_error(TeSet.u1*TeSet.mask, TeSet.u1a*TeSet.mask).data
    error_p_history[0] = relative_error(TeSet.p*TeSet.mask, TeSet.pa*TeSet.mask).data
    print('epoch: %d, loss: %.3e, error_u0: %.3e, error_u1: %.3e, error_p: %.3e'
          %(0, loss_history[0], error_u0_history[0], error_u1_history[0], error_p_history[0]))
    
    for it_o in range(epochs_o):
        start_time = time.time()
        
        InSet.sample()
        
        if optim_type=='lbfgs':
            Optim = torch.optim.LBFGS(Net.parameters(), lr=1, max_iter=epochs_i,
                                      tolerance_grad=1e-16, tolerance_change=1e-16,
                                      line_search_fn='strong_wolfe')
        

        """ Train neural network """
        loss_history[it_o+1] = train(Net, InSet, BdSet, IntSet, beta, Optim, optim_type, epochs_i)

        """ Evaluate error """
        TeSet.u = Net(TeSet.x)
        TeSet.u0 = TeSet.u[:,0:1]
        TeSet.u1 = TeSet.u[:,1:2]
        TeSet.p = TeSet.u[:,2:3]
        error_u0_history[it_o+1] = relative_error(TeSet.u0*TeSet.mask, TeSet.u0a*TeSet.mask).data
        error_u1_history[it_o+1] = relative_error(TeSet.u1*TeSet.mask, TeSet.u1a*TeSet.mask).data
        error_p_history[it_o+1] = relative_error(TeSet.p*TeSet.mask, TeSet.pa*TeSet.mask).data
        
        elapsed = time.time() - start_time
        print('epoch: %d, loss: %.3e, error_u0: %.3e, error_u1: %.3e, error_p: %.3e, time: %.2f'
              %((it_o+1)*epochs_i, loss_history[it_o+1], error_u0_history[it_o+1],
                error_u1_history[it_o+1], error_p_history[it_o+1], elapsed))
        
        Sched.step()
    
    os.makedirs('./result', exist_ok=True)
    np.savetxt(f'./result/loss_history_pinns_{beta}.txt', loss_history.cpu())
    np.savetxt(f'./result/error_u0_history_pinns_{beta}.txt', error_u0_history.cpu())
    np.savetxt(f'./result/error_u1_history_pinns_{beta}.txt', error_u1_history.cpu())
    np.savetxt(f'./result/error_p_history_pinns_{beta}.txt', error_p_history.cpu())

# ----------------------------------------------------------------------------------------------------
def main():
    
    """ Configurations """
    parser = argparse.ArgumentParser(description='Physics-informed Neural Network Method')
    parser.add_argument('--re', type=float, default=100,
                        help='renold number')
    parser.add_argument('--bounds', type=float, default=[-1.50,1.50, -0.50,0.50],
                        help='lower and upper bounds of the domain')
    parser.add_argument('--center', type=float, default=[-0.30,0.00],
                        help='circle center')
    parser.add_argument('--radius', type=float, default=0.20,
                        help='radius of the circle center')
    parser.add_argument('--u0_inlet', type=float, default=1.00,
                        help='inlet velocity')
    parser.add_argument('--nx_in', type=int, default=[90,30],
                        help='size of the interior set')
    parser.add_argument('--nx_bd', type=int, default=[90,30],
                        help='size of the boundary set')
    parser.add_argument('--beta', type=float, default=10,
                        help='penalty coefficient')
    parser.add_argument('--layers', type=int, default=[2,256,256,256,3],
                        help='network structure')
    parser.add_argument('--optim_type', type=str, default='adam',
                        help='opimizer type')
    parser.add_argument('--epochs_o', type=int, default=200,
                        help='number of outer iterations')
    parser.add_argument('--epochs_i', type=int, default=100,
                        help='number of inner iterations')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='device')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='learning rate')
    args = parser.parse_args()
    
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(device)

    if args.dtype=='float32':
        dtype = torch.float32
    if args.dtype=='float64':
        dtype = torch.float64
    torch.set_default_dtype(dtype)

    dim = 2
    bounds = torch.tensor(args.bounds).reshape(dim,2)
    center = torch.tensor(args.center)
    nx_in = torch.tensor(args.nx_in).int()
    nx_bd = torch.tensor(args.nx_bd).int()
    
    """ Generate data set """
    in_set = InSet(bounds, center, args.radius, nx_in, args.re, device)
    bd_set = BdSet(bounds, center, args.radius, nx_bd, device)
    int_set = IntSet(bounds, center, args.radius, nx_in, device)
    te_set = TeSet('te_set.csv', dtype, device)
    
    """ Construct neural network """
    net = Net(args.layers, bounds).to(device)
    if args.optim_type=='adam':
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.optim_type=='lbfgs':
        optim = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=args.epochs_i,
                                  tolerance_grad=1e-16, tolerance_change=1e-16,
                                  line_search_fn='strong_wolfe')
    print(f'amount of parameters: {network.count_params(net)}')
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.95)
    
    """ Train neural network """
    start_time = time.time()
    solve(net, in_set, bd_set, int_set, te_set, args.beta,
          optim, sched, args.optim_type, args.epochs_o, args.epochs_i)
    elapsed = time.time() - start_time
    print('train time: %.2f' %(elapsed))
    
if __name__ == '__main__':
    main()
