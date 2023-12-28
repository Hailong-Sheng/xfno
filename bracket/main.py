import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

import geometry
import data
import networks

def loss_func(Net, Conv, TrSet):
    """ loss function """
    '''
    TrSet.u = Net(TrSet.p)
    TrSet.u0 = TrSet.u[:,0:1,...]
    TrSet.u1 = TrSet.u[:,1:2,...]
    TrSet.u2 = TrSet.u[:,2:3,...]
    '''
    TrSet.u = Net(TrSet.x.reshape(TrSet.space_size,TrSet.dim))
    TrSet.u0 = TrSet.u[:,0:1].reshape(1,1,TrSet.nx[0],TrSet.nx[1],TrSet.nx[2])
    TrSet.u1 = TrSet.u[:,1:2].reshape(1,1,TrSet.nx[0],TrSet.nx[1],TrSet.nx[2])
    TrSet.u2 = TrSet.u[:,2:3].reshape(1,1,TrSet.nx[0],TrSet.nx[1],TrSet.nx[2])
    
    TrSet.u0 = TrSet.u0 * TrSet.mask
    TrSet.u1 = TrSet.u1 * TrSet.mask
    TrSet.u2 = TrSet.u2 * TrSet.mask
    
    TrSet.u0_ = Conv(TrSet.u0)
    TrSet.u1_ = Conv(TrSet.u1)
    TrSet.u2_ = Conv(TrSet.u2)
    
    TrSet.u0_ = TrSet.u0_ + TrSet.v0
    TrSet.u1_ = TrSet.u1_ + TrSet.v1
    TrSet.u2_ = TrSet.u2_ + TrSet.v2
    
    TrSet.res0 = (TrSet.wei0_u0 * TrSet.u0_ + TrSet.wei0_u1 * TrSet.u1_ + 
                  TrSet.wei0_u2 * TrSet.u2_).sum(1,keepdims=True) - TrSet.r0
    TrSet.res1 = (TrSet.wei1_u0 * TrSet.u0_ + TrSet.wei1_u1 * TrSet.u1_ + 
                  TrSet.wei1_u2 * TrSet.u2_).sum(1,keepdims=True) - TrSet.r1
    TrSet.res2 = (TrSet.wei2_u0 * TrSet.u0_ + TrSet.wei2_u1 * TrSet.u1_ + 
                  TrSet.wei2_u2 * TrSet.u2_).sum(1,keepdims=True) - TrSet.r2
    
    # print((TrSet.res0*mask)[0,0,2,6,4])
    
    loss = ((TrSet.res0*TrSet.mask)**2).sum() + \
           ((TrSet.res1*TrSet.mask)**2).sum() + \
           ((TrSet.res2*TrSet.mask)**2).sum()
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
def train(Net, Conv, TrSet, Optim, optim_type, epochs_i): 
    """ Train neural network

    parameters
    Net: network
    TrSet: training set
    Optim: optimizer for training network
    epochs_i: number of inner iterations

    returns
    loss: value of loss function
    """

    """ Record initial loss """
    loss0 = loss_func(Net, Conv, TrSet).data

    """ Forward and backward propagation """
    if optim_type=='adam':
        for it_i in range(epochs_i):
            Optim.zero_grad()
            loss = loss_func(Net, Conv, TrSet)
            loss.backward()
            Optim.step()
    
    if optim_type=='lbfgs':
        def closure():
            Optim.zero_grad()
            loss = loss_func(Net, Conv, TrSet)
            loss.backward()
            return loss
        Optim.step(closure)

    """ Record the optimal parameters """
    loss = loss_func(Net, Conv, TrSet).data
    if loss < loss0:
        torch.save(Net.state_dict(), './optimal_state/optimal_state.tar')
    
    return loss

def solve(Net, Conv, TrSet, TeSet, Optim, Sched, optim_type, epochs_o, epochs_i):
    """ Train neural network

    parameters
    Net: network
    TrSet: training set
    TeSet: test set
    Optim: optimizer for training network
    epochs_o: number of outer iterations
    epochs_i: number of inner iterations
    """
    print('Train Neural Network')
    
    """ Record the optimal parameters """
    torch.save(Net.state_dict(), './optimal_state/optimal_state.tar')
    
    """ Evaluate loss and error """
    loss_history = torch.zeros(epochs_o+1)
    loss_history[0] = loss_func(Net, Conv, TrSet).data
    
    error_u0_history = torch.zeros(epochs_o+1)
    error_u1_history = torch.zeros(epochs_o+1)
    error_u2_history = torch.zeros(epochs_o+1)
    '''
    TeSet.u = Net(TeSet.p)
    TeSet.u0 = TeSet.u[:,0:1,...]
    TeSet.u1 = TeSet.u[:,1:2,...]
    TeSet.u2 = TeSet.u[:,2:3,...]
    '''
    TrSet.u = Net(TrSet.x.reshape(TrSet.space_size,TrSet.dim))
    TeSet.u0 = TrSet.u[:,0:1].reshape(1,1,TrSet.nx[0],TrSet.nx[1],TrSet.nx[2])
    TeSet.u1 = TrSet.u[:,1:2].reshape(1,1,TrSet.nx[0],TrSet.nx[1],TrSet.nx[2])
    TeSet.u2 = TrSet.u[:,2:3].reshape(1,1,TrSet.nx[0],TrSet.nx[1],TrSet.nx[2])
    error_u0_history[0] = relative_error(TeSet.u0*TeSet.mask, TeSet.u0a*TeSet.mask).data
    error_u1_history[0] = relative_error(TeSet.u1*TeSet.mask, TeSet.u1a*TeSet.mask).data
    error_u2_history[0] = relative_error(TeSet.u2*TeSet.mask, TeSet.u2a*TeSet.mask).data
    print('epoch: %d, loss: %.3e, error_u0: %.3e, error_u1: %.3e, error_u2: %.3e'
          %(0, loss_history[0], error_u0_history[0], error_u1_history[0], error_u2_history[0]))
    
    """ Training cycle """
    for it_o in range(epochs_o):
        
        start_time = time.time()
        
        """ Train neural network """
        loss_history[it_o+1] = train(Net, Conv, TrSet, Optim, optim_type, epochs_i)
        
        """ Evaluate error """
        '''
        TeSet.u = Net(TeSet.p)
        TeSet.u0 = TeSet.u[:,0:1,...]
        TeSet.u1 = TeSet.u[:,1:2,...]
        TeSet.u2 = TeSet.u[:,2:3,...]
        '''
        TrSet.u = Net(TrSet.x.reshape(TrSet.space_size,TrSet.dim))
        TeSet.u0 = TrSet.u[:,0:1].reshape(1,1,TrSet.nx[0],TrSet.nx[1],TrSet.nx[2])
        TeSet.u1 = TrSet.u[:,1:2].reshape(1,1,TrSet.nx[0],TrSet.nx[1],TrSet.nx[2])
        TeSet.u2 = TrSet.u[:,2:3].reshape(1,1,TrSet.nx[0],TrSet.nx[1],TrSet.nx[2])
        error_u0_history[it_o+1] = relative_error(TeSet.u0*TeSet.mask, TeSet.u0a*TeSet.mask).data
        error_u1_history[it_o+1] = relative_error(TeSet.u1*TeSet.mask, TeSet.u1a*TeSet.mask).data
        error_u2_history[it_o+1] = relative_error(TeSet.u2*TeSet.mask, TeSet.u2a*TeSet.mask).data
        
        """ Print """
        elapsed = time.time() - start_time
        print('epoch: %d, loss: %.3e, error_u0: %.3e, error_u1: %.3e, error_u2: %.3e, time: %.2f'
              %((it_o+1)*epochs_i, loss_history[it_o+1], error_u0_history[it_o+1],
                error_u1_history[it_o+1], error_u2_history[it_o+1], elapsed))

        """ decay learning rate """
        if optim_type=='adam':
            Sched.step()
            print(Optim.state_dict()['param_groups'][0]['lr'])
    '''
    np.savetxt('./results/loss_history.txt', loss_history)
    np.savetxt('./results/error_u0_history.txt', error_u0_history)
    np.savetxt('./results/error_u1_history.txt', error_u1_history)
    np.savetxt('./results/error_u2_history.txt', error_u2_history)
    '''
def main():
    """ Configurations """
    parser = argparse.ArgumentParser(description='Neural Network Method')
    parser.add_argument('--lm', type=float, default=1.0,
                        help='paramter lm')
    parser.add_argument('--mu', type=float, default=0.5,
                        help='paramter mu')
    parser.add_argument('--stress', type=float, default=1.0,
                        help='boundary condition')
    parser.add_argument('--bounds', type=float, default=[-1.0,1.0, -1.0,1.0, -1.0,1.0],
                        help='lower and upper bounds of the domain')
    parser.add_argument('--bounds_p1', type=float, default=[-1.0,-0.8, -1.0,1.0, -1.0,1.0],
                        help='lower and upper bounds of the domain')
    parser.add_argument('--bounds_p2', type=float, default=[-1.0,1.0, -1.0,1.0, -0.2,0.2],
                        help='lower and upper bounds of the domain')
    parser.add_argument('--center', type=float, default=[0.0, 0.0, 0.0],
                        help='center of the hole')
    parser.add_argument('--radius', type=float, default=0.5,
                        help='radius of the hole')
    parser.add_argument('--nx', type=int, default=[20,20,20],
                        help='size of the mesh')
    parser.add_argument('--fc_layer', type=int, default=[3,128,128,128,3],
                        help='fc layer')
    parser.add_argument('--fno_mode1', type=int, default=12,
                        help='fno mode1')
    parser.add_argument('--fno_mode2', type=int, default=12,
                        help='fno mode2')
    parser.add_argument('--fno_mode3', type=int, default=12,
                        help='fno mode3')
    parser.add_argument('--fno_width', type=int, default=32,
                        help='fno width')
    parser.add_argument('--optim_type', type=str, default='lbfgs',
                        help='opimizer type')
    parser.add_argument('--epochs_o', type=int, default=5000,
                        help='number of outer iterations')
    parser.add_argument('--epochs_i', type=int, default=100,
                        help='number of inner iterations')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='device')
    parser.add_argument('--dtype', type=str, default='float64',
                        help='learning rate')
    parser.add_argument('--load_intp_coef', type=bool, default=True,
                        help='load interpolation coefficent')
    args = parser.parse_args()

    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    if args.dtype=='float16':
        dtype = torch.float16
    if args.dtype=='float32':
        dtype = torch.float32
    if args.dtype=='float64':
        dtype = torch.float64
    torch.set_default_dtype(dtype)

    """ geometry and mesh """ 
    dim = 3
    center = torch.tensor(args.center)
    bounds = torch.tensor(args.bounds).reshape(dim,2)
    bounds_p1 = torch.tensor(args.bounds_p1).reshape(dim,2)
    bounds_p2 = torch.tensor(args.bounds_p2).reshape(dim,2)
    nx = torch.tensor(args.nx).int()
    
    geo = geometry.Geometry(bounds_p1, bounds_p2, center, args.radius)
    mesh = geometry.Mesh(geo, bounds, nx)
    
    """ dataset """
    tr_set = data.TrSet(geo, mesh, args.lm, args.mu, args.stress, dtype,
                        args.load_intp_coef)
    tr_set.to(device)
    
    te_set = data.TeSet('solution.csv', nx, dtype)
    te_set.to(device)
    te_set.p = tr_set.p
    
    tr_set.u0 = te_set.u0a
    tr_set.u1 = te_set.u1a
    tr_set.u2 = te_set.u2a
    
    """ network """
    # net = networks.FNO3d(args.fno_modes1, args.fno_modes2, args.fno_modes3, args.fno_width).to(device)
    net = networks.FCNN(args.fc_layer, device).to(device)
    conv = networks.FixedConv3D().to(device)

    # net.load_state_dict(torch.load('optimal_state/optimal_state.tar'))
    
    """ optimizer and scheduler """
    if args.optim_type=='adam':
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.optim_type=='lbfgs':
        optim = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=args.epochs_i,
                                  tolerance_grad=1e-16, tolerance_change=1e-16,
                                  line_search_fn='strong_wolfe')
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.98)
    '''
    loss = loss_func(net, conv, tr_set)
    print(loss)
    '''
    """ Train neural network """
    start_time = time.time()
    solve(net, conv, tr_set, te_set, optim, sched, args.optim_type, args.epochs_o, args.epochs_i)
    elapsed = time.time() - start_time
    print('train time: %.2f' %(elapsed))
    
if __name__=='__main__':
    main()
