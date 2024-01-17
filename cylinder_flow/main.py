import os
import argparse
import torch
import numpy as np
import time

import geometry
import mesh
import data
import networks

def loss_func(Net, Conv, TrSet):
    """ loss function """
    TrSet.u = Net(TrSet.parm)
    TrSet.u0 = TrSet.u[:,0:1,...]
    TrSet.u1 = TrSet.u[:,1:2,...]
    TrSet.p = TrSet.u[:,2:3,...]
    
    TrSet.u0 = TrSet.u0 * TrSet.mask
    TrSet.u1 = TrSet.u1 * TrSet.mask
    TrSet.p = TrSet.p * TrSet.mask
    
    TrSet.u0_nb = Conv(TrSet.u0) + TrSet.v0
    TrSet.u1_nb = Conv(TrSet.u1) + TrSet.v1
    TrSet.p_nb = Conv(TrSet.p) + TrSet.v2
    
    TrSet.res0 = (TrSet.wei0_u0 * TrSet.u0_nb + TrSet.wei0_u1 * TrSet.u1_nb + 
                  TrSet.wei0_p * TrSet.p_nb).sum(1,keepdims=True) - TrSet.r0
    TrSet.res1 = (TrSet.wei1_u0 * TrSet.u0_nb + TrSet.wei1_u1 * TrSet.u1_nb + 
                  TrSet.wei1_p * TrSet.p_nb).sum(1,keepdims=True) - TrSet.r1
    
    c0 = TrSet.wei0_u0[:,4:5,:,:] * TrSet.mask + (1-TrSet.mask)
    c1 = TrSet.wei1_u1[:,4:5,:,:] * TrSet.mask + (1-TrSet.mask)
    TrSet.wei0_u0_nb = -TrSet.wei0_u0 / c0
    TrSet.wei0_u1_nb = -TrSet.wei0_u1 / c0
    TrSet.wei0_p_nb  = -TrSet.wei0_p  / c0
    TrSet.r0_nb      =  TrSet.r0      / c0
    TrSet.wei1_u0_nb = -TrSet.wei1_u0 / c1
    TrSet.wei1_u1_nb = -TrSet.wei1_u1 / c1
    TrSet.wei1_p_nb = -TrSet.wei1_p   / c1
    TrSet.r1_nb      =  TrSet.r1      / c1
    TrSet.wei0_u0_nb[:,4,:,:] = 0
    TrSet.wei1_u1_nb[:,4,:,:] = 0
    
    TrSet.u0_intp = (TrSet.wei0_u0_nb * TrSet.u0_nb + TrSet.wei0_u1_nb * TrSet.u1_nb + 
                     TrSet.wei0_p_nb * TrSet.p_nb).sum(1,keepdims=True) + TrSet.r0_nb
    TrSet.u1_intp = (TrSet.wei1_u0_nb * TrSet.u0_nb + TrSet.wei1_u1_nb * TrSet.u1_nb + 
                     TrSet.wei1_p_nb * TrSet.p_nb).sum(1,keepdims=True) + TrSet.r1_nb

    TrSet.res2 = (TrSet.wei2_u0 * (Conv(TrSet.u0_intp) + TrSet.v0) + 
                  TrSet.wei2_u1 * (Conv(TrSet.u1_intp) + TrSet.v1)
                  ).sum(1,keepdims=True) - TrSet.r2
    
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
    os.makedirs('./optimal_state', exist_ok=True)
    torch.save(Net.state_dict(), './optimal_state/optimal_state.tar')
    
    """ Evaluate loss and error """
    loss_history = torch.zeros(epochs_o+1)
    loss_history[0] = loss_func(Net, Conv, TrSet).data
    
    error_u0_history = torch.zeros(epochs_o+1)
    error_u1_history = torch.zeros(epochs_o+1)
    error_p_history = torch.zeros(epochs_o+1)
    
    TeSet.u = Net(TeSet.parm)
    TeSet.u0 = TeSet.u[:,0:1,...]
    TeSet.u1 = TeSet.u[:,1:2,...]
    TeSet.p = TeSet.u[:,2:3,...]
    
    error_u0_history[0] = relative_error(TeSet.u0*TeSet.mask, TeSet.u0a*TeSet.mask).data
    error_u1_history[0] = relative_error(TeSet.u1*TeSet.mask, TeSet.u1a*TeSet.mask).data
    error_p_history[0] = relative_error(TeSet.p*TeSet.mask, TeSet.pa*TeSet.mask).data
    print('epoch: %d, loss: %.3e, error_u0: %.3e, error_u1: %.3e, error_p: %.3e'
          %(0, loss_history[0], error_u0_history[0], error_u1_history[0], error_p_history[0]))
    
    """ Training cycle """
    for it_o in range(epochs_o):
        
        start_time = time.time()
        
        """ Train neural network """
        loss_history[it_o+1] = train(Net, Conv, TrSet, Optim, optim_type, epochs_i)
        
        """ Evaluate error """
        TeSet.u = Net(TeSet.parm)
        TeSet.u0 = TeSet.u[:,0:1,...]
        TeSet.u1 = TeSet.u[:,1:2,...]
        TeSet.p = TeSet.u[:,2:3,...]
        
        error_u0_history[it_o+1] = relative_error(TeSet.u0*TeSet.mask, TeSet.u0a*TeSet.mask).data
        error_u1_history[it_o+1] = relative_error(TeSet.u1*TeSet.mask, TeSet.u1a*TeSet.mask).data
        error_p_history[it_o+1] = relative_error(TeSet.p*TeSet.mask, TeSet.pa*TeSet.mask).data
        
        """ Print """
        elapsed = time.time() - start_time
        print('epoch: %d, loss: %.3e, error_u0: %.3e, error_u1: %.3e, error_p: %.3e, time: %.2f'
              %((it_o+1)*epochs_i, loss_history[it_o+1], error_u0_history[it_o+1],
                error_u1_history[it_o+1], error_p_history[it_o+1], elapsed))

        """ decay learning rate """
        if optim_type=='adam':
            Sched.step()
            print(Optim.state_dict()['param_groups'][0]['lr'])
    
    os.makedirs('./results', exist_ok=True)
    np.savetxt('./results/loss_history.txt', loss_history)
    np.savetxt('./results/error_u0_history.txt', error_u0_history)
    np.savetxt('./results/error_u1_history.txt', error_u1_history)
    np.savetxt('./results/error_p_history.txt', error_p_history)

def main():
    """ Configurations """
    parser = argparse.ArgumentParser(description='Neural Network Method')
    parser.add_argument('--re', type=float, default=100,
                        help='renold number')
    parser.add_argument('--bounds', type=float, default=[-1.50,1.50, -0.50,0.50],
                        help='lower and upper bounds of the domain')
    parser.add_argument('--nx', type=int, default=[60,20],
                        help='size of the mesh')
    parser.add_argument('--fno_modes1', type=int, default=12,
                        help='fno mode1')
    parser.add_argument('--fno_modes2', type=int, default=12,
                        help='fno mode2')
    parser.add_argument('--fno_width', type=int, default=32,
                        help='fno width')
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

    if args.dtype=='float16':
        dtype = torch.float16
    if args.dtype=='float32':
        dtype = torch.float32
    if args.dtype=='float64':
        dtype = torch.float64
    torch.set_default_dtype(dtype)

    parm_size = 20
    dim = 2
    bounds = torch.tensor(args.bounds).reshape(dim,2)
    nx = torch.tensor(args.nx).int()

    center = [[-0.50,-0.15],[-0.25,-0.15],[0.00,-0.15],[0.25,-0.15],[0.50,-0.15],
              [-0.50,-0.05],[-0.25,-0.05],[0.00,-0.05],[0.25,-0.05],[0.50,-0.05],
              [-0.50, 0.05],[-0.25, 0.05],[0.00, 0.05],[0.25, 0.05],[0.50, 0.05],
              [-0.50, 0.15],[-0.25, 0.15],[0.00, 0.15],[0.25, 0.15],[0.50, 0.15]]
    center = torch.tensor(center).reshape(parm_size,dim)
    radius = 0.2
    
    """ geometry and mesh """ 
    geo = geometry.Geometry(bounds, center, radius)
    msh = mesh.Mesh(geo, bounds, nx)

    """ dataset """
    tr_set = data.TrSet(geo, msh, args.re, dtype)
    tr_set.to(device)
    
    # center = [[-0.4,-0.1],[0.0,0.0],[0.4,0.1]]
    te_set = data.TeSet('solution.csv', 3, nx, dtype)
    te_set.to(device)
    
    tr_set.u0 = te_set.u0a
    tr_set.u1 = te_set.u1a
    tr_set.p = te_set.pa
    
    """ network """
    net = networks.FNO2d(args.fno_modes1, args.fno_modes2, args.fno_width).to(device)
    conv = networks.FixedConv2D().to(device)
    
    """ optimizer and scheduler """
    if args.optim_type=='adam':
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.optim_type=='lbfgs':
        optim = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=args.epochs_i,
                                  tolerance_grad=1e-16, tolerance_change=1e-16,
                                  line_search_fn='strong_wolfe')
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.98)
    
    """ Train neural network """
    start_time = time.time()
    solve(net, conv, tr_set, te_set, optim, sched, args.optim_type, args.epochs_o, args.epochs_i)
    elapsed = time.time() - start_time
    print('train time: %.2f' %(elapsed))
    
    """ Save results """
    net.load_state_dict(torch.load('optimal_state/optimal_state.tar'))
    te_set.u = net(te_set.parm)
    te_set.x0 = te_set.x0.reshape(-1,1)
    te_set.x1 = te_set.x1.view(-1,1)
    te_set.parm = te_set.parm.view(-1,1)
    te_set.u0 = te_set.u[:,0:1,...].reshape(te_set.parm_size*te_set.nx[0]*te_set.nx[1],1)
    te_set.u1 = te_set.u[:,1:2,...].reshape(te_set.parm_size*te_set.nx[0]*te_set.nx[1],1)
    te_set.p = te_set.u[:,2:3,...].reshape(te_set.parm_size*te_set.nx[0]*te_set.nx[1],1)
    te_set.mask = te_set.mask.view(-1,1)
    solution = torch.cat([te_set.x0,te_set.x1,te_set.parm,
                          te_set.u0,te_set.u1,te_set.p,te_set.mask],1)
    np.savetxt('./results/solution.txt',solution.detach().cpu())

if __name__=='__main__':
    main()
