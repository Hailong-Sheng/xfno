import os
import argparse
import torch
import numpy as np
import time
from torch.utils.data import DataLoader

import geometry
import mesh
import dataset
import network

def loss_func(Net, Conv, Data):
    """ loss function """
    ''''''
    Data['u'] = Net(Data['parm'])
    Data['u0'] = Data['u'][:,0:1,...]
    Data['u1'] = Data['u'][:,1:2,...]
    Data['p'] = Data['u'][:,2:3,...]
    
    Data['u0'] = Data['u0'] * Data['mask']
    Data['u1'] = Data['u1'] * Data['mask']
    Data['p'] = Data['p'] * Data['mask']
    
    Data['u0_nb '] = Conv(Data['u0']) + Data['v0']
    Data['u1_nb '] = Conv(Data['u1']) + Data['v1']
    Data['p_nb '] = Conv(Data['p']) + Data['v2']
    
    Data['wei0_u0'] = (Data['wei0_u0_diff'] + 
        Data['wei0_u0_conv_u0']*Data['u0'] + Data['wei0_u0_conv_u1']*Data['u1'])
    Data['wei1_u1'] = (Data['wei1_u1_diff'] + 
        Data['wei1_u1_conv_u0']*Data['u0'] + Data['wei1_u1_conv_u1']*Data['u1'])

    Data['res0'] = (Data['wei0_u0'] * Data['u0_nb '] + 
                    Data['wei0_p'] * Data['p_nb ']).sum(1,keepdims=True) - Data['r0']
    Data['res1'] = (Data['wei1_u1'] * Data['u1_nb '] + 
                    Data['wei1_p'] * Data['p_nb ']).sum(1,keepdims=True) - Data['r1']
    
    c0 = Data['wei0_u0'][:,4:5,:,:] * Data['mask'] + (1-Data['mask'])
    c1 = Data['wei1_u1'][:,4:5,:,:] * Data['mask'] + (1-Data['mask'])
    Data['wei0_u0_nb'] = -Data['wei0_u0'] / c0
    Data['wei0_p_nb']  = -Data['wei0_p']  / c0
    Data['r0_nb']      =  Data['r0']      / c0
    Data['wei1_u1_nb'] = -Data['wei1_u1'] / c1
    Data['wei1_p_nb']  = -Data['wei1_p']  / c1
    Data['r1_nb']      =  Data['r1']      / c1
    Data['wei0_u0_nb'][:,4,:,:] = 0
    Data['wei1_u1_nb'][:,4,:,:] = 0
    
    Data['u0_intp'] = (Data['wei0_u0_nb'] * Data['u0_nb '] + 
                       Data['wei0_p_nb'] * Data['p_nb ']).sum(1,keepdims=True) + Data['r0_nb']
    Data['u1_intp'] = (Data['wei1_u1_nb'] * Data['u1_nb '] + 
                       Data['wei1_p_nb'] * Data['p_nb ']).sum(1,keepdims=True) + Data['r1_nb']

    Data['res2'] = (Data['wei2_u0'] * (Conv(Data['u0_intp']) + Data['v0']) + 
                    Data['wei2_u1'] * (Conv(Data['u1_intp']) + Data['v1'])
                   ).sum(1,keepdims=True) - Data['r2']
    
    loss = ((Data['res0']*Data['mask'])**2).sum() + \
           ((Data['res1']*Data['mask'])**2).sum() + \
           ((Data['res2']*Data['mask'])**2).sum()
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
def train(Net, Conv, TrLoader, Optim, epochs_i): 
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
    for it_i in range(epochs_i):
        for data in TrLoader:
            Optim.zero_grad()
            loss = loss_func(Net, Conv, data)
            loss.backward()
            Optim.step()

    """ Record the optimal parameters """
    torch.save(Net.state_dict(), './checkpoint/train/checkpoint_xfno.pth')
    
    return loss

def solve(Net, Conv, TrLoader, TeSet, Optim, Sched, epochs_o, epochs_i):
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
    os.makedirs('./checkpoint/train', exist_ok=True)
    torch.save(Net.state_dict(), 'checkpoint/train/checkpoint_xfno.pth')
    
    """ Evaluate loss and error """
    loss_history = torch.zeros(epochs_o+1)
    error_u0_history = torch.zeros(epochs_o+1)
    error_u1_history = torch.zeros(epochs_o+1)
    error_p_history = torch.zeros(epochs_o+1)
    
    TeSet.u = Net(TeSet.parm)
    TeSet.u0 = TeSet.u[:,0:1,...]
    TeSet.u1 = TeSet.u[:,1:2,...]
    TeSet.p = TeSet.u[:,2:3,...]
    
    error_u0_history[0] = relative_error(TeSet.u0*TeSet.mask, TeSet.u0a*TeSet.mask).detach()
    error_u1_history[0] = relative_error(TeSet.u1*TeSet.mask, TeSet.u1a*TeSet.mask).detach()
    error_p_history[0] = relative_error(TeSet.p*TeSet.mask, TeSet.pa*TeSet.mask).detach()
    print('epoch: %d, loss: %.3e, error_u0: %.3e, error_u1: %.3e, error_p: %.3e'
          %(0, loss_history[0], error_u0_history[0], error_u1_history[0], error_p_history[0]))
    
    """ Training cycle """
    for it_o in range(epochs_o):
        
        start_time = time.time()
        
        """ Train neural network """
        loss_history[it_o+1] = train(Net, Conv, TrLoader, Optim, epochs_i).detach().cpu()
        
        """ Evaluate error """
        TeSet.u = Net(TeSet.parm)
        TeSet.u0 = TeSet.u[:,0:1,...]
        TeSet.u1 = TeSet.u[:,1:2,...]
        TeSet.p = TeSet.u[:,2:3,...]
        
        error_u0_history[it_o+1] = relative_error(TeSet.u0*TeSet.mask, TeSet.u0a*TeSet.mask).detach().cpu()
        error_u1_history[it_o+1] = relative_error(TeSet.u1*TeSet.mask, TeSet.u1a*TeSet.mask).detach().cpu()
        error_p_history[it_o+1] = relative_error(TeSet.p*TeSet.mask, TeSet.pa*TeSet.mask).detach().cpu()
        
        """ Print """
        elapsed = time.time() - start_time
        print('epoch: %d, loss: %.3e, error_u0: %.3e, error_u1: %.3e, error_p: %.3e, time: %.2f'
              %((it_o+1)*epochs_i, loss_history[it_o+1], error_u0_history[it_o+1],
                error_u1_history[it_o+1], error_p_history[it_o+1], elapsed))

        """ decay learning rate """
        Sched.step()
        print(Optim.state_dict()['param_groups'][0]['lr'])
    
    os.makedirs('./result', exist_ok=True)
    np.savetxt('./result/loss_history.txt', loss_history)
    np.savetxt('./result/error_u0_history.txt', error_u0_history)
    np.savetxt('./result/error_u1_history.txt', error_u1_history)
    np.savetxt('./result/error_p_history.txt', error_p_history)

def generate_parm(center_bounds, u0_inlet_bounds, parm_size, load_loss_weight):
    if load_loss_weight:
        center = torch.load('./loss_weight/center.pt')
        u0_inlet = torch.load('./loss_weight/u0_inlet.pt')
        print(center.shape)
    else:
        dim = 2
        center = center_bounds[0,:] + torch.rand(parm_size,dim) * (
            center_bounds[1,:]-center_bounds[0,:])
        u0_inlet = u0_inlet_bounds[0] + torch.rand(parm_size) * (
            u0_inlet_bounds[1]-u0_inlet_bounds[0])
        '''
        center = [[-0.5,-0.1],[-0.1,0.0],[0.3,0.1]]
        center = torch.tensor(center).reshape(3,dim)
        u0_inlet = [0.80, 1.00, 1.20]
        '''
        print(center.shape)

        os.makedirs('./loss_weight', exist_ok=True)
        torch.save(center, './loss_weight/center.pt')
        torch.save(u0_inlet, './loss_weight/u0_inlet.pt')

    return center, u0_inlet

def debug_loss(tr_set, te_set, net, conv):
    tr_loader = DataLoader(tr_set, batch_size=tr_set.parm_size, shuffle=False)
    for data in tr_loader:
        data['u0'] = te_set.u0a
        data['u1'] = te_set.u1a
        data['p'] = te_set.pa
        loss = loss_func(net, conv, data)
        print(loss)

def main():
    """ Configurations """
    parser = argparse.ArgumentParser(description='Neural Network Method')
    parser.add_argument('--re', type=float, default=100,
                        help='renold number')
    parser.add_argument('--bounds', type=float, default=[-1.50,1.50, -0.50,0.50],
                        help='lower and upper bounds of the domain')
    parser.add_argument('--center_bounds', type=float, default=[-0.75,-0.15, 0.50,0.15],
                        help='lower and upper bounds of the circle center')
    parser.add_argument('--radius', type=float, default=0.2,
                        help='radius of the circle center')
    parser.add_argument('--u0_inlet_bounds', type=float, default=[0.75,1.25],
                        help='lower and upper bounds of the inlet volecity')
    parser.add_argument('--parm_size', type=int, default=100,
                        help='size of the parameter')
    parser.add_argument('--batch_size', type=int, default=25,
                        help='size of the batch')
    parser.add_argument('--nx', type=int, default=[90,30],
                        help='size of the mesh')
    parser.add_argument('--fno_modes1', type=int, default=12,
                        help='fno mode1')
    parser.add_argument('--fno_modes2', type=int, default=12,
                        help='fno mode2')
    parser.add_argument('--fno_width', type=int, default=32,
                        help='fno width')
    parser.add_argument('--epochs_o', type=int, default=1000,
                        help='number of outer iterations')
    parser.add_argument('--epochs_i', type=int, default=100,
                        help='number of inner iterations')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='device')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='learning rate')
    parser.add_argument('--load_loss_weight', type=bool, default=True,
                        help='load weight for calculate loss function')
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

    dim = 2
    bounds = torch.tensor(args.bounds).reshape(dim,2)
    center_bounds = torch.tensor(args.center_bounds).reshape(dim,2)
    u0_inlet_bounds = torch.tensor(args.u0_inlet_bounds)
    nx = torch.tensor(args.nx).int()
    center, u0_inlet = generate_parm(center_bounds, u0_inlet_bounds, 
                                     args.parm_size, args.load_loss_weight)
    
    """ geometry and mesh """ 
    geo = geometry.Geometry(bounds, center, args.radius)
    msh = mesh.Mesh(geo, bounds, nx)
    
    """ dataset """
    tr_set = dataset.TrSet(geo, msh, args.re, u0_inlet, dtype, device,
                           args.load_loss_weight)
    tr_set.to(device)
    tr_loader = DataLoader(tr_set, batch_size=args.batch_size)
    
    te_set = dataset.TeSet('reference.csv', 3, nx, dtype)
    te_set.to(device)
    
    """ network """
    net = network.FNO2d(args.fno_modes1, args.fno_modes2, args.fno_width).to(device)
    conv = network.FixedConv2D().to(device)
    print(f'amount of parameters: {network.count_params(net)}')
    
    """ optimizer and scheduler """
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.99)
    '''
    debug_loss(tr_set, te_set, net, conv)
    '''
    """ Train neural network """
    start_time = time.time()
    solve(net, conv, tr_loader, te_set, optim, sched, args.epochs_o, args.epochs_i)
    elapsed = time.time() - start_time
    print('train time: %.2f' %(elapsed))
    '''
    """ Save result """
    net.load_state_dict(torch.load('checkpoint/train/checkpoint_xfno.pth'))
    st = time.time()
    te_set.u = net(te_set.parm)
    elapsed = time.time()-st
    te_set.x0 = te_set.x0.reshape(-1,1)
    te_set.x1 = te_set.x1.view(-1,1)
    te_set.c_a = te_set.c_a.view(-1,1)
    te_set.u0_inlet = te_set.u0_inlet.view(-1,1)
    te_set.u0 = te_set.u[:,0:1,...].reshape(te_set.parm_size*te_set.nx[0]*te_set.nx[1],1)
    te_set.u1 = te_set.u[:,1:2,...].reshape(te_set.parm_size*te_set.nx[0]*te_set.nx[1],1)
    te_set.p = te_set.u[:,2:3,...].reshape(te_set.parm_size*te_set.nx[0]*te_set.nx[1],1)
    te_set.mask = te_set.mask.view(-1,1)
    te_set_pred = torch.cat([te_set.x0,te_set.x1,te_set.c_a,te_set.u0_inlet,
                             te_set.u0,te_set.u1,te_set.p,te_set.mask],1)
    np.savetxt('prediction.txt',te_set_pred.detach().cpu())
    print(f'inference time: {elapsed}')
    '''
if __name__=='__main__':
    main()
