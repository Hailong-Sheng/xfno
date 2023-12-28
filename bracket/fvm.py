import argparse
import torch
import numpy as np
import csv
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import geometry
import data

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
    parser.add_argument('--dtype', type=str, default='float64',
                        help='learning rate')
    parser.add_argument('--load_intp_coef', type=bool, default=False,
                        help='load interpolation coefficent')
    args = parser.parse_args()

    if args.dtype=='float16':
        dtype = torch.float16
    if args.dtype=='float32':
        dtype = torch.float32
    if args.dtype=='float64':
        dtype = torch.float64
    
    dim = 3
    
    center = torch.tensor(args.center)
    bounds = torch.tensor(args.bounds).reshape(dim,2)
    bounds_p1 = torch.tensor(args.bounds_p1).reshape(dim,2)
    bounds_p2 = torch.tensor(args.bounds_p2).reshape(dim,2)
    nx = torch.tensor(args.nx).int()

    geo = geometry.Geometry(bounds_p1, bounds_p2, center, args.radius)
    mesh = geometry.Mesh(geo, bounds, nx)
    
    tr_set = data.TrSet(geo, mesh, args.lm, args.mu, args.stress, dtype,
                        args.load_intp_coef)
    
    idx = tr_set.mesh.c_loc.reshape(tr_set.mesh.c_size)==1
    a00 = np.array(tr_set.a00[idx,:][:,idx])
    a01 = np.array(tr_set.a01[idx,:][:,idx])
    a02 = np.array(tr_set.a02[idx,:][:,idx])
    a10 = np.array(tr_set.a10[idx,:][:,idx])
    a11 = np.array(tr_set.a11[idx,:][:,idx])
    a12 = np.array(tr_set.a12[idx,:][:,idx])
    a20 = np.array(tr_set.a20[idx,:][:,idx])
    a21 = np.array(tr_set.a21[idx,:][:,idx])
    a22 = np.array(tr_set.a22[idx,:][:,idx])
    b0 = np.array(tr_set.b0[idx,:])
    b1 = np.array(tr_set.b1[idx,:])
    b2 = np.array(tr_set.b2[idx,:])
    
    a = np.concatenate([np.concatenate([a00,a01,a02],1),
                        np.concatenate([a10,a11,a12],1),
                        np.concatenate([a20,a21,a22],1)],0)
    a = csr_matrix(a)
    b = np.concatenate([b0,b1,b2],0)
    
    u_ = spsolve(a, b)

    u0 = np.zeros([tr_set.mesh.c_size,1])
    u1 = np.zeros([tr_set.mesh.c_size,1])
    u2 = np.zeros([tr_set.mesh.c_size,1])
    
    idx = np.array(idx, bool)
    u0[idx,0] = u_[0*idx.sum():1*idx.sum()]
    u1[idx,0] = u_[1*idx.sum():2*idx.sum()]
    u2[idx,0] = u_[2*idx.sum():3*idx.sum()]
    
    x = tr_set.mesh.c_x.reshape(tr_set.mesh.c_size,tr_set.dim)
    mask = idx.reshape(-1,1)
    solution = np.concatenate([x,u0,u1,u2,mask],-1)

    with open('solution.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for row in solution:
            csv_writer.writerow(row)

if __name__=='__main__':
    main()
