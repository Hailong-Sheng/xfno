import os
import torch
import numpy as np
import h5py
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from torch_geometric.data import Data as PygData
from torch_geometric.data import Dataset as PygDataset
from torch_geometric.loader import DataLoader as PygDataLoader

import xfno

class Dataset(torch.utils.data.Dataset):
    """ dataset """
    def __init__(self, dataset_type, geo, mesh, param_size, load_cache,
                 dtype=torch.float32, device='cpu'):
        """ initialization
        args:
            dataset_type: type of dataset ('train' or 'valid')
            geo: geometry
            mesh: mesh
            param_size: size of sample in parametric space
            load_cache: whether to load cache
            dtype: datatype
            device: computing device
        """
        super().__init__()
        self.dataset_type = dataset_type
        self.geo = geo
        self.mesh = mesh
        self.param_size = param_size
        self.load_cache = load_cache
        self.dtype = dtype
        self.device = device

        # interpolation function
        self.intp_func = xfno.interpolation.InterpolationFunction(
            self.geo, self.mesh, self.dtype)
        
        # parameter
        if self.load_cache:
            self.load_param()
        else:
            self.generate_param()
            self.save_param()
        
        # boundary value
        self.generate_boundary_value()

        # loss weight
        if self.load_cache:
            self.load_loss_weight()
        else:
            self.generate_loss_weight()
            self.save_loss_weight()
        
        # label
        if self.load_cache:
            self.load_label()
        else:
            self.generate_label()
            self.save_label()

        self.save_dataset()
    
    def generate_param(self):
        print('Generating parameter ...')

        self.param = torch.zeros(self.param_size,1,self.mesh.nx[0],self.mesh.nx[1])
        for p in range(self.param_size):
            norm_a = xfno.gaussian_random_field.grf(alpha=2, tau=3, s=int(self.mesh.nx[0]))
            for i in range(self.mesh.nx[0]):
                for j in range(self.mesh.nx[1]):
                    m = i*self.mesh.nx[1] + j
                    if self.mesh.c_loc[m]!=1:
                        continue
                    # self.param[p,0,i,j] = np.exp(norm_a[i,j])
                    self.param[p,0,i,j] = 12*(norm_a[i,j]>=0) + 4*(norm_a[i,j]<0)

    def generate_boundary_value(self):
        print('Generating boundary value ...')

        # boundary value on the cell face (if cell face is located on the boundary)
        self.mesh.fw_v = torch.zeros(self.mesh.c_size)
        self.mesh.fe_v = torch.zeros(self.mesh.c_size)
        self.mesh.fs_v = torch.zeros(self.mesh.c_size)
        self.mesh.fn_v = torch.zeros(self.mesh.c_size)

        # boundary value on the interpolation node (if node is located on the boundary)
        self.intp_func.v = torch.zeros(self.param_size,self.mesh.c_size,self.intp_func.n_size)
        self.intp_func.a = torch.zeros(self.param_size,self.mesh.c_size,self.intp_func.n_size)
        for p in range(self.param_size):
            intp_a = self.param[p,0,:,:].reshape(self.mesh.c_size)
            for i in range(self.mesh.nx[0]):
                for j in range(self.mesh.nx[1]):
                    m = i*self.mesh.nx[1] + j
                    if self.mesh.c_loc[m]!=1:
                        continue
                    
                    ii = self.intp_func.i[m,:]
                    for r in range(3):
                        for s in range(3):
                            n = r*3 + s
                            if ii[n]!=-1:
                                self.intp_func.a[p,m,n] = intp_a[ii[n]]
                            else:
                                self.intp_func.a[p,m,n] = intp_a[m]
                                self.intp_func.v[p,m,n] = 0.0

        # reshape for calculating the value of loss function
        self.v = (self.intp_func.v.permute(0,2,1)).reshape(
            self.param_size,self.intp_func.n_size,self.mesh.nx[0],self.mesh.nx[1])

    def generate_loss_weight(self):
        print('Generating weight for evaluating the equation residual ...')

        # right hand side
        self.r = 1 * self.mesh.c_a.reshape(1,1,self.mesh.nx[0],self.mesh.nx[1])
        self.r = self.r.repeat(self.param_size,1,1,1)
        
        self.wei_u = torch.zeros(self.param_size,self.intp_func.n_size,self.mesh.nx[0],self.mesh.nx[1])
        for i in range(self.mesh.nx[0]):
            for j in range(self.mesh.nx[1]):
                m = i*self.mesh.nx[1] + j
                if self.mesh.c_loc[m]!=1:
                    continue
                
                intp_x = self.intp_func.x[m,:,:]
                intp_i = self.intp_func.i[m,:]
                intp_v = self.intp_func.v[:,m,:]
                intp_a = self.intp_func.a[:,m,:]
                
                # west face
                if (intp_i!=-1).all():
                    c, c_x0, c_x1 = self.intp_func.re_c[0,:], self.intp_func.re_c_x0[0,:], self.intp_func.re_c_x1[0,:]
                else:
                    c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh.fw_x[m,:])
                
                ca = torch.zeros(self.param_size)
                for n in range(self.intp_func.n_size):
                    ca += c[n] * intp_a[:,n]
                for n in range(self.intp_func.n_size):
                    diff = -ca * (c_x0[n]*self.mesh.fw_n[m,0] + c_x1[n]*self.mesh.fw_n[m,1])
                    self.wei_u[:,n,i,j] += diff * self.mesh.fw_l[m]
                
                # east face
                if (intp_i!=-1).all():
                    c, c_x0, c_x1 = self.intp_func.re_c[1,:], self.intp_func.re_c_x0[1,:], self.intp_func.re_c_x1[1,:]
                else:
                    c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh.fe_x[m,:])
                
                ca = torch.zeros(self.param_size)
                for n in range(self.intp_func.n_size):
                    ca += c[n] * intp_a[:,n]
                for n in range(self.intp_func.n_size):
                    diff = -ca * (c_x0[n]*self.mesh.fe_n[m,0] + c_x1[n]*self.mesh.fe_n[m,1])
                    self.wei_u[:,n,i,j] += diff * self.mesh.fe_l[m]
                
                # south face
                if (intp_i!=-1).all():
                    c, c_x0, c_x1 = self.intp_func.re_c[2,:], self.intp_func.re_c_x0[2,:], self.intp_func.re_c_x1[2,:]
                else:
                    c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh.fs_x[m,:])
                
                ca = torch.zeros(self.param_size)
                for n in range(self.intp_func.n_size):
                    ca += c[n] * intp_a[:,n]
                for n in range(self.intp_func.n_size):
                    diff = -ca * (c_x0[n]*self.mesh.fs_n[m,0] + c_x1[n]*self.mesh.fs_n[m,1])
                    self.wei_u[:,n,i,j] += diff * self.mesh.fs_l[m]
                
                # north face
                if (intp_i!=-1).all():
                    c, c_x0, c_x1 = self.intp_func.re_c[3,:], self.intp_func.re_c_x0[3,:], self.intp_func.re_c_x1[3,:]
                else:
                    c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh.fn_x[m,:])
                
                ca = torch.zeros(self.param_size)
                for n in range(self.intp_func.n_size):
                    ca += c[n] * intp_a[:,n]
                for n in range(self.intp_func.n_size):
                    diff = -ca * (c_x0[n]*self.mesh.fn_n[m,0] + c_x1[n]*self.mesh.fn_n[m,1])
                    self.wei_u[:,n,i,j] += diff * self.mesh.fn_l[m]
                
                # west south face
                tol = 1e-6
                if self.mesh.fws_l[m]>tol:
                    c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh.fws_x[m,:])
                    
                    ca = torch.zeros(self.param_size)
                    for n in range(self.intp_func.n_size):
                        ca += c[n] * intp_a[:,n]
                    for n in range(self.intp_func.n_size):
                        diff = -ca * (c_x0[n]*self.mesh.fws_n[m,0] + c_x1[n]*self.mesh.fws_n[m,1])
                        self.wei_u[:,n,i,j] += diff * self.mesh.fws_l[m]
                
                # west north face
                if self.mesh.fwn_l[m]>tol:
                    c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh.fwn_x[m,:])
                    
                    ca = torch.zeros(self.param_size)
                    for n in range(self.intp_func.n_size):
                        ca += c[n] * intp_a[:,n]
                    for n in range(self.intp_func.n_size):
                        diff = -ca * (c_x0[n]*self.mesh.fwn_n[m,0] + c_x1[n]*self.mesh.fwn_n[m,1])
                        self.wei_u[:,n,i,j] += diff * self.mesh.fwn_l[m]
                
                # east south face
                if self.mesh.fes_l[m]>tol:
                    c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh.fes_x[m,:])
                    
                    ca = torch.zeros(self.param_size)
                    for n in range(self.intp_func.n_size):
                        ca += c[n] * intp_a[:,n]
                    for n in range(self.intp_func.n_size):
                        diff = -ca * (c_x0[n]*self.mesh.fes_n[m,0] + c_x1[n]*self.mesh.fes_n[m,1])
                        self.wei_u[:,n,i,j] += diff * self.mesh.fes_l[m]
                
                # east north face
                if self.mesh.fen_l[m]>tol:
                    c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh.fen_x[m,:])
                    
                    ca = torch.zeros(self.param_size)
                    for n in range(self.intp_func.n_size):
                        ca += c[n] * intp_a[:,n]
                    for n in range(self.intp_func.n_size):
                        diff = -ca * (c_x0[n]*self.mesh.fen_n[m,0] + c_x1[n]*self.mesh.fen_n[m,1])
                        self.wei_u[:,n,i,j] += diff * self.mesh.fen_l[m]
    
    def generate_label(self):
        print('Generating label ...')

        self.a, self.b = [], []
        for p in range(self.param_size):
            print(f'for the {p}th data')
            a = scipy.sparse.coo_matrix((self.mesh.c_size,self.mesh.c_size))
            b = torch.zeros(self.mesh.c_size,1)
            for i in range(self.mesh.nx[0]):
                for j in range(self.mesh.nx[1]):
                    m = i*self.mesh.nx[1] + j
                    if self.mesh.c_loc[m]!=1:
                        continue
                    
                    b[m] += self.r[p,0,i,j]
                    
                    xi = self.intp_func.x[m,:,:]
                    ii = self.intp_func.i[m,:]
                    vi = self.intp_func.v[p,m,:]
                    
                    for n in range(self.intp_func.n_size):
                        if ii[n]==-1:
                            b[m] -= self.wei_u[p,n,i,j] * vi[n]
                        else:
                            a = a + scipy.sparse.coo_matrix(([self.wei_u[p,n,i,j]], 
                                ([m],[ii[n]])), shape=(self.mesh.c_size,self.mesh.c_size))
            self.a.append(a)
            self.b.append(b)

        self.label = np.zeros([self.param_size,self.mesh.c_size])
        for p in range(self.param_size):
            idx = self.mesh.c_loc.reshape(self.mesh.c_size)==1
            a = self.a[p][idx,:][:,idx]
            a = a.toarray()
            a = csr_matrix(a)
            b = np.array(self.b[p][idx,:])
            
            u_ = spsolve(a, b)
            
            idx = np.array(idx, bool)
            self.label[p,idx] = u_

        param = self.param.reshape(self.param_size,1,self.mesh.nx[0],self.mesh.nx[1])
        label = self.label.reshape(self.param_size,1,self.mesh.nx[0],self.mesh.nx[1])
        mask = idx.reshape(1,1,self.mesh.nx[0],self.mesh.nx[1])
        
        self.param = torch.as_tensor(param)
        self.label = torch.as_tensor(label)
        self.mask = torch.as_tensor(mask)
    
    def load_param(self):
        print('loading parameter ...')
        self.param = torch.load(f'./cache/{self.dataset_type}/param.pt')
        self.param_size = self.param.shape[0]
        print('parameter shape:')
        print(self.param.shape)
        print(f"parameter mean: {self.param.mean():.5e}, std: {self.param.std():.5e}")
    
    def save_param(self):
        print('saving parameter ...')
        os.makedirs(f'./cache/{self.dataset_type}', exist_ok=True)
        torch.save(self.param, f'./cache/{self.dataset_type}/param.pt')

    def load_loss_weight(self):
        print('loading loss weight ...')
        self.r = torch.load(f'./cache/{self.dataset_type}/r.pt')
        self.wei_u = torch.load(f'./cache/{self.dataset_type}/wei_u.pt')

    def save_loss_weight(self):
        print('saving loss weight ...')
        os.makedirs(f'./cache/{self.dataset_type}', exist_ok=True)
        torch.save(self.r, f'./cache/{self.dataset_type}/r.pt')
        torch.save(self.wei_u, f'./cache/{self.dataset_type}/wei_u.pt')
    
    def load_label(self):
        print('loading label ...')
        self.label = torch.load(f'./cache/{self.dataset_type}/label.pt')
        self.mask = torch.load(f'./cache/{self.dataset_type}/mask.pt')
    
    def save_label(self):
        print('saving label ...')
        os.makedirs(f'./cache/{self.dataset_type}', exist_ok=True)
        torch.save(self.label, f'./cache/{self.dataset_type}/label.pt')
        torch.save(self.mask, f'./cache/{self.dataset_type}/mask.pt')
    
    def load_dataset(self):
        print('loading dataset ...')
        data = h5py.File(f'./dataset/{self.dataset_type}_{self.mesh.nx[0]}.hdf5', "r")
        self.param = torch.as_tensor(data['param'])
        self.label = torch.as_tensor(data['label'])
        self.mask = torch.as_tensor(data['mask'])
    
    def save_dataset(self):
        print('saving dataset ...')
        data_dict = {}
        data_dict['param'] = np.array(self.param)
        data_dict['label'] = np.array(self.label)
        data_dict['mask'] = np.array(self.mask)

        os.makedirs(f'./dataset', exist_ok=True)
        with h5py.File(f'./dataset/{self.dataset_type}_{self.mesh.nx[0]}.hdf5', 'w') as f:
            for key, value in data_dict.items():
                f.create_dataset(key, data=value)

    def __getitem__(self, idx):
        data = {}
        data['param'] = self.param[idx].to(self.device)
        data['label'] = self.label[idx].to(self.device)
        data['mask'] = self.mask[0].to(self.device)
        data['wei_u'] = self.wei_u[idx].to(self.device)
        data['r'] = self.r[idx].to(self.device)
        return data

    def __len__(self):
        return self.param.shape[0]

class DatasetGeoFNO():
    def __init__(self, dataset_type, dataset_intp, geo, mesh, param_size,
                 dtype=torch.float32, device='cpu'):
        self.dataset_type = dataset_type
        self.dataset_intp = dataset_intp
        self.geo = geo
        self.mesh = mesh
        self.param_size = param_size
        self.dtype = dtype
        self.device = device

        self.mesh_non = self.mesh
        self.mesh_car = dataset_intp.mesh
        self.intp_func = dataset_intp.intp_func

        self.intp_func.m = torch.zeros(self.mesh_non.cen_size, dtype=torch.long)
        for i in range(self.mesh_non.nx[0]):
            for j in range(self.mesh_non.nx[1]):
                m = i*self.mesh_non.nx[1] + j
                if self.mesh_non.cen_loc[m]!=1:
                    continue
                
                dis = ((self.mesh_car.c_x-self.mesh_non.cen_x[m,:])**2).sum(1,keepdims=True)
                dis[self.mesh_car.c_loc!=1] = 10000
                mm = torch.argmin(dis[:,0])
                self.intp_func.m[m] = mm
        
        self.param = torch.zeros(self.param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1])
        self.label = torch.zeros(self.param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1])
        self.a = torch.zeros(self.param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1])
        self.ax0 = torch.zeros(self.param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1])
        self.ax1 = torch.zeros(self.param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1])
        for p in range(self.param_size):
            for i in range(self.mesh_non.nx[0]):
                for j in range(self.mesh_non.nx[1]):
                    m = i*self.mesh_non.nx[1] + j
                    if self.mesh_non.cen_loc[m]!=1:
                        continue
                    
                    mm = self.intp_func.m[m]
                    intp_x = self.intp_func.x[mm,:,:]
                    intp_i = self.intp_func.i[mm,:]
                    intp_a = self.intp_func.a[p,mm,:]

                    intp_v = torch.zeros(self.intp_func.n_size)
                    for n in range(self.intp_func.n_size):
                        if intp_i[n]!=-1:
                            ii = intp_i[n]//self.mesh_car.nx[1]
                            jj = intp_i[n]-ii*self.mesh_car.nx[1]
                            intp_v[n] = self.dataset_intp.label[p,0,ii,jj]
                    
                    c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh_non.cen_x[m,:])
                    for n in range(self.intp_func.n_size):
                        self.param[p,0,i,j] += c[n] * intp_a[n]
                        self.label[p,0,i,j] += c[n] * intp_v[n]
                        self.a[p,0,i,j] += c[n] * intp_a[n]
                        self.ax0[p,0,i,j] += c_x0[n] * intp_a[n]
                        self.ax1[p,0,i,j] += c_x1[n] * intp_a[n]

        self.mask = (self.mesh_non.cen_loc==1).reshape(1,1,self.mesh_non.nx[0],self.mesh_non.nx[1])

    def __getitem__(self, idx):
        data = {}
        data['param'] = self.param[idx].to(self.device)
        data['label'] = self.label[idx].to(self.device)
        data['mask'] = self.mask[0].to(self.device)
        return data

    def __len__(self):
        return self.param.shape[0]

class DatasetGeoPINO():
    def __init__(self, dataset_type, dataset_intp, geo, mesh, param_size,
                 dtype=torch.float32, device='cpu'):
        self.dataset_type = dataset_type
        self.dataset_intp = dataset_intp
        self.geo = geo
        self.mesh = mesh
        self.param_size = param_size
        self.dtype = dtype
        self.device = device

        self.mesh_non = self.mesh
        self.mesh_car = dataset_intp.mesh
        self.intp_func = dataset_intp.intp_func
        
        self.x = torch.zeros(self.param_size,self.mesh_non.dim,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
        self.y = torch.zeros(self.param_size,self.mesh_non.dim,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
        for k in range(self.mesh_non.dim):
            self.x[:,k,:,:] = self.mesh_non.cor_x[:,k].reshape(1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
            self.y[:,k,:,:] = self.mesh_non.cor_y[:,k].reshape(1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
        self.x.requires_grad = True
        self.y.requires_grad = True

        self.intp_func.m = torch.zeros(self.mesh_non.cor_size, dtype=torch.long)
        for i in range(self.mesh_non.nx[0]+1):
            for j in range(self.mesh_non.nx[1]+1):
                m = i*(self.mesh_non.nx[1]+1) + j
                if self.mesh_non.cor_loc[m]!=1:
                    continue
                
                dis = ((self.mesh_car.c_x-self.mesh_non.cor_x[m,:])**2).sum(1,keepdims=True)
                dis[self.mesh_car.c_loc!=1] = 10000
                mm = torch.argmin(dis[:,0])
                self.intp_func.m[m] = mm
        
        self.param = torch.zeros(self.param_size,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
        self.label = torch.zeros(self.param_size,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
        self.a = torch.zeros(self.param_size,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
        self.ax0 = torch.zeros(self.param_size,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
        self.ax1 = torch.zeros(self.param_size,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
        for p in range(self.param_size):
            for i in range(self.mesh_non.nx[0]+1):
                for j in range(self.mesh_non.nx[1]+1):
                    m = i*(self.mesh_non.nx[1]+1) + j
                    if self.mesh_non.cor_loc[m]!=1:
                        continue
                    
                    mm = self.intp_func.m[m]
                    intp_x = self.intp_func.x[mm,:,:]
                    intp_i = self.intp_func.i[mm,:]
                    intp_a = self.intp_func.a[p,mm,:]
                    
                    intp_v = torch.zeros(self.intp_func.n_size)
                    for n in range(self.intp_func.n_size):
                        if intp_i[n]!=-1:
                            ii = intp_i[n]//self.mesh_car.nx[1]
                            jj = intp_i[n]-ii*self.mesh_car.nx[1]
                            intp_v[n] = self.dataset_intp.label[p,0,ii,jj]
                    
                    c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh_non.cor_x[m,:])
                    for n in range(self.intp_func.n_size):
                        self.param[p,0,i,j] += c[n] * intp_a[n]
                        self.label[p,0,i,j] += c[n] * intp_v[n]
                        self.a[p,0,i,j] += c[n] * intp_a[n]
                        self.ax0[p,0,i,j] += c_x0[n] * intp_a[n]
                        self.ax1[p,0,i,j] += c_x1[n] * intp_a[n]

        self.mask = self.mesh_non.cor_loc.reshape(1,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)

        self.cordinate_transformation()

    def cordinate_transformation(self, model_c=None):
        if model_c==None:
            self.c0 = torch.zeros(self.param_size,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
            self.c0x0 = torch.zeros(self.param_size,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
            self.c0x1 = torch.zeros(self.param_size,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
            self.c0x0x0 = torch.zeros(self.param_size,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
            self.c0x1x1 = torch.zeros(self.param_size,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)

            self.c1 = torch.zeros(self.param_size,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
            self.c1x0 = torch.zeros(self.param_size,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
            self.c1x1 = torch.zeros(self.param_size,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
            self.c1x0x0 = torch.zeros(self.param_size,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
            self.c1x1x1 = torch.zeros(self.param_size,1,self.mesh_non.nx[0]+1,self.mesh_non.nx[1]+1)
        else:
            self.x = self.x.to(self.device)
            self.c = model_c(self.x)
            self.c0 = self.c[:,0:1,:,:]
            self.c1 = self.c[:,1:2,:,:]

            self.c0x, = torch.autograd.grad(self.c0.sum(), self.x, create_graph=True)
            self.c0x0 = self.c0x[:,0:1,:,:]
            self.c0x1 = self.c0x[:,1:2,:,:]
            self.c0x0x, = torch.autograd.grad(self.c0x0.sum(), self.x, create_graph=True)
            self.c0x0x0 = self.c0x0x[:,0:1,:,:]
            self.c0x1x, = torch.autograd.grad(self.c0x1.sum(), self.x, create_graph=True)
            self.c0x1x1 = self.c0x1x[:,1:2,:,:]

            self.c1x, = torch.autograd.grad(self.c1.sum(), self.x, create_graph=True)
            self.c1x0 = self.c1x[:,0:1,:,:]
            self.c1x1 = self.c1x[:,1:2,:,:]
            self.c1x0x, = torch.autograd.grad(self.c1x0.sum(), self.x, create_graph=True)
            self.c1x0x0 = self.c1x0x[:,0:1,:,:]
            self.c1x1x, = torch.autograd.grad(self.c1x1.sum(), self.x, create_graph=True)
            self.c1x1x1 = self.c1x1x[:,1:2,:,:]

            self.x = self.x.to('cpu')

            self.c0 = self.c0.detach()
            self.c0x0 = self.c0x0.detach()
            self.c0x1 = self.c0x1.detach()
            self.c0x0x0 = self.c0x0x0.detach()
            self.c0x1x1 = self.c0x1x1.detach()

            self.c1 = self.c1.detach()
            self.c1x0 = self.c1x0.detach()
            self.c1x1 = self.c1x1.detach()
            self.c1x0x0 = self.c1x0x0.detach()
            self.c1x1x1 = self.c1x1x1.detach()

    def __getitem__(self, idx):
        data = {}
        data['param'] = self.param[idx].to(self.device)
        data['label'] = self.label[idx].to(self.device)
        data['mask'] = self.mask[0].to(self.device)
        
        data['x'] = self.x[idx].to(self.device)
        data['y'] = self.y[idx].to(self.device)
        data['a'] = self.a[idx].to(self.device)
        data['ax0'] = self.ax0[idx].to(self.device)
        data['ax1'] = self.ax1[idx].to(self.device)
        
        data['c0'] = self.c0[idx].to(self.device)
        data['c0x0'] = self.c0x0[idx].to(self.device)
        data['c0x1'] = self.c0x1[idx].to(self.device)
        data['c0x0x0'] = self.c0x0x0[idx].to(self.device)
        data['c0x1x1'] = self.c0x1x1[idx].to(self.device)
        
        data['c1'] = self.c1[idx].to(self.device)
        data['c1x0'] = self.c1x0[idx].to(self.device)
        data['c1x1'] = self.c1x1[idx].to(self.device)
        data['c1x0x0'] = self.c1x0x0[idx].to(self.device)
        data['c1x1x1'] = self.c1x1x1[idx].to(self.device)
        return data

    def __len__(self):
        return self.param.shape[0]

class DatasetGINO(PygDataset):
    def __init__(self, dataset_type, dataset_intp, geo, mesh, param_size,
                 dtype=torch.float32, device='cpu'):
        super(DatasetGINO, self).__init__()
        self.dataset_type = dataset_type
        self.dataset_intp = dataset_intp
        self.geo = geo
        self.mesh = mesh
        self.param_size = param_size
        self.dtype = dtype
        self.device = device
        
        self.mesh_non = self.mesh
        self.mesh_car = dataset_intp.mesh
        self.intp_func = dataset_intp.intp_func

        self.intp_func.m = torch.zeros(self.mesh_non.cen_size, dtype=torch.long)
        for i in range(self.mesh_non.nx[0]):
            for j in range(self.mesh_non.nx[1]):
                m = i*self.mesh_non.nx[1] + j
                if self.mesh_non.cen_loc[m]!=1:
                    continue
                
                dis = ((self.mesh_car.c_x-self.mesh_non.cen_x[m,:])**2).sum(1,keepdims=True)
                dis[self.mesh_car.c_loc!=1] = 10000
                mm = torch.argmin(dis[:,0])
                self.intp_func.m[m] = mm
        
        self.param = torch.zeros(self.param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1])
        self.label = torch.zeros(self.param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1])
        for p in range(self.param_size):
            for i in range(self.mesh_non.nx[0]):
                for j in range(self.mesh_non.nx[1]):
                    m = i*self.mesh_non.nx[1] + j
                    if self.mesh_non.cen_loc[m]!=1:
                        continue

                    mm = self.intp_func.m[m]
                    intp_x = self.intp_func.x[mm,:,:]
                    intp_i = self.intp_func.i[mm,:]
                    intp_a = self.intp_func.a[p,mm,:]

                    intp_v = torch.zeros(self.intp_func.n_size)
                    for n in range(self.intp_func.n_size):
                        if intp_i[n]!=-1:
                            ii = intp_i[n]//self.mesh_car.nx[1]
                            jj = intp_i[n]-ii*self.mesh_car.nx[1]
                            intp_v[n] = self.dataset_intp.label[p,0,ii,jj]
                    
                    c, _, _ = self.intp_func.intp_coef_2(intp_x, self.mesh_non.cen_x[m,:])
                    for n in range(self.intp_func.n_size):
                        self.param[p,0,i,j] += c[n] * intp_a[n]
                        self.label[p,0,i,j] += c[n] * intp_v[n]
        
        self.mask = (self.mesh_non.cen_loc==1).reshape(1,1,self.mesh_non.nx[0],self.mesh_non.nx[1])

        self.graph = []
        for p in range(self.param_size):
            param = torch.zeros(self.mesh_non.cen_size+self.mesh_non.cen_size,1)
            param[:self.mesh_non.cen_size,:] = self.param[p].reshape(self.mesh_non.cen_size,1)
            
            node_pos = torch.cat([self.mesh_non.cen_x, self.mesh_non.cen_y])
            edge_index = self.mesh_non.ball_connectivity(0.2,'encode')
            edge_attr = torch.cat([node_pos[edge_index[0,:],:], node_pos[edge_index[1,:],:], 
                                   node_pos[edge_index[0,:],:]-node_pos[edge_index[1,:],:]], 1)
            
            graph = PygData(x=param, y=self.label[p:p+1], mask=self.mask,
                            edge_index=edge_index, edge_attr=edge_attr)        
            self.graph.append(graph)

    def get(self, idx):
        return self.graph[idx].to(self.device)

    def len(self):
        return len(self.graph)

class DataLoader(torch.utils.data.DataLoader):
    """ dataLoader for loading data """
    def __init__(self, dataset, batch_size: int=1, shuffle: bool=True):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle)
        """ initialization
        args:
            dataset: dataset class
            batch_size: size of batch data
            shuffle: whether to shuffle the data
        """

class PyGDataLoader(PygDataLoader):
    """ dataLoader for loading graph data """
    def __init__(self, dataset, batch_size: int=1, shuffle: bool=True):
        super(PyGDataLoader, self).__init__(dataset, batch_size, shuffle)
        """ initialization
        args:
            dataset: dataset class
            batch_size: size of batch data
            shuffle: whether to shuffle the data
        """