import os
import torch
import numpy as np
import h5py
import scipy.io
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import xfno

class Dataset():
    """ dataset """
    def __init__(self, geo, mesh, param_size, load_cache=False):
        """ initialization
        args:
            geo: geometry
            mesh: mesh
            param_size: size of sample in parametric space
            load_cache: whether to load cache
        """
        self.geo = geo
        self.mesh = mesh
        self.param_size = param_size

        self.dim = 2

        self.intp_func = xfno.InterpolationFunction(self.geo, self.mesh)
        
        if not load_cache:
            self.generate_param()
            self.generate_label()
            self.save_data()
        else:
            self.load_data()
        
    def generate_param(self):
        print('Generating parameter ...')

        self.param = np.zeros([self.param_size,1,self.mesh.nx[0],self.mesh.nx[1]])
        for p in range(self.param_size):
            norm_a = xfno.gaussian_random_field.grf(alpha=2, tau=3, s=int(self.mesh.nx[0]))
            for i in range(self.mesh.nx[0]):
                for j in range(self.mesh.nx[1]):
                    m = i*self.mesh.nx[1] + j
                    if self.mesh.c_loc[m]!=1:
                        continue
                    # self.param[p,0,i,j] = np.exp(norm_a[i,j])
                    self.param[p,0,i,j] = 12*(norm_a[i,j]>=0) + 4*(norm_a[i,j]<0)
        
        self.coord = self.mesh.c_x.reshape(1,self.mesh.nx[0],self.mesh.nx[1],self.dim)
        self.coord = self.coord.transpose(0,3,1,2)

        self.mask = (self.mesh.c_loc==1).reshape(1,1,self.mesh.nx[0],self.mesh.nx[1])
        
    def generate_label(self):
        print('Generating label ...')

        # interpolation value
        self.intp_func.v = np.zeros([self.param_size,self.mesh.c_size,self.intp_func.n_size])
        self.intp_func.a = np.zeros([self.param_size,self.mesh.c_size,self.intp_func.n_size])
        for p in range(self.param_size):
            intp_a = self.param[p,0,:,:].reshape(self.mesh.c_size)
            for m in range(self.mesh.c_size):
                if self.mesh.c_loc[m]!=1:
                    continue
                
                ii = self.intp_func.i[m,:]
                for n in range(self.intp_func.n_size):
                    if ii[n]!=-1:
                        self.intp_func.a[p,m,n] = intp_a[ii[n]]
                    else:
                        self.intp_func.a[p,m,n] = intp_a[m]

        # boundary value
        self.bd_val = np.zeros([self.param_size,self.mesh.c_size,self.intp_func.n_size])
        
        # coefficient matrix
        self.ma = np.zeros([self.param_size,self.mesh.c_size,self.intp_func.n_size])
        for m in range(self.mesh.c_size):
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
            
            ca = np.zeros(self.param_size)
            for n in range(self.intp_func.n_size):
                ca += c[n] * intp_a[:,n]
            for n in range(self.intp_func.n_size):
                diff = -ca * (c_x0[n]*self.mesh.fw_n[m,0] + c_x1[n]*self.mesh.fw_n[m,1])
                self.ma[:,m,n] += diff * self.mesh.fw_l[m]
            
            # east face
            if (intp_i!=-1).all():
                c, c_x0, c_x1 = self.intp_func.re_c[1,:], self.intp_func.re_c_x0[1,:], self.intp_func.re_c_x1[1,:]
            else:
                c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh.fe_x[m,:])
            
            ca = np.zeros(self.param_size)
            for n in range(self.intp_func.n_size):
                ca += c[n] * intp_a[:,n]
            for n in range(self.intp_func.n_size):
                diff = -ca * (c_x0[n]*self.mesh.fe_n[m,0] + c_x1[n]*self.mesh.fe_n[m,1])
                self.ma[:,m,n] += diff * self.mesh.fe_l[m]
            
            # south face
            if (intp_i!=-1).all():
                c, c_x0, c_x1 = self.intp_func.re_c[2,:], self.intp_func.re_c_x0[2,:], self.intp_func.re_c_x1[2,:]
            else:
                c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh.fs_x[m,:])
            
            ca = np.zeros(self.param_size)
            for n in range(self.intp_func.n_size):
                ca += c[n] * intp_a[:,n]
            for n in range(self.intp_func.n_size):
                diff = -ca * (c_x0[n]*self.mesh.fs_n[m,0] + c_x1[n]*self.mesh.fs_n[m,1])
                self.ma[:,m,n] += diff * self.mesh.fs_l[m]
            
            # north face
            if (intp_i!=-1).all():
                c, c_x0, c_x1 = self.intp_func.re_c[3,:], self.intp_func.re_c_x0[3,:], self.intp_func.re_c_x1[3,:]
            else:
                c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh.fn_x[m,:])
            
            ca = np.zeros(self.param_size)
            for n in range(self.intp_func.n_size):
                ca += c[n] * intp_a[:,n]
            for n in range(self.intp_func.n_size):
                diff = -ca * (c_x0[n]*self.mesh.fn_n[m,0] + c_x1[n]*self.mesh.fn_n[m,1])
                self.ma[:,m,n] += diff * self.mesh.fn_l[m]
            
            # west south face
            tol = 1e-6
            if self.mesh.fws_l[m]>tol:
                c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh.fws_x[m,:])
                
                ca = np.zeros(self.param_size)
                for n in range(self.intp_func.n_size):
                    ca += c[n] * intp_a[:,n]
                for n in range(self.intp_func.n_size):
                    diff = -ca * (c_x0[n]*self.mesh.fws_n[m,0] + c_x1[n]*self.mesh.fws_n[m,1])
                    self.ma[:,m,n] += diff * self.mesh.fws_l[m]
            
            # west north face
            if self.mesh.fwn_l[m]>tol:
                c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh.fwn_x[m,:])
                
                ca = np.zeros(self.param_size)
                for n in range(self.intp_func.n_size):
                    ca += c[n] * intp_a[:,n]
                for n in range(self.intp_func.n_size):
                    diff = -ca * (c_x0[n]*self.mesh.fwn_n[m,0] + c_x1[n]*self.mesh.fwn_n[m,1])
                    self.ma[:,m,n] += diff * self.mesh.fwn_l[m]
            
            # east south face
            if self.mesh.fes_l[m]>tol:
                c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh.fes_x[m,:])
                
                ca = np.zeros(self.param_size)
                for n in range(self.intp_func.n_size):
                    ca += c[n] * intp_a[:,n]
                for n in range(self.intp_func.n_size):
                    diff = -ca * (c_x0[n]*self.mesh.fes_n[m,0] + c_x1[n]*self.mesh.fes_n[m,1])
                    self.ma[:,m,n] += diff * self.mesh.fes_l[m]
            
            # east north face
            if self.mesh.fen_l[m]>tol:
                c, c_x0, c_x1 = self.intp_func.intp_coef_2(intp_x, self.mesh.fen_x[m,:])
                
                ca = np.zeros(self.param_size)
                for n in range(self.intp_func.n_size):
                    ca += c[n] * intp_a[:,n]
                for n in range(self.intp_func.n_size):
                    diff = -ca * (c_x0[n]*self.mesh.fen_n[m,0] + c_x1[n]*self.mesh.fen_n[m,1])
                    self.ma[:,m,n] += diff * self.mesh.fen_l[m]

        # right hand side
        self.vb = 1 * self.mesh.c_a.reshape(1,self.mesh.c_size,1).repeat(self.param_size, axis=0)
        
        # label
        self.label = np.zeros([self.param_size,self.mesh.c_size])
        for p in range(self.param_size):
            print(f'For the {p}th data')
            a = scipy.sparse.coo_matrix((self.mesh.c_size,self.mesh.c_size))
            b = np.zeros([self.mesh.c_size,1])

            for m in range(self.mesh.c_size):
                if self.mesh.c_loc[m]!=1:
                    continue
                
                b[m,0] += self.vb[p,m,0]
                
                xi = self.intp_func.x[m,:,:]
                ii = self.intp_func.i[m,:]
                vi = self.intp_func.v[p,m,:]
                
                for n in range(self.intp_func.n_size):
                    if ii[n]==-1:
                        b[m,0] -= self.ma[p,m,n] * vi[n]
                    else:
                        a = a + scipy.sparse.coo_matrix(([self.ma[p,m,n]], 
                            ([m],[ii[n]])), shape=(self.mesh.c_size,self.mesh.c_size))

            idx = self.mesh.c_loc.reshape(self.mesh.c_size)==1
            a = csr_matrix(a[idx,:][:,idx])
            b = np.array(self.vb[p,idx,:])
            u = spsolve(a, b)
            
            idx = np.array(idx, bool)
            self.label[p,idx] = u
        
        self.label = self.label.reshape(self.param_size,1,self.mesh.nx[0],self.mesh.nx[1])
    
    def save_data(self):
        print('saving dataset ...')
        os.makedirs('./dataset', exist_ok=True)
        with h5py.File('./dataset/dataset.h5', 'w') as f:
            f.create_dataset('param', data=self.param)
            f.create_dataset('label', data=self.label)
            f.create_dataset('coord', data=self.coord)
            f.create_dataset('mask', data=self.mask)
            f.create_dataset('bd_val', data=self.bd_val)
            f.create_dataset('ma', data=self.ma)
            f.create_dataset('vb', data=self.vb)
    
    def load_data(self):
        with h5py.File('./dataset/dataset.h5', 'r') as f:
            self.param = f['param'][:]
            self.label = f['label'][:]
            self.coord = f['coord'][:]
            self.mask = f['mask'][:]
            self.bd_val = f['bd_val'][:]
            self.ma = f['ma'][:]
            self.vb = f['vb'][:]

class DatasetFNO(torch.utils.data.Dataset):
    """ dataset for FNO """
    def __init__(self, dataset, idx, dtype=torch.float32, device='cpu'):
        """ initialization
        args:
            dataset: basic dataset
            idx: index of data in basic dataset
            dtype: datatype
            device: computing device
        """
        super().__init__()
        self.param = dataset.param[idx,:,:,:]
        self.label = dataset.label[idx,:,:,:]
        self.coord = dataset.coord
        self.mask = dataset.mask
        self.dtype = dtype
        self.device = device

        self.param = torch.tensor(self.param, dtype=self.dtype)
        self.label = torch.tensor(self.label, dtype=self.dtype)
        self.coord = torch.tensor(self.coord, dtype=self.dtype)
        self.mask = torch.tensor(self.mask, dtype=self.dtype)

    def __getitem__(self, idx):
        data = {}
        data['param'] = self.param[idx].to(self.device)
        data['label'] = self.label[idx].to(self.device)
        data['coord'] = self.coord[0].to(self.device)
        data['mask'] = self.mask[0].to(self.device)
        return data

    def __len__(self):
        return self.param.shape[0]

class DatasetXFNO(torch.utils.data.Dataset):
    """ dataset """
    def __init__(self, dataset, idx, dtype=torch.float32, device='cpu'):
        """ initialization
        args:
            dataset: basic dataset
            idx: index of data in basic dataset
            dtype: datatype
            device: computing device
        """
        self.param = dataset.param[idx,:,:,:]
        self.label = dataset.label[idx,:,:,:]
        self.coord = dataset.coord
        self.mask = dataset.mask
        self.dtype = dtype
        self.device = device

        self.param_size = self.param.shape[0]

        self.bd_val = dataset.bd_val[idx,:,:].transpose(0,2,1).reshape(
            self.param_size, dataset.intp_func.n_size, dataset.mesh.nx[0], dataset.mesh.nx[1])
        self.weight = dataset.ma[idx,:,:].transpose(0,2,1).reshape(
            self.param_size, dataset.intp_func.n_size, dataset.mesh.nx[0], dataset.mesh.nx[1])
        self.right = dataset.vb[idx,:,:].transpose(0,2,1).reshape(
            self.param_size, 1, dataset.mesh.nx[0], dataset.mesh.nx[1])
        
        self.param = torch.tensor(self.param, dtype=self.dtype)
        self.label = torch.tensor(self.label, dtype=self.dtype)
        self.coord = torch.tensor(self.coord, dtype=self.dtype)
        self.mask = torch.tensor(self.mask, dtype=self.dtype)
        self.bd_val = torch.tensor(self.bd_val, dtype=self.dtype)
        self.weight = torch.tensor(self.weight, dtype=self.dtype)
        self.right = torch.tensor(self.right, dtype=self.dtype)

    def __getitem__(self, idx):
        data = {}
        data['param'] = self.param[idx].to(self.device)
        data['label'] = self.label[idx].to(self.device)
        data['coord'] = self.coord[0].to(self.device)
        data['mask'] = self.mask[0].to(self.device)
        data['bd_val'] = self.bd_val[idx].to(self.device)
        data['weight'] = self.weight[idx].to(self.device)
        data['right'] = self.right[idx].to(self.device)
        return data

    def __len__(self):
        return self.param_size

class DataLoader(torch.utils.data.DataLoader):
    """ dataloader """
    def __init__(self, dataset, batch_size, shuffle=True):
        super().__init__(dataset, batch_size, shuffle)