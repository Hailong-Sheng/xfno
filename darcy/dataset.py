import os
import torch
import numpy as np
import h5py
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from torch.utils.data import DataLoader

import gaussian_random_field

class Dataset(torch.utils.data.Dataset):
    """ Dataset
    Args:
    """
    def __init__(self, dataset_type, mesh, param_size, load_cache,
                 dtype=torch.float32, device='cpu'):
        super().__init__()
        self.dataset_type = dataset_type
        self.mesh = mesh
        self.param_size = param_size
        self.load_cache = load_cache
        self.dtype = dtype
        self.device = device

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
            self.calculate_loss_weight()
            self.save_loss_weight()
        
        # label
        if self.load_cache:
            self.load_label()
        else:
            self.calculate_label()
            self.save_label()

        self.save_dataset()
    
    def generate_param(self):
        self.norm_a = torch.zeros(self.param_size,1,self.mesh.nx[0],self.mesh.nx[1])
        self.param = torch.zeros(self.param_size,1,self.mesh.nx[0],self.mesh.nx[1])
        for p in range(self.param_size):
            norm_a = gaussian_random_field.grf(alpha=2, tau=3, s=int(self.mesh.nx[0]))

            for i in range(self.mesh.nx[0]):
                for j in range(self.mesh.nx[1]):
                    m = i*self.mesh.nx[1] + j
                    if self.mesh.c_loc[m]!=1:
                        continue

                    self.norm_a[p,0,i,j] = norm_a[i,j]
                    self.param[p,0,i,j] = self.convert_a(norm_a[i,j])
        '''
        data = pd.read_csv('solution.csv', header=None)
        data = np.array(data)
        data = torch.tensor(data, dtype=self.dtype)
        self.norm_a = data[:,2:3].reshape(self.param_size,1,self.mesh.nx[0],self.mesh.nx[1])
        print(self.norm_a)
        '''
    def convert_a(self, norm_a):
        lognorm_a = np.exp(norm_a)
        if isinstance(norm_a, float):
            thresh_a = 12 if norm_a>=0 else 4
        else:
            thresh_a = 12*(norm_a>=0) + 4*(norm_a<0)
        return thresh_a
        
    def generate_boundary_value(self):
        print('Generating boundary value ...')

        """ boundary value on the cell face (if cell face is located on the boundary) """
        self.mesh.fw_v = torch.zeros(self.mesh.c_size)
        self.mesh.fe_v = torch.zeros(self.mesh.c_size)
        self.mesh.fs_v = torch.zeros(self.mesh.c_size)
        self.mesh.fn_v = torch.zeros(self.mesh.c_size)

        """ boundary value on the interpolation node (if node is located on the boundary) """
        self.mesh.intp_v = torch.zeros(self.param_size,self.mesh.c_size,self.mesh.intp_n_size)
        self.mesh.intp_a = torch.zeros(self.param_size,self.mesh.c_size,self.mesh.intp_n_size)
        self.mesh.c_norm_a = self.norm_a.reshape(self.param_size,self.mesh.c_size)
        for p in range(self.param_size):
            for i in range(self.mesh.nx[0]):
                for j in range(self.mesh.nx[1]):
                    m = i*self.mesh.nx[1] + j
                    if self.mesh.c_loc[m]!=1:
                        continue

                    ii = self.mesh.intp_i[m,:]
                    for r in range(3):
                        for s in range(3):
                            n = r*3 + s
                            if ii[n]==-1:
                                self.mesh.intp_v[p,m,n] = 0.0
                                self.mesh.intp_a[p,m,n] = 0.0
                            else:
                                self.mesh.intp_a[p,m,n] = self.convert_a(self.mesh.c_norm_a[p,ii[n]])
        
        for p in range(self.param_size):
            for i in range(self.mesh.nx[0]):
                for j in range(self.mesh.nx[1]):
                    m = i*self.mesh.nx[1] + j
                    if self.mesh.c_loc[m]!=1:
                        continue

                    ii = self.mesh.intp_i[m,:]
                    if (ii==-1).sum()>0:
                        dis = 10000
                        for r in range(3):
                            for s in range(3):
                                rr = i+r-1; ss = j+s-1
                                if rr>=0 and rr<self.mesh.nx[0] and ss>=0 and ss<self.mesh.nx[1]:
                                    mm = rr*self.mesh.nx[1] + ss
                                    intp_x = self.mesh.intp_x[mm,:,:]
                                    intp_a = self.mesh.intp_a[p,mm,:]
                                    intp_i = self.mesh.intp_i[mm,:]
                                    if (intp_i==-1).sum()==0:
                                        tmp_dis = ((intp_x-self.mesh.c_x[m,:])**2).sum()
                                        if tmp_dis<dis:
                                            xi = intp_x; ai = intp_a; dis = tmp_dis
                    
                    for r in range(3):
                        for s in range(3):
                            n = r*3 + s

                            if ii[n]==-1:
                                c, c_x1, c_x2 = self.intp_coef_2(xi, self.mesh.intp_x[m,n,:])
                                ca = 0
                                for nn in range(self.mesh.intp_n_size):
                                    ca = ca + c[nn] * ai[nn]

                                self.mesh.intp_a[p,m,n] = self.convert_a(ca)
        
        # Reshape for calculating the value of loss function
        self.v = (self.mesh.intp_v.permute(0,2,1)).reshape(
            self.param_size,self.mesh.intp_n_size,self.mesh.nx[0],self.mesh.nx[1])

    def calculate_loss_weight(self):
        """ weight for calculating the value of loss function """
        print('Generating weight for evaluating the residual of equation ...')

        """ right hand side """
        self.r = 1 * self.mesh.c_a.reshape(1,1,self.mesh.nx[0],self.mesh.nx[1])
        self.r = self.r.repeat(self.param_size,1,1,1)

        """ interpolation coefficients for regular unit """
        self.re_c = torch.zeros(4,self.mesh.intp_n_size)
        self.re_c_x0 = torch.zeros(4,self.mesh.intp_n_size)
        self.re_c_x1 = torch.zeros(4,self.mesh.intp_n_size)
        flag = False
        for i in range(self.mesh.nx[0]):
            for j in range(self.mesh.nx[1]):
                m = i*self.mesh.nx[1] + j
                if self.mesh.c_loc[m]!=1:
                    continue
                
                xi = self.mesh.intp_x[m,:,:]
                ii = self.mesh.intp_i[m,:]

                if (ii!=-1).all():
                    self.re_c[0,:], self.re_c_x0[0,:], self.re_c_x1[0,:] = \
                        self.intp_coef_2(xi, self.mesh.fw_x[m,:])
                    self.re_c[1,:], self.re_c_x0[1,:], self.re_c_x1[1,:] = \
                        self.intp_coef_2(xi, self.mesh.fe_x[m,:])
                    self.re_c[2,:], self.re_c_x0[2,:], self.re_c_x1[2,:] = \
                        self.intp_coef_2(xi, self.mesh.fs_x[m,:])
                    self.re_c[3,:], self.re_c_x0[3,:], self.re_c_x1[3,:] = \
                        self.intp_coef_2(xi, self.mesh.fn_x[m,:])
                    flag = True

                if flag: break
            if flag: break
        
        self.wei_u = torch.zeros(self.param_size,self.mesh.intp_n_size,self.mesh.nx[0],self.mesh.nx[1])
        for i in range(self.mesh.nx[0]):
            for j in range(self.mesh.nx[1]):
                m = i*self.mesh.nx[1] + j
                # print(m)
                if self.mesh.c_loc[m]!=1:
                    continue
                
                intp_x = self.mesh.intp_x[m,:,:]
                intp_i = self.mesh.intp_i[m,:]
                intp_v = self.mesh.intp_v[:,m,:]
                intp_a = self.mesh.intp_a[:,m,:]
                
                # west face
                c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fw_x[m,:])
                ca = torch.zeros(self.param_size)
                for n in range(self.mesh.intp_n_size):
                    ca += c[n] * intp_a[:,n]

                if (intp_i!=-1).all():
                    c, c_x0, c_x1 = self.re_c[0,:], self.re_c_x0[0,:], self.re_c_x1[0,:]
                else:
                    c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fw_x[m,:])
                for n in range(self.mesh.intp_n_size):
                    diff = -ca * (c_x0[n]*self.mesh.fw_n[m,0] + c_x1[n]*self.mesh.fw_n[m,1])
                    self.wei_u[:,n,i,j] += diff * self.mesh.fw_l[m]
                
                # east face
                c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fe_x[m,:])
                ca = torch.zeros(self.param_size)
                for n in range(self.mesh.intp_n_size):
                    ca += c[n] * intp_a[:,n]

                if (intp_i!=-1).all():
                    c, c_x0, c_x1 = self.re_c[1,:], self.re_c_x0[1,:], self.re_c_x1[1,:]
                else:
                    c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fe_x[m,:])
                for n in range(self.mesh.intp_n_size):
                    diff = -ca * (c_x0[n]*self.mesh.fe_n[m,0] + c_x1[n]*self.mesh.fe_n[m,1])
                    self.wei_u[:,n,i,j] += diff * self.mesh.fe_l[m]
                
                # south face
                c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fs_x[m,:])
                ca = torch.zeros(self.param_size)
                for n in range(self.mesh.intp_n_size):
                    ca += c[n] * intp_a[:,n]

                if (intp_i!=-1).all():
                    c, c_x0, c_x1 = self.re_c[2,:], self.re_c_x0[2,:], self.re_c_x1[2,:]
                else:
                    c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fs_x[m,:])
                for n in range(self.mesh.intp_n_size):
                    diff = -ca * (c_x0[n]*self.mesh.fs_n[m,0] + c_x1[n]*self.mesh.fs_n[m,1])
                    self.wei_u[:,n,i,j] += diff * self.mesh.fs_l[m]
                
                # north face
                c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fn_x[m,:])
                ca = torch.zeros(self.param_size)
                for n in range(self.mesh.intp_n_size):
                    ca += c[n] * intp_a[:,n]

                if (intp_i!=-1).all():
                    c, c_x0, c_x1 = self.re_c[3,:], self.re_c_x0[3,:], self.re_c_x1[3,:]
                else:
                    c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fn_x[m,:])
                for n in range(self.mesh.intp_n_size):
                    diff = -ca * (c_x0[n]*self.mesh.fn_n[m,0] + c_x1[n]*self.mesh.fn_n[m,1])
                    self.wei_u[:,n,i,j] += diff * self.mesh.fn_l[m]
                
                # west south face
                tol = 1e-6
                if self.mesh.fws_l[m]>tol:
                    c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fws_x[m,:])
                    ca = torch.zeros(self.param_size)
                    for n in range(self.mesh.intp_n_size):
                        ca += c[n] * intp_a[:,n]

                    c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fws_x[m,:])
                    for n in range(self.mesh.intp_n_size):
                        diff = -ca * (c_x0[n]*self.mesh.fws_n[m,0] + c_x1[n]*self.mesh.fws_n[m,1])
                        self.wei_u[:,n,i,j] += diff * self.mesh.fws_l[m]
                
                # west north face
                if self.mesh.fwn_l[m]>tol:
                    c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fwn_x[m,:])
                    ca = torch.zeros(self.param_size)
                    for n in range(self.mesh.intp_n_size):
                        ca += c[n] * intp_a[:,n]

                    c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fwn_x[m,:])
                    for n in range(self.mesh.intp_n_size):
                        diff = -ca * (c_x0[n]*self.mesh.fwn_n[m,0] + c_x1[n]*self.mesh.fwn_n[m,1])
                        self.wei_u[:,n,i,j] += diff * self.mesh.fwn_l[m]
                
                # east south face
                if self.mesh.fes_l[m]>tol:
                    c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fes_x[m,:])
                    ca = torch.zeros(self.param_size)
                    for n in range(self.mesh.intp_n_size):
                        ca += c[n] * intp_a[:,n]

                    c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fes_x[m,:])
                    for n in range(self.mesh.intp_n_size):
                        diff = -ca * (c_x0[n]*self.mesh.fes_n[m,0] + c_x1[n]*self.mesh.fes_n[m,1])
                        self.wei_u[:,n,i,j] += diff * self.mesh.fes_l[m]
                
                # east north face
                if self.mesh.fen_l[m]>tol:
                    c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fen_x[m,:])
                    ca = torch.zeros(self.param_size)
                    for n in range(self.mesh.intp_n_size):
                        ca += c[n] * intp_a[:,n]

                    c, c_x0, c_x1 = self.intp_coef_2(intp_x, self.mesh.fen_x[m,:])
                    for n in range(self.mesh.intp_n_size):
                        diff = -ca * (c_x0[n]*self.mesh.fen_n[m,0] + c_x1[n]*self.mesh.fen_n[m,1])
                        self.wei_u[:,n,i,j] += diff * self.mesh.fen_l[m]
        
        #self.wei_u /= self.mesh.c_a.max()
        #self.r /= self.mesh.c_a.max()

    def intp_coef_2(self, xi, x):
        xi = xi.clone().to(torch.float64)
        x = x.clone().to(torch.float64)
        
        intp_n_size = 3**2
        p = torch.zeros(intp_n_size,intp_n_size, dtype=torch.float64)
        for r in range(3):
            for s in range(3):
                n = r*3 + s
                p[:,n] = xi[:,0]**r * xi[:,1]**s
        b = p
        b = torch.inverse(b)
        
        pp = torch.zeros(1,intp_n_size, dtype=torch.float64)
        pp_x0 = torch.zeros(1,intp_n_size, dtype=torch.float64)
        pp_x1 = torch.zeros(1,intp_n_size, dtype=torch.float64)
        for r in range(3):
            for s in range(3):
                n = r*3 + s
                pp[0,n] = x[0]**r * x[1]**s
                if r==0:
                    pp_x0[0,n] = 0 * x[1]**s
                else:
                    pp_x0[0,n] = r*x[0]**(r-1) * x[1]**s
                if s==0:
                    pp_x1[0,n] = x[0]**r * 0
                else:
                    pp_x1[0,n] = x[0]**r * s*x[1]**(s-1)
        
        c = (pp @ b).reshape(intp_n_size)
        c_x0 = (pp_x0 @ b).reshape(intp_n_size)
        c_x1 = (pp_x1 @ b).reshape(intp_n_size)
        
        c = c.clone().to(self.dtype)
        c_x0 = c_x0.clone().to(self.dtype)
        c_x1 = c_x1.clone().to(self.dtype)
        return c, c_x0, c_x1
    
    def calculate_label(self):
        self.a, self.b = [], []
        for p in range(self.param_size):
            print(p)
            a = scipy.sparse.coo_matrix((self.mesh.c_size,self.mesh.c_size))
            b = torch.zeros(self.mesh.c_size,1)
            for i in range(self.mesh.nx[0]):
                for j in range(self.mesh.nx[1]):
                    m = i*self.mesh.nx[1] + j
                    if self.mesh.c_loc[m]!=1:
                        continue
                    
                    b[m] += self.r[p,0,i,j]
                    
                    xi = self.mesh.intp_x[m,:,:]
                    ii = self.mesh.intp_i[m,:]
                    vi = self.mesh.intp_v[p,m,:]
                    
                    for n in range(self.mesh.intp_n_size):
                        if ii[n]==-1:
                            b[m] -= self.wei_u[p,n,i,j] * vi[n]
                        else:
                            a = a + scipy.sparse.coo_matrix(([self.wei_u[p,n,i,j]], 
                                ([m],[ii[n]])), shape=(self.mesh.c_size,self.mesh.c_size))
            self.a.append(a)
            self.b.append(b)

        self.label = np.zeros([self.param_size,self.mesh.c_size])
        for p in range(self.param_size):
            print(p)

            idx = self.mesh.c_loc.reshape(self.mesh.c_size)==1
            a = self.a[p][idx,:][:,idx]
            a = a.toarray()
            a = csr_matrix(a)
            b = np.array(self.b[p][idx,:])
            
            u_ = spsolve(a, b)
            
            idx = np.array(idx, bool)
            self.label[p,idx] = u_
            if p==0:
                print(self.label[p,:].reshape(self.mesh.nx[0],self.mesh.nx[1]))
        
        param = self.param.reshape(self.param_size,1,self.mesh.nx[0],self.mesh.nx[1])
        label = self.label.reshape(self.param_size,1,self.mesh.nx[0],self.mesh.nx[1])
        mask = idx.reshape(1,1,self.mesh.nx[0],self.mesh.nx[1])
        
        self.param = torch.as_tensor(param)
        self.label = torch.as_tensor(label)
        self.mask = torch.as_tensor(mask)
    
    def load_param(self):
        self.norm_a = torch.load(f'./cache/{self.dataset_type}/norm_a.pt')
        self.param = torch.load(f'./cache/{self.dataset_type}/param.pt')
        self.param_size = self.param.shape[0]
        print('parameter shape:')
        print(self.param.shape)
        print(f"parameter mean: {self.param.mean():.5e}, std: {self.param.std():.5e}")
    
    def save_param(self):
        os.makedirs(f'./cache/{self.dataset_type}', exist_ok=True)
        torch.save(self.param, f'./cache/{self.dataset_type}/norm_a.pt')
        torch.save(self.param, f'./cache/{self.dataset_type}/param.pt')

    def load_loss_weight(self):
        self.r = torch.load(f'./cache/{self.dataset_type}/r.pt')
        self.wei_u = torch.load(f'./cache/{self.dataset_type}/wei_u.pt')

    def save_loss_weight(self):
        os.makedirs(f'./cache/{self.dataset_type}', exist_ok=True)
        torch.save(self.r, f'./cache/{self.dataset_type}/r.pt')
        torch.save(self.wei_u, f'./cache/{self.dataset_type}/wei_u.pt')
    
    def load_label(self):
        self.label = torch.load(f'./cache/{self.dataset_type}/label.pt')
        self.mask = torch.load(f'./cache/{self.dataset_type}/mask.pt')
    
    def save_label(self):
        os.makedirs(f'./cache/{self.dataset_type}', exist_ok=True)
        torch.save(self.label, f'./cache/{self.dataset_type}/label.pt')
        torch.save(self.mask, f'./cache/{self.dataset_type}/mask.pt')
    
    def load_dataset(self):
        data = h5py.File(f'./dataset/{self.dataset_type}_{self.mesh.nx[0]}.hdf5', "r")
        self.param = torch.as_tensor(data['param'])
        self.label = torch.as_tensor(data['label'])
        self.mask = torch.as_tensor(data['mask'])
    
    def save_dataset(self):
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

def get_dataloader(dataset, batch_size: int, shuffle: bool=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

class TeSet():
    def __init__(self, file_name, param_size, nx, dtype):
        self.param_size = param_size
        self.nx = nx
        self.dtype = dtype
        
        data = pd.read_csv(file_name, header=None)
        data = np.array(data)
        data = torch.tensor(data, dtype=self.dtype)
        
        self.x0 = data[:,0:1].reshape(self.param_size,1,self.nx[0],self.nx[1])
        self.x1 = data[:,1:2].reshape(self.param_size,1,self.nx[0],self.nx[1])
        self.param = data[:,2:3].reshape(self.param_size,1,self.nx[0],self.nx[1])
        self.label = data[:,3:4].reshape(self.param_size,1,self.nx[0],self.nx[1])
        self.mask = data[:,4:5].reshape(self.param_size,1,self.nx[0],self.nx[1])

    def to(self, device):
        self.device = device

        self.x0 = self.x0.to(self.device)
        self.x1 = self.x1.to(self.device)
        self.param = self.param.to(self.device)
        self.label = self.label.to(self.device)
        self.mask = self.mask.to(self.device)