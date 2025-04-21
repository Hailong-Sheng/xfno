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

class Dataset():
    """ dataset """
    def __init__(self, geo, mesh, param_size, load_cache):
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
        self.load_cache = load_cache

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
        self.dist = self.geo.distance(self.coord)
        self.coord = self.coord.transpose(0,3,1,2)
        self.dist = self.dist.transpose(0,3,1,2)

        self.mask = (self.mesh.c_loc==1).reshape(1,1,self.mesh.nx[0],self.mesh.nx[1])

        self.schar = np.tanh(5*np.abs(self.dist)) * (self.mask-0.5) + 0.5
        
    def generate_label(self):
        print('Generating label ...')

        # boundary value on the interpolation node (if node is located on the boundary)
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
                        self.intp_func.v[p,m,n] = 0.0

        # boundary value
        self.bd_val = np.zeros([self.param_size,self.mesh.c_size,self.intp_func.n_size]) #########

        # coefficient matrix
        self.ma = np.zeros([self.param_size,self.mesh.c_size,self.intp_func.n_size]) ############
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
                    c, c_x0, c_x1 = self.intp_func.intp_coef(intp_x, self.mesh.fw_x[m,:])
                
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
                    c, c_x0, c_x1 = self.intp_func.intp_coef(intp_x, self.mesh.fe_x[m,:])
                
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
                    c, c_x0, c_x1 = self.intp_func.intp_coef(intp_x, self.mesh.fs_x[m,:])
                
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
                    c, c_x0, c_x1 = self.intp_func.intp_coef(intp_x, self.mesh.fn_x[m,:])
                
                ca = np.zeros(self.param_size)
                for n in range(self.intp_func.n_size):
                    ca += c[n] * intp_a[:,n]
                for n in range(self.intp_func.n_size):
                    diff = -ca * (c_x0[n]*self.mesh.fn_n[m,0] + c_x1[n]*self.mesh.fn_n[m,1])
                    self.ma[:,m,n] += diff * self.mesh.fn_l[m]
                
                # west south face
                tol = 1e-6
                if self.mesh.fws_l[m]>tol:
                    c, c_x0, c_x1 = self.intp_func.intp_coef(intp_x, self.mesh.fws_x[m,:])
                    
                    ca = np.zeros(self.param_size)
                    for n in range(self.intp_func.n_size):
                        ca += c[n] * intp_a[:,n]
                    for n in range(self.intp_func.n_size):
                        diff = -ca * (c_x0[n]*self.mesh.fws_n[m,0] + c_x1[n]*self.mesh.fws_n[m,1])
                        self.ma[:,m,n] += diff * self.mesh.fws_l[m]
                
                # west north face
                if self.mesh.fwn_l[m]>tol:
                    c, c_x0, c_x1 = self.intp_func.intp_coef(intp_x, self.mesh.fwn_x[m,:])
                    
                    ca = np.zeros(self.param_size)
                    for n in range(self.intp_func.n_size):
                        ca += c[n] * intp_a[:,n]
                    for n in range(self.intp_func.n_size):
                        diff = -ca * (c_x0[n]*self.mesh.fwn_n[m,0] + c_x1[n]*self.mesh.fwn_n[m,1])
                        self.ma[:,m,n] += diff * self.mesh.fwn_l[m]
                
                # east south face
                if self.mesh.fes_l[m]>tol:
                    c, c_x0, c_x1 = self.intp_func.intp_coef(intp_x, self.mesh.fes_x[m,:])
                    
                    ca = np.zeros(self.param_size)
                    for n in range(self.intp_func.n_size):
                        ca += c[n] * intp_a[:,n]
                    for n in range(self.intp_func.n_size):
                        diff = -ca * (c_x0[n]*self.mesh.fes_n[m,0] + c_x1[n]*self.mesh.fes_n[m,1])
                        self.ma[:,m,n] += diff * self.mesh.fes_l[m]
                
                # east north face
                if self.mesh.fen_l[m]>tol:
                    c, c_x0, c_x1 = self.intp_func.intp_coef(intp_x, self.mesh.fen_x[m,:])
                    
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
            print(f'for the {p}th data')
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
            f.create_dataset('dist', data=self.mask)
            f.create_dataset('schar', data=self.schar)
            f.create_dataset('mask', data=self.mask)
            f.create_dataset('bd_val', data=self.bd_val)
            f.create_dataset('ma', data=self.ma)
            f.create_dataset('vb', data=self.vb)
    
    def load_data(self):
        with h5py.File('./dataset/dataset.h5', 'r') as f:
            self.param = f['param'][:]
            self.label = f['label'][:]
            self.coord = f['coord'][:]
            self.dist = f['dist'][:]
            self.schar = f['schar'][:]
            self.mask = f['mask'][:]
            self.bd_val = f['bd_val'][:]
            self.ma = f['ma'][:]
            self.vb = f['vb'][:]

class DatasetFNO(torch.utils.data.Dataset):
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
        return self.param_size

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

class DatasetDAFNO(torch.utils.data.Dataset):
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
        self.schar = dataset.schar
        self.mask = dataset.mask
        self.dtype = dtype
        self.device = device
        
        self.param_size = self.param.shape[0]

        self.param = torch.tensor(self.param, dtype=self.dtype)
        self.label = torch.tensor(self.label, dtype=self.dtype)
        self.coord = torch.tensor(self.coord, dtype=self.dtype)
        self.schar = torch.tensor(self.schar, dtype=self.dtype)
        self.mask = torch.tensor(self.mask, dtype=self.dtype)

    def __getitem__(self, idx):
        data = {}
        data['param'] = self.param[idx].to(self.device)
        data['label'] = self.label[idx].to(self.device)
        data['coord'] = self.coord[0].to(self.device)
        data['schar'] = self.schar[0].to(self.device)
        data['mask'] = self.mask[0].to(self.device)
        return data

    def __len__(self):
        return self.param_size

class DatasetGeoFNO():
    def __init__(self, dataset, idx, mesh, dtype=torch.float32, device='cpu'):
        self.mesh_non = mesh
        self.mesh_car = dataset.mesh
        self.intp_func = dataset.intp_func
        self.dtype = dtype
        self.device = device
        
        idx_n_in_c = self.mesh_non.get_idx_n_in_c(self.mesh_car)
        self.param, self.label = self.intp_c2n(dataset.param[idx,:,:,:], dataset.label[idx,:,:,:], idx_n_in_c)

        self.param_size = self.param.shape[0]
        self.dim = 2
        
        self.coord = self.mesh_non.cen_x.transpose()
        self.coord = self.coord.reshape(1,self.dim,self.mesh_non.nx[0],self.mesh_non.nx[1])
        self.mask = (self.mesh_non.cen_loc==1).reshape(1,1,self.mesh_non.nx[0],self.mesh_non.nx[1])
        
        self.param = torch.tensor(self.param, dtype=self.dtype)
        self.label = torch.tensor(self.label, dtype=self.dtype)
        self.coord = torch.tensor(self.coord, dtype=self.dtype)
        self.mask = torch.tensor(self.mask, dtype=self.dtype)

    def intp_c2n(self, param_car, label_car, idx_n_in_c):
        param_size = param_car.shape[0]
        param = np.zeros([param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1]])
        label = np.zeros([param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1]])
        a = np.zeros([param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1]])
        ax0 = np.zeros([param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1]])
        ax1 = np.zeros([param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1]])
        for p in range(param_size):
            for i in range(self.mesh_non.nx[0]):
                for j in range(self.mesh_non.nx[1]):
                    m = i*self.mesh_non.nx[1] + j
                    if self.mesh_non.cen_loc[m]!=1:
                        continue
                    
                    mm = idx_n_in_c[m]
                    intp_x = self.intp_func.x[mm,:,:]
                    intp_i = self.intp_func.i[mm,:]

                    intp_p = np.zeros(self.intp_func.n_size)
                    intp_l = np.zeros(self.intp_func.n_size)
                    for n in range(self.intp_func.n_size):
                        if intp_i[n]!=-1:
                            ii = intp_i[n]//self.mesh_car.nx[1]
                            jj = intp_i[n]-ii*self.mesh_car.nx[1]
                            intp_p[n] = param_car[p,0,ii,jj]
                            intp_l[n] = label_car[p,0,ii,jj]
                        else:
                            ii = mm//self.mesh_car.nx[1]
                            jj = mm-ii*self.mesh_car.nx[1]
                            intp_p[n] = param_car[p,0,ii,jj]

                    c, c_x0, c_x1 = self.intp_func.intp_coef(intp_x, self.mesh_non.cen_x[m,:])
                    for n in range(self.intp_func.n_size):
                        param[p,0,i,j] += c[n] * intp_p[n]
                        label[p,0,i,j] += c[n] * intp_l[n]
                        a[p,0,i,j] += c[n] * intp_p[n]
                        ax0[p,0,i,j] += c_x0[n] * intp_p[n]
                        ax1[p,0,i,j] += c_x1[n] * intp_p[n]
        
        return param, label
    
    def __getitem__(self, idx):
        data = {}
        data['param'] = self.param[idx].to(self.device)
        data['label'] = self.label[idx].to(self.device)
        data['coord'] = self.coord[0].to(self.device)
        data['mask'] = self.mask[0].to(self.device)
        return data

    def __len__(self):
        return self.param_size

class DatasetGINO(PygDataset):
    def __init__(self, dataset, idx, mesh, dtype=torch.float32, device='cpu'):
        super(DatasetGINO, self).__init__()
        self.mesh_non = mesh
        self.mesh_car = dataset.mesh
        self.intp_func = dataset.intp_func
        self.dtype = dtype
        self.device = device

        idx_n_in_c = self.mesh_non.get_idx_n_in_c(self.mesh_car)
        self.param, self.label = self.intp_c2n(dataset.param[idx,:,:,:], dataset.label[idx,:,:,:], idx_n_in_c)

        self.param_size = self.param.shape[0]
        self.dim = 2
        
        self.coord = self.mesh_non.cen_x.transpose()
        self.coord = self.coord.reshape(1,self.dim,self.mesh_non.nx[0],self.mesh_non.nx[1])
        self.mask = (self.mesh_non.cen_loc==1).reshape(1,1,self.mesh_non.nx[0],self.mesh_non.nx[1])

        self.graph = []
        for p in range(self.param_size):
            param = np.zeros([self.mesh_non.cen_size+self.mesh_non.cen_size,1])
            param[:self.mesh_non.cen_size,:] = self.param[p].reshape(self.mesh_non.cen_size,1)
            label = self.label[p].reshape(self.mesh_non.cen_size,1)
            mask = self.mask.reshape(self.mesh_non.cen_size,1)

            node_pos = np.concatenate([self.mesh_non.cen_x, self.mesh_non.cen_y])
            edge_index = self.mesh_non.ball_connectivity(0.2,'encode')
            edge_attr = np.concatenate([node_pos[edge_index[0,:],:], node_pos[edge_index[1,:],:], 
                                        node_pos[edge_index[0,:],:]-node_pos[edge_index[1,:],:]], 1)

            param = torch.tensor(param, dtype=self.dtype)
            label = torch.tensor(label, dtype=self.dtype)
            mask = torch.tensor(mask, dtype=self.dtype)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=self.dtype)
            
            graph = PygData(x=param, y=label, node_pos=node_pos, mask=mask,
                            edge_index=edge_index, edge_attr=edge_attr)        
            self.graph.append(graph)

        self.param = torch.tensor(self.param, dtype=self.dtype)
        self.label = torch.tensor(self.label, dtype=self.dtype)
        self.coord = torch.tensor(self.coord, dtype=self.dtype)
        self.mask = torch.tensor(self.mask, dtype=self.dtype)

    def intp_c2n(self, param_car, label_car, idx_n_in_c):
        param_size = param_car.shape[0]
        param = np.zeros([param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1]])
        label = np.zeros([param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1]])
        a = np.zeros([param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1]])
        ax0 = np.zeros([param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1]])
        ax1 = np.zeros([param_size,1,self.mesh_non.nx[0],self.mesh_non.nx[1]])
        for p in range(param_size):
            for i in range(self.mesh_non.nx[0]):
                for j in range(self.mesh_non.nx[1]):
                    m = i*self.mesh_non.nx[1] + j
                    if self.mesh_non.cen_loc[m]!=1:
                        continue
                    
                    mm = idx_n_in_c[m]
                    intp_x = self.intp_func.x[mm,:,:]
                    intp_i = self.intp_func.i[mm,:]

                    intp_p = np.zeros(self.intp_func.n_size)
                    intp_l = np.zeros(self.intp_func.n_size)
                    for n in range(self.intp_func.n_size):
                        if intp_i[n]!=-1:
                            ii = intp_i[n]//self.mesh_car.nx[1]
                            jj = intp_i[n]-ii*self.mesh_car.nx[1]
                            intp_p[n] = param_car[p,0,ii,jj]
                            intp_l[n] = label_car[p,0,ii,jj]
                        else:
                            ii = mm//self.mesh_car.nx[1]
                            jj = mm-ii*self.mesh_car.nx[1]
                            intp_p[n] = param_car[p,0,ii,jj]

                    c, c_x0, c_x1 = self.intp_func.intp_coef(intp_x, self.mesh_non.cen_x[m,:])
                    for n in range(self.intp_func.n_size):
                        param[p,0,i,j] += c[n] * intp_p[n]
                        label[p,0,i,j] += c[n] * intp_l[n]
                        a[p,0,i,j] += c[n] * intp_p[n]
                        ax0[p,0,i,j] += c_x0[n] * intp_p[n]
                        ax1[p,0,i,j] += c_x1[n] * intp_p[n]
        
        return param, label
    
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