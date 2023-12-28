import torch
import numpy as np
import pandas as pd

import time

class TrSet():
    def __init__(self, geo, mesh, lm, mu, stress, dtype, load_intp_coef):
        self.geo = geo
        self.mesh = mesh
        self.lm = lm
        self.mu = mu
        self.stress = stress
        self.dtype = dtype

        self.bounds = self.mesh.bounds
        self.nx = self.mesh.nx
        self.dim = self.bounds.shape[0]
        
        self.space_size = self.nx[0]*self.nx[1]*self.nx[2]
        self.x = self.mesh.c_x
        self.p = self.mesh.c_v.reshape(1,1,self.nx[0],self.nx[1],self.nx[2])
        self.mask = (self.mesh.c_loc==1)

        self.mesh.fw_v0 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fw_v1 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fw_v2 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fe_v0 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fe_v1 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fe_v2 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fs_v0 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fs_v1 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fs_v2 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fn_v0 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fn_v1 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fn_v2 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fb_v0 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fb_v1 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fb_v2 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.ft_v0 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.ft_v1 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.ft_v2 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        for i in range(self.mesh.nx[0]):
            for j in range(self.mesh.nx[1]):
                for k in range(self.mesh.nx[2]):
                    if self.mesh.c_loc[i,j,k]!=1:
                        continue
                    
                    if self.mesh.fe_loc[i,j,k]==0 and self.mesh.fe_t[i,j,k]==2:
                        if i==self.mesh.nx[0]-1:
                            self.mesh.fe_v2[i,j,k] = self.stress

        self.mesh.fws_v0 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fws_v1 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fws_v2 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fwn_v0 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fwn_v1 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fwn_v2 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fes_v0 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fes_v1 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fes_v2 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fen_v0 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fen_v1 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.mesh.fen_v2 = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
                    
        tol = 1e-4
        self.intp_node_num = 3**3
        self.x_intp = torch.zeros(self.mesh.nx[0],self.mesh.nx[1],self.mesh.nx[2],self.intp_node_num,self.dim)
        self.n_intp = torch.zeros(self.mesh.nx[0],self.mesh.nx[1],self.mesh.nx[2],self.intp_node_num,self.dim)
        self.t_intp = torch.zeros(self.mesh.nx[0],self.mesh.nx[1],self.mesh.nx[2],self.intp_node_num)
        self.i_intp = torch.zeros(self.mesh.nx[0],self.mesh.nx[1],self.mesh.nx[2],self.intp_node_num).long()
        self.v_intp = torch.zeros(self.mesh.nx[0],self.mesh.nx[1],self.mesh.nx[2],3,self.intp_node_num)
        for i in range(self.mesh.nx[0]):
            for j in range(self.mesh.nx[1]):
                for k in range(self.mesh.nx[2]):
                    if mesh.c_loc[i,j,k]!=1:
                        continue
                    
                    self.x_intp[i,j,k,:,:], self.n_intp[i,j,k,:,:], self.t_intp[i,j,k,:], self.i_intp[i,j,k,:] = \
                        self.intp_node(geo, mesh, [i,j,k])
                    
                    xi = self.x_intp[i,j,k,:,:].reshape(self.intp_node_num,self.dim)
                    ni = self.n_intp[i,j,k,:,:].reshape(self.intp_node_num,self.dim)
                    ti = self.t_intp[i,j,k,:].reshape(self.intp_node_num,1)
                    for s in range(self.intp_node_num):
                        if ti[s]==2:
                            if abs(xi[s,0]-self.bounds[0,1])<tol:
                                self.v_intp[i,j,k,2,s] = self.stress
        
        self.v0 = torch.zeros(1,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.v1 = torch.zeros(1,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.v2 = torch.zeros(1,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    if self.mesh.c_loc[i,j,k]!=1:
                        continue

                    xi = self.x_intp[i,j,k,:,:]
                    ni = self.n_intp[i,j,k,:,:]
                    ti = self.t_intp[i,j,k,:]
                    ii = self.i_intp[i,j,k,:]
                    vi = self.v_intp[i,j,k,:]
                    
                    for s in range(self.intp_node_num):
                        self.v0[0,s,i,j,k] = vi[0,s] if ii[s]==-1 else 0
                        self.v1[0,s,i,j,k] = vi[1,s] if ii[s]==-1 else 0
                        self.v2[0,s,i,j,k] = vi[2,s] if ii[s]==-1 else 0

        self.c0_x0 = torch.zeros(6,3,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.c0_x1 = torch.zeros(6,3,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.c0_x2 = torch.zeros(6,3,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.c1_x0 = torch.zeros(6,3,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.c1_x1 = torch.zeros(6,3,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.c1_x2 = torch.zeros(6,3,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.c2_x0 = torch.zeros(6,3,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.c2_x1 = torch.zeros(6,3,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.c2_x2 = torch.zeros(6,3,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        if load_intp_coef:
            self.load_interpolation_coefficient()
        
        self.r0 = torch.zeros(1,1,self.nx[0],self.nx[1],self.nx[2])
        self.r1 = torch.zeros(1,1,self.nx[0],self.nx[1],self.nx[2])
        self.r2 = torch.zeros(1,1,self.nx[0],self.nx[1],self.nx[2])

        self.wei0_u0 = torch.zeros(1,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.wei0_u1 = torch.zeros(1,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.wei0_u2 = torch.zeros(1,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.wei1_u0 = torch.zeros(1,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.wei1_u1 = torch.zeros(1,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.wei1_u2 = torch.zeros(1,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.wei2_u0 = torch.zeros(1,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.wei2_u1 = torch.zeros(1,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])
        self.wei2_u2 = torch.zeros(1,self.intp_node_num,self.nx[0],self.nx[1],self.nx[2])

        """ interpolation coefficients for regular unit """
        self.re_c0_x0 = torch.zeros(6,3,self.intp_node_num)
        self.re_c1_x0 = torch.zeros(6,3,self.intp_node_num)
        self.re_c2_x0 = torch.zeros(6,3,self.intp_node_num)
        self.re_c0_x1 = torch.zeros(6,3,self.intp_node_num)
        self.re_c1_x1 = torch.zeros(6,3,self.intp_node_num)
        self.re_c2_x1 = torch.zeros(6,3,self.intp_node_num)
        self.re_c0_x2 = torch.zeros(6,3,self.intp_node_num)
        self.re_c1_x2 = torch.zeros(6,3,self.intp_node_num)
        self.re_c2_x2 = torch.zeros(6,3,self.intp_node_num)
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    if self.mesh.c_loc[i,j,k]!=1:
                        continue
                    
                    m = (i*self.nx[1] + j)*self.nx[2] + k
                    xi = self.x_intp[i,j,k,:,:]
                    ni = self.n_intp[i,j,k,:,:]
                    ti = self.t_intp[i,j,k,:]
                    ii = self.i_intp[i,j,k,:]
                    vi = self.v_intp[i,j,k,:]

                    if (ii!=-1).all():
                        self.re_c0_x0[0,:,:], self.re_c1_x0[0,:,:], self.re_c2_x0[0,:,:], \
                        self.re_c0_x1[0,:,:], self.re_c1_x1[0,:,:], self.re_c2_x1[0,:,:], \
                        self.re_c0_x2[0,:,:], self.re_c1_x2[0,:,:], self.re_c2_x2[0,:,:] = \
                            self.intp_coef(xi, ni, self.mesh.fw_x[i,j,k,:])
                        
                        self.re_c0_x0[1,:,:], self.re_c1_x0[1,:,:], self.re_c2_x0[1,:,:], \
                        self.re_c0_x1[1,:,:], self.re_c1_x1[1,:,:], self.re_c2_x1[1,:,:], \
                        self.re_c0_x2[1,:,:], self.re_c1_x2[1,:,:], self.re_c2_x2[1,:,:] = \
                            self.intp_coef(xi, ni, self.mesh.fe_x[i,j,k,:])
                        
                        self.re_c0_x0[2,:,:], self.re_c1_x0[2,:,:], self.re_c2_x0[2,:,:], \
                        self.re_c0_x1[2,:,:], self.re_c1_x1[2,:,:], self.re_c2_x1[2,:,:], \
                        self.re_c0_x2[2,:,:], self.re_c1_x2[2,:,:], self.re_c2_x2[2,:,:] = \
                            self.intp_coef(xi, ni, self.mesh.fs_x[i,j,k,:])
                        
                        self.re_c0_x0[3,:,:], self.re_c1_x0[3,:,:], self.re_c2_x0[3,:,:], \
                        self.re_c0_x1[3,:,:], self.re_c1_x1[3,:,:], self.re_c2_x1[3,:,:], \
                        self.re_c0_x2[3,:,:], self.re_c1_x2[3,:,:], self.re_c2_x2[3,:,:] = \
                            self.intp_coef(xi, ni, self.mesh.fn_x[i,j,k,:])
                        
                        self.re_c0_x0[4,:,:], self.re_c1_x0[4,:,:], self.re_c2_x0[4,:,:], \
                        self.re_c0_x1[4,:,:], self.re_c1_x1[4,:,:], self.re_c2_x1[4,:,:], \
                        self.re_c0_x2[4,:,:], self.re_c1_x2[4,:,:], self.re_c2_x2[4,:,:] = \
                            self.intp_coef(xi, ni, self.mesh.fb_x[i,j,k,:])
                        
                        self.re_c0_x0[5,:,:], self.re_c1_x0[5,:,:], self.re_c2_x0[5,:,:], \
                        self.re_c0_x1[5,:,:], self.re_c1_x1[5,:,:], self.re_c2_x1[5,:,:], \
                        self.re_c0_x2[5,:,:], self.re_c1_x2[5,:,:], self.re_c2_x2[5,:,:] = \
                            self.intp_coef(xi, ni, self.mesh.ft_x[i,j,k,:])
        
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    if self.mesh.c_loc[i,j,k]!=1:
                        continue

                    if self.mesh.fw_t[i,j,k]==2:
                        self.r0[0,0,i,j,k] += self.mesh.fw_v0[i,j,k] * self.mesh.fw_a[i,j,k]
                        self.r1[0,0,i,j,k] += self.mesh.fw_v1[i,j,k] * self.mesh.fw_a[i,j,k]
                        self.r2[0,0,i,j,k] += self.mesh.fw_v2[i,j,k] * self.mesh.fw_a[i,j,k]
                        continue
                    
                    m = (i*self.nx[1] + j)*self.nx[2] + k
                    xi = self.x_intp[i,j,k,:,:]
                    ni = self.n_intp[i,j,k,:,:]
                    ti = self.t_intp[i,j,k,:]
                    ii = self.i_intp[i,j,k,:]
                    vi = self.v_intp[i,j,k,:]
                    
                    if load_intp_coef:
                        c0_x0 = self.c0_x0[0,:,:,i,j,k]
                        c1_x0 = self.c1_x0[0,:,:,i,j,k]
                        c2_x0 = self.c2_x0[0,:,:,i,j,k]
                        c0_x1 = self.c0_x1[0,:,:,i,j,k]
                        c1_x1 = self.c1_x1[0,:,:,i,j,k]
                        c2_x1 = self.c2_x1[0,:,:,i,j,k]
                        c0_x2 = self.c0_x2[0,:,:,i,j,k]
                        c1_x2 = self.c1_x2[0,:,:,i,j,k]
                        c2_x2 = self.c2_x2[0,:,:,i,j,k]
                    else:
                        print(m)
                        if (ii!=-1).all():
                            c0_x0, c1_x0, c2_x0 = self.re_c0_x0[0,:,:], self.re_c1_x0[0,:,:], self.re_c2_x0[0,:,:]
                            c0_x1, c1_x1, c2_x1 = self.re_c0_x1[0,:,:], self.re_c1_x1[0,:,:], self.re_c2_x1[0,:,:]
                            c0_x2, c1_x2, c2_x2 = self.re_c0_x2[0,:,:], self.re_c1_x2[0,:,:], self.re_c2_x2[0,:,:]
                        else:
                            c0_x0, c1_x0, c2_x0, c0_x1, c1_x1, c2_x1, c0_x2, c1_x2, c2_x2 = \
                                self.intp_coef(xi, ni, self.mesh.fw_x[i,j,k,:])
                        self.c0_x0[0,:,:,i,j,k] = c0_x0
                        self.c1_x0[0,:,:,i,j,k] = c1_x0
                        self.c2_x0[0,:,:,i,j,k] = c2_x0
                        self.c0_x1[0,:,:,i,j,k] = c0_x1
                        self.c1_x1[0,:,:,i,j,k] = c1_x1
                        self.c2_x1[0,:,:,i,j,k] = c2_x1
                        self.c0_x2[0,:,:,i,j,k] = c0_x2
                        self.c1_x2[0,:,:,i,j,k] = c1_x2
                        self.c2_x2[0,:,:,i,j,k] = c2_x2

                    for s in range(self.intp_node_num):
                        self.wei0_u0[0,s,i,j,k] += ((self.lm+2*self.mu)*c0_x0[0,s] + self.lm*c1_x1[0,s] + self.lm*c2_x2[0,s]) * self.mesh.fw_a[i,j,k]
                        self.wei0_u1[0,s,i,j,k] += ((self.lm+2*self.mu)*c0_x0[1,s] + self.lm*c1_x1[1,s] + self.lm*c2_x2[1,s]) * self.mesh.fw_a[i,j,k]
                        self.wei0_u2[0,s,i,j,k] += ((self.lm+2*self.mu)*c0_x0[2,s] + self.lm*c1_x1[2,s] + self.lm*c2_x2[2,s]) * self.mesh.fw_a[i,j,k]
                        self.wei1_u0[0,s,i,j,k] += (self.mu*c1_x0[0,s] + self.mu*c0_x1[0,s]) * self.mesh.fw_a[i,j,k]
                        self.wei1_u1[0,s,i,j,k] += (self.mu*c1_x0[1,s] + self.mu*c0_x1[1,s]) * self.mesh.fw_a[i,j,k]
                        self.wei1_u2[0,s,i,j,k] += (self.mu*c1_x0[2,s] + self.mu*c0_x1[2,s]) * self.mesh.fw_a[i,j,k]
                        self.wei2_u0[0,s,i,j,k] += (self.mu*c2_x0[0,s] + self.mu*c0_x2[0,s]) * self.mesh.fw_a[i,j,k]
                        self.wei2_u1[0,s,i,j,k] += (self.mu*c2_x0[1,s] + self.mu*c0_x2[1,s]) * self.mesh.fw_a[i,j,k]
                        self.wei2_u2[0,s,i,j,k] += (self.mu*c2_x0[2,s] + self.mu*c0_x2[2,s]) * self.mesh.fw_a[i,j,k]
        
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    if self.mesh.c_loc[i,j,k]!=1:
                        continue

                    if self.mesh.fe_t[i,j,k]==2:
                        self.r0[0,0,i,j,k] += self.mesh.fe_v0[i,j,k] * self.mesh.fe_a[i,j,k]
                        self.r1[0,0,i,j,k] += self.mesh.fe_v1[i,j,k] * self.mesh.fe_a[i,j,k]
                        self.r2[0,0,i,j,k] += self.mesh.fe_v2[i,j,k] * self.mesh.fe_a[i,j,k]
                        continue
                    
                    m = (i*self.nx[1] + j)*self.nx[2] + k
                    xi = self.x_intp[i,j,k,:,:]
                    ni = self.n_intp[i,j,k,:,:]
                    ti = self.t_intp[i,j,k,:]
                    ii = self.i_intp[i,j,k,:]
                    vi = self.v_intp[i,j,k,:]
                    
                    if load_intp_coef:
                        c0_x0 = self.c0_x0[1,:,:,i,j,k]
                        c1_x0 = self.c1_x0[1,:,:,i,j,k]
                        c2_x0 = self.c2_x0[1,:,:,i,j,k]
                        c0_x1 = self.c0_x1[1,:,:,i,j,k]
                        c1_x1 = self.c1_x1[1,:,:,i,j,k]
                        c2_x1 = self.c2_x1[1,:,:,i,j,k]
                        c0_x2 = self.c0_x2[1,:,:,i,j,k]
                        c1_x2 = self.c1_x2[1,:,:,i,j,k]
                        c2_x2 = self.c2_x2[1,:,:,i,j,k]
                    else:
                        print(m)
                        if (ii!=-1).all():
                            c0_x0, c1_x0, c2_x0 = self.re_c0_x0[1,:,:], self.re_c1_x0[1,:,:], self.re_c2_x0[1,:,:]
                            c0_x1, c1_x1, c2_x1 = self.re_c0_x1[1,:,:], self.re_c1_x1[1,:,:], self.re_c2_x1[1,:,:]
                            c0_x2, c1_x2, c2_x2 = self.re_c0_x2[1,:,:], self.re_c1_x2[1,:,:], self.re_c2_x2[1,:,:]
                        else:
                            c0_x0, c1_x0, c2_x0, c0_x1, c1_x1, c2_x1, c0_x2, c1_x2, c2_x2 = \
                                self.intp_coef(xi, ni, self.mesh.fe_x[i,j,k,:])
                        self.c0_x0[1,:,:,i,j,k] = c0_x0
                        self.c1_x0[1,:,:,i,j,k] = c1_x0
                        self.c2_x0[1,:,:,i,j,k] = c2_x0
                        self.c0_x1[1,:,:,i,j,k] = c0_x1
                        self.c1_x1[1,:,:,i,j,k] = c1_x1
                        self.c2_x1[1,:,:,i,j,k] = c2_x1
                        self.c0_x2[1,:,:,i,j,k] = c0_x2
                        self.c1_x2[1,:,:,i,j,k] = c1_x2
                        self.c2_x2[1,:,:,i,j,k] = c2_x2

                    for s in range(self.intp_node_num):
                        self.wei0_u0[0,s,i,j,k] -= ((self.lm+2*self.mu)*c0_x0[0,s] + self.lm*c1_x1[0,s] + self.lm*c2_x2[0,s]) * self.mesh.fe_a[i,j,k]
                        self.wei0_u1[0,s,i,j,k] -= ((self.lm+2*self.mu)*c0_x0[1,s] + self.lm*c1_x1[1,s] + self.lm*c2_x2[1,s]) * self.mesh.fe_a[i,j,k]
                        self.wei0_u2[0,s,i,j,k] -= ((self.lm+2*self.mu)*c0_x0[2,s] + self.lm*c1_x1[2,s] + self.lm*c2_x2[2,s]) * self.mesh.fe_a[i,j,k]
                        self.wei1_u0[0,s,i,j,k] -= (self.mu*c1_x0[0,s] + self.mu*c0_x1[0,s]) * self.mesh.fe_a[i,j,k]
                        self.wei1_u1[0,s,i,j,k] -= (self.mu*c1_x0[1,s] + self.mu*c0_x1[1,s]) * self.mesh.fe_a[i,j,k]
                        self.wei1_u2[0,s,i,j,k] -= (self.mu*c1_x0[2,s] + self.mu*c0_x1[2,s]) * self.mesh.fe_a[i,j,k]
                        self.wei2_u0[0,s,i,j,k] -= (self.mu*c2_x0[0,s] + self.mu*c0_x2[0,s]) * self.mesh.fe_a[i,j,k]
                        self.wei2_u1[0,s,i,j,k] -= (self.mu*c2_x0[1,s] + self.mu*c0_x2[1,s]) * self.mesh.fe_a[i,j,k]
                        self.wei2_u2[0,s,i,j,k] -= (self.mu*c2_x0[2,s] + self.mu*c0_x2[2,s]) * self.mesh.fe_a[i,j,k]

        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    if self.mesh.c_loc[i,j,k]!=1:
                        continue

                    if self.mesh.fs_t[i,j,k]==2:
                        self.r0[0,0,i,j,k] += self.mesh.fs_v0[i,j,k] * self.mesh.fs_a[i,j,k]
                        self.r1[0,0,i,j,k] += self.mesh.fs_v1[i,j,k] * self.mesh.fs_a[i,j,k]
                        self.r2[0,0,i,j,k] += self.mesh.fs_v2[i,j,k] * self.mesh.fs_a[i,j,k]
                        continue
                    
                    m = (i*self.nx[1] + j)*self.nx[2] + k
                    xi = self.x_intp[i,j,k,:,:]
                    ni = self.n_intp[i,j,k,:,:]
                    ti = self.t_intp[i,j,k,:]
                    ii = self.i_intp[i,j,k,:]
                    vi = self.v_intp[i,j,k,:]
                    
                    if load_intp_coef:
                        c0_x0 = self.c0_x0[2,:,:,i,j,k]
                        c1_x0 = self.c1_x0[2,:,:,i,j,k]
                        c2_x0 = self.c2_x0[2,:,:,i,j,k]
                        c0_x1 = self.c0_x1[2,:,:,i,j,k]
                        c1_x1 = self.c1_x1[2,:,:,i,j,k]
                        c2_x1 = self.c2_x1[2,:,:,i,j,k]
                        c0_x2 = self.c0_x2[2,:,:,i,j,k]
                        c1_x2 = self.c1_x2[2,:,:,i,j,k]
                        c2_x2 = self.c2_x2[2,:,:,i,j,k]
                    else:
                        print(m)
                        if (ii!=-1).all():
                            c0_x0, c1_x0, c2_x0 = self.re_c0_x0[2,:,:], self.re_c1_x0[2,:,:], self.re_c2_x0[2,:,:]
                            c0_x1, c1_x1, c2_x1 = self.re_c0_x1[2,:,:], self.re_c1_x1[2,:,:], self.re_c2_x1[2,:,:]
                            c0_x2, c1_x2, c2_x2 = self.re_c0_x2[2,:,:], self.re_c1_x2[2,:,:], self.re_c2_x2[2,:,:]
                        else:
                            c0_x0, c1_x0, c2_x0, c0_x1, c1_x1, c2_x1, c0_x2, c1_x2, c2_x2 = \
                                self.intp_coef(xi, ni, self.mesh.fs_x[i,j,k,:])    
                        self.c0_x0[2,:,:,i,j,k] = c0_x0
                        self.c1_x0[2,:,:,i,j,k] = c1_x0
                        self.c2_x0[2,:,:,i,j,k] = c2_x0
                        self.c0_x1[2,:,:,i,j,k] = c0_x1
                        self.c1_x1[2,:,:,i,j,k] = c1_x1
                        self.c2_x1[2,:,:,i,j,k] = c2_x1
                        self.c0_x2[2,:,:,i,j,k] = c0_x2
                        self.c1_x2[2,:,:,i,j,k] = c1_x2
                        self.c2_x2[2,:,:,i,j,k] = c2_x2

                    for s in range(self.intp_node_num):
                        self.wei0_u0[0,s,i,j,k] += (self.mu*c0_x1[0,s] + self.mu*c1_x0[0,s]) * self.mesh.fs_a[i,j,k]
                        self.wei0_u1[0,s,i,j,k] += (self.mu*c0_x1[1,s] + self.mu*c1_x0[1,s]) * self.mesh.fs_a[i,j,k]
                        self.wei0_u2[0,s,i,j,k] += (self.mu*c0_x1[2,s] + self.mu*c1_x0[2,s]) * self.mesh.fs_a[i,j,k]
                        self.wei1_u0[0,s,i,j,k] += (self.lm*c0_x0[0,s] + (self.lm+2*self.mu)*c1_x1[0,s] + self.lm*c2_x2[0,s]) * self.mesh.fs_a[i,j,k]
                        self.wei1_u1[0,s,i,j,k] += (self.lm*c0_x0[1,s] + (self.lm+2*self.mu)*c1_x1[1,s] + self.lm*c2_x2[1,s]) * self.mesh.fs_a[i,j,k]
                        self.wei1_u2[0,s,i,j,k] += (self.lm*c0_x0[2,s] + (self.lm+2*self.mu)*c1_x1[2,s] + self.lm*c2_x2[2,s]) * self.mesh.fs_a[i,j,k]
                        self.wei2_u0[0,s,i,j,k] += (self.mu*c2_x1[0,s] + self.mu*c1_x2[0,s]) * self.mesh.fs_a[i,j,k]
                        self.wei2_u1[0,s,i,j,k] += (self.mu*c2_x1[1,s] + self.mu*c1_x2[1,s]) * self.mesh.fs_a[i,j,k]
                        self.wei2_u2[0,s,i,j,k] += (self.mu*c2_x1[2,s] + self.mu*c1_x2[2,s]) * self.mesh.fs_a[i,j,k]

        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    if self.mesh.c_loc[i,j,k]!=1:
                        continue
                    
                    if self.mesh.fn_t[i,j,k]==2:
                        self.r0[0,0,i,j,k] += self.mesh.fn_v0[i,j,k] * self.mesh.fn_a[i,j,k]
                        self.r1[0,0,i,j,k] += self.mesh.fn_v1[i,j,k] * self.mesh.fn_a[i,j,k]
                        self.r2[0,0,i,j,k] += self.mesh.fn_v2[i,j,k] * self.mesh.fn_a[i,j,k]
                        continue
                    
                    m = (i*self.nx[1] + j)*self.nx[2] + k
                    xi = self.x_intp[i,j,k,:,:]
                    ni = self.n_intp[i,j,k,:,:]
                    ti = self.t_intp[i,j,k,:]
                    ii = self.i_intp[i,j,k,:]
                    vi = self.v_intp[i,j,k,:]
                    
                    if load_intp_coef:
                        c0_x0 = self.c0_x0[3,:,:,i,j,k]
                        c1_x0 = self.c1_x0[3,:,:,i,j,k]
                        c2_x0 = self.c2_x0[3,:,:,i,j,k]
                        c0_x1 = self.c0_x1[3,:,:,i,j,k]
                        c1_x1 = self.c1_x1[3,:,:,i,j,k]
                        c2_x1 = self.c2_x1[3,:,:,i,j,k]
                        c0_x2 = self.c0_x2[3,:,:,i,j,k]
                        c1_x2 = self.c1_x2[3,:,:,i,j,k]
                        c2_x2 = self.c2_x2[3,:,:,i,j,k]
                    else:
                        print(m)
                        if (ii!=-1).all():
                            c0_x0, c1_x0, c2_x0 = self.re_c0_x0[3,:,:], self.re_c1_x0[3,:,:], self.re_c2_x0[3,:,:]
                            c0_x1, c1_x1, c2_x1 = self.re_c0_x1[3,:,:], self.re_c1_x1[3,:,:], self.re_c2_x1[3,:,:]
                            c0_x2, c1_x2, c2_x2 = self.re_c0_x2[3,:,:], self.re_c1_x2[3,:,:], self.re_c2_x2[3,:,:]
                        else:
                            c0_x0, c1_x0, c2_x0, c0_x1, c1_x1, c2_x1, c0_x2, c1_x2, c2_x2 = \
                                self.intp_coef(xi, ni, self.mesh.fn_x[i,j,k,:])
                        self.c0_x0[3,:,:,i,j,k] = c0_x0
                        self.c1_x0[3,:,:,i,j,k] = c1_x0
                        self.c2_x0[3,:,:,i,j,k] = c2_x0
                        self.c0_x1[3,:,:,i,j,k] = c0_x1
                        self.c1_x1[3,:,:,i,j,k] = c1_x1
                        self.c2_x1[3,:,:,i,j,k] = c2_x1
                        self.c0_x2[3,:,:,i,j,k] = c0_x2
                        self.c1_x2[3,:,:,i,j,k] = c1_x2
                        self.c2_x2[3,:,:,i,j,k] = c2_x2
                    
                    for s in range(self.intp_node_num):
                        self.wei0_u0[0,s,i,j,k] -= (self.mu*c0_x1[0,s] + self.mu*c1_x0[0,s]) * self.mesh.fn_a[i,j,k]
                        self.wei0_u1[0,s,i,j,k] -= (self.mu*c0_x1[1,s] + self.mu*c1_x0[1,s]) * self.mesh.fn_a[i,j,k]
                        self.wei0_u2[0,s,i,j,k] -= (self.mu*c0_x1[2,s] + self.mu*c1_x0[2,s]) * self.mesh.fn_a[i,j,k]
                        self.wei1_u0[0,s,i,j,k] -= (self.lm*c0_x0[0,s] + (self.lm+2*self.mu)*c1_x1[0,s] + self.lm*c2_x2[0,s]) * self.mesh.fn_a[i,j,k]
                        self.wei1_u1[0,s,i,j,k] -= (self.lm*c0_x0[1,s] + (self.lm+2*self.mu)*c1_x1[1,s] + self.lm*c2_x2[1,s]) * self.mesh.fn_a[i,j,k]
                        self.wei1_u2[0,s,i,j,k] -= (self.lm*c0_x0[2,s] + (self.lm+2*self.mu)*c1_x1[2,s] + self.lm*c2_x2[2,s]) * self.mesh.fn_a[i,j,k]
                        self.wei2_u0[0,s,i,j,k] -= (self.mu*c2_x1[0,s] + self.mu*c1_x2[0,s]) * self.mesh.fn_a[i,j,k]
                        self.wei2_u1[0,s,i,j,k] -= (self.mu*c2_x1[1,s] + self.mu*c1_x2[1,s]) * self.mesh.fn_a[i,j,k]
                        self.wei2_u2[0,s,i,j,k] -= (self.mu*c2_x1[2,s] + self.mu*c1_x2[2,s]) * self.mesh.fn_a[i,j,k]

        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    if self.mesh.c_loc[i,j,k]!=1:
                        continue
                    
                    if self.mesh.fb_t[i,j,k]==2:
                        self.r0[0,0,i,j,k] += self.mesh.fb_v0[i,j,k] * self.mesh.fb_a[i,j,k]
                        self.r1[0,0,i,j,k] += self.mesh.fb_v1[i,j,k] * self.mesh.fb_a[i,j,k]
                        self.r2[0,0,i,j,k] += self.mesh.fb_v2[i,j,k] * self.mesh.fb_a[i,j,k]
                        continue
                    
                    m = (i*self.nx[1] + j)*self.nx[2] + k
                    xi = self.x_intp[i,j,k,:,:]
                    ni = self.n_intp[i,j,k,:,:]
                    ti = self.t_intp[i,j,k,:]
                    ii = self.i_intp[i,j,k,:]
                    vi = self.v_intp[i,j,k,:]
                    
                    if load_intp_coef:
                        c0_x0 = self.c0_x0[4,:,:,i,j,k]
                        c1_x0 = self.c1_x0[4,:,:,i,j,k]
                        c2_x0 = self.c2_x0[4,:,:,i,j,k]
                        c0_x1 = self.c0_x1[4,:,:,i,j,k]
                        c1_x1 = self.c1_x1[4,:,:,i,j,k]
                        c2_x1 = self.c2_x1[4,:,:,i,j,k]
                        c0_x2 = self.c0_x2[4,:,:,i,j,k]
                        c1_x2 = self.c1_x2[4,:,:,i,j,k]
                        c2_x2 = self.c2_x2[4,:,:,i,j,k]
                    else:
                        print(m)
                        if (ii!=-1).all():
                            c0_x0, c1_x0, c2_x0 = self.re_c0_x0[4,:,:], self.re_c1_x0[4,:,:], self.re_c2_x0[4,:,:]
                            c0_x1, c1_x1, c2_x1 = self.re_c0_x1[4,:,:], self.re_c1_x1[4,:,:], self.re_c2_x1[4,:,:]
                            c0_x2, c1_x2, c2_x2 = self.re_c0_x2[4,:,:], self.re_c1_x2[4,:,:], self.re_c2_x2[4,:,:]
                        else:
                            c0_x0, c1_x0, c2_x0, c0_x1, c1_x1, c2_x1, c0_x2, c1_x2, c2_x2 = \
                                self.intp_coef(xi, ni, self.mesh.fb_x[i,j,k,:])
                        self.c0_x0[4,:,:,i,j,k] = c0_x0
                        self.c1_x0[4,:,:,i,j,k] = c1_x0
                        self.c2_x0[4,:,:,i,j,k] = c2_x0
                        self.c0_x1[4,:,:,i,j,k] = c0_x1
                        self.c1_x1[4,:,:,i,j,k] = c1_x1
                        self.c2_x1[4,:,:,i,j,k] = c2_x1
                        self.c0_x2[4,:,:,i,j,k] = c0_x2
                        self.c1_x2[4,:,:,i,j,k] = c1_x2
                        self.c2_x2[4,:,:,i,j,k] = c2_x2

                    for s in range(self.intp_node_num):
                        self.wei0_u0[0,s,i,j,k] += (self.mu*c0_x2[0,s] + self.mu*c2_x0[0,s]) * self.mesh.fb_a[i,j,k]
                        self.wei0_u1[0,s,i,j,k] += (self.mu*c0_x2[1,s] + self.mu*c2_x0[1,s]) * self.mesh.fb_a[i,j,k]
                        self.wei0_u2[0,s,i,j,k] += (self.mu*c0_x2[2,s] + self.mu*c2_x0[2,s]) * self.mesh.fb_a[i,j,k]
                        self.wei1_u0[0,s,i,j,k] += (self.mu*c1_x2[0,s] + self.mu*c2_x1[0,s]) * self.mesh.fb_a[i,j,k]
                        self.wei1_u1[0,s,i,j,k] += (self.mu*c1_x2[1,s] + self.mu*c2_x1[1,s]) * self.mesh.fb_a[i,j,k]
                        self.wei1_u2[0,s,i,j,k] += (self.mu*c1_x2[2,s] + self.mu*c2_x1[2,s]) * self.mesh.fb_a[i,j,k]
                        self.wei2_u0[0,s,i,j,k] += (self.lm*c0_x0[0,s] + self.lm*c1_x1[0,s] + (self.lm+2*self.mu)*c2_x2[0,s]) * self.mesh.fb_a[i,j,k]
                        self.wei2_u1[0,s,i,j,k] += (self.lm*c0_x0[1,s] + self.lm*c1_x1[1,s] + (self.lm+2*self.mu)*c2_x2[1,s]) * self.mesh.fb_a[i,j,k]
                        self.wei2_u2[0,s,i,j,k] += (self.lm*c0_x0[2,s] + self.lm*c1_x1[2,s] + (self.lm+2*self.mu)*c2_x2[2,s]) * self.mesh.fb_a[i,j,k]

        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    if self.mesh.c_loc[i,j,k]!=1:
                        continue
                    
                    if self.mesh.ft_t[i,j,k]==2:
                        self.r0[0,0,i,j,k] += self.mesh.ft_v0[i,j,k] * self.mesh.ft_a[i,j,k]
                        self.r1[0,0,i,j,k] += self.mesh.ft_v0[i,j,k] * self.mesh.ft_a[i,j,k]
                        self.r2[0,0,i,j,k] += self.mesh.ft_v0[i,j,k] * self.mesh.ft_a[i,j,k]
                        continue
                    
                    m = (i*self.nx[1] + j)*self.nx[2] + k
                    xi = self.x_intp[i,j,k,:,:]
                    ni = self.n_intp[i,j,k,:,:]
                    ti = self.t_intp[i,j,k,:]
                    ii = self.i_intp[i,j,k,:]
                    vi = self.v_intp[i,j,k,:]
                    
                    if load_intp_coef:
                        c0_x0 = self.c0_x0[5,:,:,i,j,k]
                        c1_x0 = self.c1_x0[5,:,:,i,j,k]
                        c2_x0 = self.c2_x0[5,:,:,i,j,k]
                        c0_x1 = self.c0_x1[5,:,:,i,j,k]
                        c1_x1 = self.c1_x1[5,:,:,i,j,k]
                        c2_x1 = self.c2_x1[5,:,:,i,j,k]
                        c0_x2 = self.c0_x2[5,:,:,i,j,k]
                        c1_x2 = self.c1_x2[5,:,:,i,j,k]
                        c2_x2 = self.c2_x2[5,:,:,i,j,k]
                    else:
                        print(m)
                        if (ii!=-1).all():
                            c0_x0, c1_x0, c2_x0 = self.re_c0_x0[5,:,:], self.re_c1_x0[5,:,:], self.re_c2_x0[5,:,:]
                            c0_x1, c1_x1, c2_x1 = self.re_c0_x1[5,:,:], self.re_c1_x1[5,:,:], self.re_c2_x1[5,:,:]
                            c0_x2, c1_x2, c2_x2 = self.re_c0_x2[5,:,:], self.re_c1_x2[5,:,:], self.re_c2_x2[5,:,:]
                        else:
                            c0_x0, c1_x0, c2_x0, c0_x1, c1_x1, c2_x1, c0_x2, c1_x2, c2_x2 = \
                                self.intp_coef(xi, ni, self.mesh.ft_x[i,j,k,:])
                        self.c0_x0[5,:,:,i,j,k] = c0_x0
                        self.c1_x0[5,:,:,i,j,k] = c1_x0
                        self.c2_x0[5,:,:,i,j,k] = c2_x0
                        self.c0_x1[5,:,:,i,j,k] = c0_x1
                        self.c1_x1[5,:,:,i,j,k] = c1_x1
                        self.c2_x1[5,:,:,i,j,k] = c2_x1
                        self.c0_x2[5,:,:,i,j,k] = c0_x2
                        self.c1_x2[5,:,:,i,j,k] = c1_x2
                        self.c2_x2[5,:,:,i,j,k] = c2_x2
                    
                    for s in range(self.intp_node_num):
                        self.wei0_u0[0,s,i,j,k] -= (self.mu*c0_x2[0,s] + self.mu*c2_x0[0,s]) * self.mesh.ft_a[i,j,k]
                        self.wei0_u1[0,s,i,j,k] -= (self.mu*c0_x2[1,s] + self.mu*c2_x0[1,s]) * self.mesh.ft_a[i,j,k]
                        self.wei0_u2[0,s,i,j,k] -= (self.mu*c0_x2[2,s] + self.mu*c2_x0[2,s]) * self.mesh.ft_a[i,j,k]
                        self.wei1_u0[0,s,i,j,k] -= (self.mu*c1_x2[0,s] + self.mu*c2_x1[0,s]) * self.mesh.ft_a[i,j,k]
                        self.wei1_u1[0,s,i,j,k] -= (self.mu*c1_x2[1,s] + self.mu*c2_x1[1,s]) * self.mesh.ft_a[i,j,k]
                        self.wei1_u2[0,s,i,j,k] -= (self.mu*c1_x2[2,s] + self.mu*c2_x1[2,s]) * self.mesh.ft_a[i,j,k]
                        self.wei2_u0[0,s,i,j,k] -= (self.lm*c0_x0[0,s] + self.lm*c1_x1[0,s] + (self.lm+2*self.mu)*c2_x2[0,s]) * self.mesh.ft_a[i,j,k]
                        self.wei2_u1[0,s,i,j,k] -= (self.lm*c0_x0[1,s] + self.lm*c1_x1[1,s] + (self.lm+2*self.mu)*c2_x2[1,s]) * self.mesh.ft_a[i,j,k]
                        self.wei2_u2[0,s,i,j,k] -= (self.lm*c0_x0[2,s] + self.lm*c1_x1[2,s] + (self.lm+2*self.mu)*c2_x2[2,s]) * self.mesh.ft_a[i,j,k]

        self.save_interpolation_coefficient()

        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    if self.mesh.c_loc[i,j,k]!=1:
                        continue

                    m = (i*self.nx[1] + j)*self.nx[2] + k
                    
                    if self.mesh.fws_loc[i,j,k]==0 and self.mesh.fws_t[i,j,k]==2:
                        self.r0[0,0,i,j,k] += self.mesh.fws_a[i,j,k] * self.mesh.fws_v0[i,j,k]
                        self.r1[0,0,i,j,k] += self.mesh.fws_a[i,j,k] * self.mesh.fws_v1[i,j,k]
                        self.r2[0,0,i,j,k] += self.mesh.fws_a[i,j,k] * self.mesh.fws_v2[i,j,k]
                    
                    if self.mesh.fwn_loc[i,j,k]==0 and self.mesh.fwn_t[i,j,k]==2:
                        self.r0[0,0,i,j,k] += self.mesh.fwn_a[i,j,k] * self.mesh.fwn_v0[i,j,k]
                        self.r1[0,0,i,j,k] += self.mesh.fwn_a[i,j,k] * self.mesh.fwn_v1[i,j,k]
                        self.r2[0,0,i,j,k] += self.mesh.fwn_a[i,j,k] * self.mesh.fwn_v2[i,j,k]
                    
                    if self.mesh.fes_loc[i,j,k]==0 and self.mesh.fes_t[i,j,k]==2:
                        self.r0[0,0,i,j,k] += self.mesh.fes_a[i,j,k] * self.mesh.fes_v0[i,j,k]
                        self.r1[0,0,i,j,k] += self.mesh.fes_a[i,j,k] * self.mesh.fes_v1[i,j,k]
                        self.r2[0,0,i,j,k] += self.mesh.fes_a[i,j,k] * self.mesh.fes_v2[i,j,k]

                    if self.mesh.fen_loc[i,j,k]==0 and self.mesh.fen_t[i,j,k]==2:
                        self.r0[0,0,i,j,k] += self.mesh.fen_a[i,j,k] * self.mesh.fen_v0[i,j,k]
                        self.r1[0,0,i,j,k] += self.mesh.fen_a[i,j,k] * self.mesh.fen_v1[i,j,k]
                        self.r2[0,0,i,j,k] += self.mesh.fen_a[i,j,k] * self.mesh.fen_v2[i,j,k]

        self.a00 = torch.zeros(self.mesh.c_size,self.mesh.c_size)
        self.a01 = torch.zeros(self.mesh.c_size,self.mesh.c_size)
        self.a02 = torch.zeros(self.mesh.c_size,self.mesh.c_size)
        self.a10 = torch.zeros(self.mesh.c_size,self.mesh.c_size)
        self.a11 = torch.zeros(self.mesh.c_size,self.mesh.c_size)
        self.a12 = torch.zeros(self.mesh.c_size,self.mesh.c_size)
        self.a20 = torch.zeros(self.mesh.c_size,self.mesh.c_size)
        self.a21 = torch.zeros(self.mesh.c_size,self.mesh.c_size)
        self.a22 = torch.zeros(self.mesh.c_size,self.mesh.c_size)
        self.b0 = torch.zeros(self.mesh.c_size,1)
        self.b1 = torch.zeros(self.mesh.c_size,1)
        self.b2 = torch.zeros(self.mesh.c_size,1)
        
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    if self.mesh.c_loc[i,j,k]!=1:
                        continue
                    
                    m = (i*self.nx[1] + j)*self.nx[2] + k
                    self.b0[m] += self.r0[0,0,i,j,k]
                    self.b1[m] += self.r1[0,0,i,j,k]
                    self.b2[m] += self.r2[0,0,i,j,k]

                    xi = self.x_intp[i,j,k,:,:]
                    ni = self.n_intp[i,j,k,:,:]
                    ti = self.t_intp[i,j,k,:]
                    ii = self.i_intp[i,j,k,:]
                    vi = self.v_intp[i,j,k,:]
                    
                    for s in range(self.intp_node_num):
                        if ii[s]==-1:
                            self.b0[m] -= self.wei0_u0[0,s,i,j,k] * vi[0,s]
                            self.b0[m] -= self.wei0_u1[0,s,i,j,k] * vi[1,s]
                            self.b0[m] -= self.wei0_u2[0,s,i,j,k] * vi[2,s]
                            self.b1[m] -= self.wei1_u0[0,s,i,j,k] * vi[0,s]
                            self.b1[m] -= self.wei1_u1[0,s,i,j,k] * vi[1,s]
                            self.b1[m] -= self.wei1_u2[0,s,i,j,k] * vi[2,s]
                            self.b2[m] -= self.wei2_u0[0,s,i,j,k] * vi[0,s]
                            self.b2[m] -= self.wei2_u1[0,s,i,j,k] * vi[1,s]
                            self.b2[m] -= self.wei2_u2[0,s,i,j,k] * vi[2,s]
                        else:
                            self.a00[m,ii[s]] += self.wei0_u0[0,s,i,j,k]
                            self.a01[m,ii[s]] += self.wei0_u1[0,s,i,j,k]
                            self.a02[m,ii[s]] += self.wei0_u2[0,s,i,j,k]
                            self.a10[m,ii[s]] += self.wei1_u0[0,s,i,j,k]
                            self.a11[m,ii[s]] += self.wei1_u1[0,s,i,j,k]
                            self.a12[m,ii[s]] += self.wei1_u2[0,s,i,j,k]
                            self.a20[m,ii[s]] += self.wei2_u0[0,s,i,j,k]
                            self.a21[m,ii[s]] += self.wei2_u1[0,s,i,j,k]
                            self.a22[m,ii[s]] += self.wei2_u2[0,s,i,j,k]

        

    def intp_node(self, geo, mesh, idx):
        tol = 1e-4
        
        node_num = 3**3
        xi = torch.zeros(node_num,self.dim)
        ni = torch.zeros(node_num,self.dim)
        ti = torch.zeros(node_num)
        ii = torch.zeros(node_num).long()

        dir = [[-1,-1,-1],[-1,-1, 0],[-1,-1, 1], [-1, 0,-1],[-1, 0, 0],[-1, 0, 1], [-1, 1,-1],[-1, 1, 0],[-1, 1, 1],
               [ 0,-1,-1],[ 0,-1, 0],[ 0,-1, 1], [ 0, 0,-1],[ 0, 0, 0],[ 0, 0, 1], [ 0, 1,-1],[ 0, 1, 0],[ 0, 1, 1],
               [ 1,-1,-1],[ 1,-1, 0],[ 1,-1, 1], [ 1, 0,-1],[ 1, 0, 0],[ 1, 0, 1], [ 1, 1,-1],[ 1, 1, 0],[ 1, 1, 1]]
        dir = torch.tensor(dir).reshape(node_num,self.dim)

        # regular point
        ix = idx[0]+dir[:,0]; iy = idx[1]+dir[:,1]; iz = idx[2]+dir[:,2]
        m = (ix*mesh.nx[1] + iy)*mesh.nx[2] + iz
        idx1 = ((ix>=0) & (ix<mesh.nx[0]) & (iy>=0) & (iy<mesh.nx[1]) & (iz>=0) & (iz<mesh.nx[2]))
        idx1[idx1==True] = (mesh.c_loc.reshape(mesh.c_size)[m[idx1]]==1)
        
        xi[idx1,:] = mesh.c_x.reshape(mesh.c_size,self.dim)[m[idx1],:]
        ni[idx1,:] = 0
        ti[idx1] = 1
        ii[idx1] = m[idx1]
        
        # irregular point
        idx1 = ~idx1
        x1 = mesh.c_x[idx[0],idx[1],idx[2],:]
        x1 = x1.reshape(1,self.dim).repeat(node_num,1)
        x2 = x1 + torch.tensor([mesh.hx[0],mesh.hx[1],mesh.hx[2]]) * dir

        xi_tmp = geo.intersection(x1[idx1,:], x2[idx1,:])
        
        xi[idx1,:] = xi_tmp
        ti[idx1] = 2
        ii[idx1] = -1

        idx2 = idx1 & (abs(xi[:,0]-geo.bounds_p1[0,0])<tol)
        ni[idx2,0] = -1
        idx1 = idx1 & (~idx2)

        idx2 = idx1 & (abs(xi[:,0]-geo.bounds_p1[0,1])<tol)
        ni[idx2,0] = 1
        idx1 = idx1 & (~idx2)

        idx2 = idx1 & (abs(xi[:,1]-geo.bounds_p1[1,0])<tol)
        ni[idx2,1] = -1
        idx1 = idx1 & (~idx2)

        idx2 = idx1 & (abs(xi[:,1]-geo.bounds_p1[1,1])<tol)
        ni[idx2,1] = 1
        idx1 = idx1 & (~idx2)

        idx2 = idx1 & (abs(xi[:,2]-geo.bounds_p1[2,0])<tol)
        ni[idx2,2] = -1
        idx1 = idx1 & (~idx2)

        idx2 = idx1 & (abs(xi[:,2]-geo.bounds_p1[2,1])<tol)
        ni[idx2,2] = 1
        idx1 = idx1 & (~idx2)

        idx2 = idx1 & (abs(xi[:,0]-geo.bounds_p2[0,1])<tol)
        ni[idx2,0] = 1
        idx1 = idx1 & (~idx2)

        idx2 = idx1 & (abs(xi[:,2]-geo.bounds_p2[2,0])<tol)
        ni[idx2,2] = -1
        idx1 = idx1 & (~idx2)

        idx2 = idx1 & (abs(xi[:,2]-geo.bounds_p2[2,1])<tol)
        ni[idx2,2] = 1
        idx1 = idx1 & (~idx2)
        
        ni[idx1,0] = -xi[idx1,0]
        ni[idx1,1] = -xi[idx1,1]
        ni[idx1,:] /= (ni[idx1,:]).norm(dim=-1,keepdim=True)

        idx1 = abs(xi[:,0]-geo.bounds_p1[0,0])<tol
        ti[idx1] = 1
        ni[idx1,:] = 0
        
        return xi, ni, ti, ii
    
    def intp_coef(self, xi, ni, x):
        xi = xi.clone().to(torch.float64)
        ni = ni.clone().to(torch.float64)
        x = x.clone().to(torch.float64)
        
        node_num = 3**3
        p = torch.zeros(node_num,node_num, dtype=torch.float64)
        p_x0 = torch.zeros(node_num,node_num, dtype=torch.float64)
        p_x1 = torch.zeros(node_num,node_num, dtype=torch.float64)
        p_x2 = torch.zeros(node_num,node_num, dtype=torch.float64)
        for r in range(3):
            for s in range(3):
                for t in range(3):
                    n = (r*3 + s)*3 + t
                    p[:,n] = xi[:,0]**r * xi[:,1]**s * xi[:,2]**t
                    if r==0:
                        p_x0[:,n] = 0 * xi[:,1]**s * xi[:,2]**t
                    else:
                        p_x0[:,n] = r*xi[:,0]**(r-1) * xi[:,1]**s * xi[:,2]**t
                    if s==0:
                        p_x1[:,n] = xi[:,0]**r * 0 * xi[:,2]**t
                    else:
                        p_x1[:,n] = xi[:,0]**r * s*xi[:,1]**(s-1) * xi[:,2]**t
                    if t==0:
                        p_x2[:,n] = xi[:,0]**r * xi[:,1]**s * 0
                    else:
                        p_x2[:,n] = xi[:,0]**r * xi[:,1]**s * t*xi[:,2]**(t-1)
        
        b00 = p*(1-ni.norm(dim=-1,keepdim=True)) + \
            (self.lm+2*self.mu)*p_x0*ni[:,0:1] + (self.mu)*p_x1*ni[:,1:2] + (self.mu)*p_x2*ni[:,2:3]
        b01 = self.lm*p_x1*ni[:,0:1] + self.mu*p_x0*ni[:,1:2]
        b02 = self.lm*p_x2*ni[:,0:1] + self.mu*p_x0*ni[:,2:3]
        b10 = self.lm*p_x0*ni[:,1:2] + self.mu*p_x1*ni[:,0:1]
        b11 = p*(1-ni.norm(dim=-1,keepdim=True)) + \
            (self.mu)*p_x0*ni[:,0:1] + (self.lm+2*self.mu)*p_x1*ni[:,1:2] + (self.mu)*p_x2*ni[:,2:3]
        b12 = self.lm*p_x2*ni[:,1:2] + self.mu*p_x1*ni[:,2:3]
        b20 = self.lm*p_x0*ni[:,2:3] + self.mu*p_x2*ni[:,0:1]
        b21 = self.lm*p_x1*ni[:,2:3] + self.mu*p_x2*ni[:,1:2]
        b22 = p*(1-ni.norm(dim=-1,keepdim=True)) + \
            (self.mu)*p_x0*ni[:,0:1] + (self.mu)*p_x1*ni[:,1:2] + (self.lm+2*self.mu)*p_x2*ni[:,2:3] 
        b = torch.cat([torch.cat([b00,b01,b02],1),
                       torch.cat([b10,b11,b12],1),
                       torch.cat([b20,b21,b22],1)],0)
        
        b = torch.inverse(b)

        b00 = b[0*self.intp_node_num:1*self.intp_node_num,0*self.intp_node_num:1*self.intp_node_num]
        b01 = b[0*self.intp_node_num:1*self.intp_node_num,1*self.intp_node_num:2*self.intp_node_num]
        b02 = b[0*self.intp_node_num:1*self.intp_node_num,2*self.intp_node_num:3*self.intp_node_num]
        b10 = b[1*self.intp_node_num:2*self.intp_node_num,0*self.intp_node_num:1*self.intp_node_num]
        b11 = b[1*self.intp_node_num:2*self.intp_node_num,1*self.intp_node_num:2*self.intp_node_num]
        b12 = b[1*self.intp_node_num:2*self.intp_node_num,2*self.intp_node_num:3*self.intp_node_num]
        b20 = b[2*self.intp_node_num:3*self.intp_node_num,0*self.intp_node_num:1*self.intp_node_num]
        b21 = b[2*self.intp_node_num:3*self.intp_node_num,1*self.intp_node_num:2*self.intp_node_num]
        b22 = b[2*self.intp_node_num:3*self.intp_node_num,2*self.intp_node_num:3*self.intp_node_num]

        pp = torch.zeros(1,node_num, dtype=torch.float64)
        pp_x0 = torch.zeros(1,node_num, dtype=torch.float64)
        pp_x1 = torch.zeros(1,node_num, dtype=torch.float64)
        pp_x2 = torch.zeros(1,node_num, dtype=torch.float64)
        for r in range(3):
            for s in range(3):
                for t in range(3):
                    n = (r*3 + s)*3 + t
                    pp[0,n] = x[0]**r * x[1]**s * x[2]**t
                    if r==0:
                        pp_x0[0,n] = 0 * x[1]**s * x[2]**t
                    else:
                        pp_x0[0,n] = r*x[0]**(r-1) * x[1]**s * x[2]**t
                    if s==0:
                        pp_x1[0,n] = x[0]**r * 0 * x[2]**t
                    else:
                        pp_x1[0,n] = x[0]**r * s*x[1]**(s-1) * x[2]**t
                    if t==0:
                        pp_x2[0,n] = x[0]**r * x[1]**s * 0
                    else:
                        pp_x2[0,n] = x[0]**r * x[1]**s * t*x[2]**(t-1)
        
        c0 = torch.zeros(3,node_num, dtype=torch.float64)
        c0_x0 = torch.zeros(3,node_num, dtype=torch.float64)
        c0_x1 = torch.zeros(3,node_num, dtype=torch.float64)
        c0_x2 = torch.zeros(3,node_num, dtype=torch.float64)
        c1 = torch.zeros(3,node_num, dtype=torch.float64)
        c1_x0 = torch.zeros(3,node_num, dtype=torch.float64)
        c1_x1 = torch.zeros(3,node_num, dtype=torch.float64)
        c1_x2 = torch.zeros(3,node_num, dtype=torch.float64)
        c2 = torch.zeros(3,node_num, dtype=torch.float64)
        c2_x0 = torch.zeros(3,node_num, dtype=torch.float64)
        c2_x1 = torch.zeros(3,node_num, dtype=torch.float64)
        c2_x2 = torch.zeros(3,node_num, dtype=torch.float64)   

        c0[0:1,:] = pp @ b00
        c0[1:2,:] = pp @ b01
        c0[2:3,:] = pp @ b02
        c0_x0[0:1,:] = pp_x0 @ b00
        c0_x0[1:2,:] = pp_x0 @ b01
        c0_x0[2:3,:] = pp_x0 @ b02
        c0_x1[0:1,:] = pp_x1 @ b00
        c0_x1[1:2,:] = pp_x1 @ b01
        c0_x1[2:3,:] = pp_x1 @ b02
        c0_x2[0:1,:] = pp_x2 @ b00
        c0_x2[1:2,:] = pp_x2 @ b01
        c0_x2[2:3,:] = pp_x2 @ b02

        c1[0:1,:] = pp @ b10
        c1[1:2,:] = pp @ b11
        c1[2:3,:] = pp @ b12
        c1_x0[0:1,:] = pp_x0 @ b10
        c1_x0[1:2,:] = pp_x0 @ b11
        c1_x0[2:3,:] = pp_x0 @ b12
        c1_x1[0:1,:] = pp_x1 @ b10
        c1_x1[1:2,:] = pp_x1 @ b11
        c1_x1[2:3,:] = pp_x1 @ b12
        c1_x2[0:1,:] = pp_x2 @ b10
        c1_x2[1:2,:] = pp_x2 @ b11
        c1_x2[2:3,:] = pp_x2 @ b12

        c2[0:1,:] = pp @ b20
        c2[1:2,:] = pp @ b21
        c2[2:3,:] = pp @ b22
        c2_x0[0:1,:] = pp_x0 @ b20
        c2_x0[1:2,:] = pp_x0 @ b21
        c2_x0[2:3,:] = pp_x0 @ b22
        c2_x1[0:1,:] = pp_x1 @ b20
        c2_x1[1:2,:] = pp_x1 @ b21
        c2_x1[2:3,:] = pp_x1 @ b22
        c2_x2[0:1,:] = pp_x2 @ b20
        c2_x2[1:2,:] = pp_x2 @ b21
        c2_x2[2:3,:] = pp_x2 @ b22

        c0_x0 = c0_x0.clone().to(self.dtype)
        c0_x1 = c0_x1.clone().to(self.dtype)
        c0_x2 = c0_x2.clone().to(self.dtype)
        c1_x0 = c1_x0.clone().to(self.dtype)
        c1_x1 = c1_x1.clone().to(self.dtype)
        c1_x2 = c1_x2.clone().to(self.dtype)
        c2_x0 = c2_x0.clone().to(self.dtype)
        c2_x1 = c2_x1.clone().to(self.dtype)
        c2_x2 = c2_x2.clone().to(self.dtype)
        return c0_x0, c1_x0, c2_x0, c0_x1, c1_x1, c2_x1, c0_x2, c1_x2, c2_x2

    def load_interpolation_coefficient(self):
        self.c0_x0 = torch.load('intp_coef/c0_x0.pt')
        self.c1_x0 = torch.load('intp_coef/c1_x0.pt')
        self.c2_x0 = torch.load('intp_coef/c2_x0.pt')
        self.c0_x1 = torch.load('intp_coef/c0_x1.pt')
        self.c1_x1 = torch.load('intp_coef/c1_x1.pt')
        self.c2_x1 = torch.load('intp_coef/c2_x1.pt')
        self.c0_x2 = torch.load('intp_coef/c0_x2.pt')
        self.c1_x2 = torch.load('intp_coef/c1_x2.pt')
        self.c2_x2 = torch.load('intp_coef/c2_x2.pt')
        
    def save_interpolation_coefficient(self):
        torch.save(self.c0_x0, 'intp_coef/c0_x0.pt')
        torch.save(self.c1_x0, 'intp_coef/c1_x0.pt')
        torch.save(self.c2_x0, 'intp_coef/c2_x0.pt')
        torch.save(self.c0_x1, 'intp_coef/c0_x1.pt')
        torch.save(self.c1_x1, 'intp_coef/c1_x1.pt')
        torch.save(self.c2_x1, 'intp_coef/c2_x1.pt')
        torch.save(self.c0_x2, 'intp_coef/c0_x2.pt')
        torch.save(self.c1_x2, 'intp_coef/c1_x2.pt')
        torch.save(self.c2_x2, 'intp_coef/c2_x2.pt')
        
    def to(self, device):
        self.device = device

        self.x = self.x.to(self.device)
        self.p = self.p.to(self.device)
        self.mask = self.mask.to(self.device)
        
        self.wei0_u0 = self.wei0_u0.to(self.device)
        self.wei0_u1 = self.wei0_u1.to(self.device)
        self.wei0_u2 = self.wei0_u2.to(self.device)
        self.wei1_u0 = self.wei1_u0.to(self.device)
        self.wei1_u1 = self.wei1_u1.to(self.device)
        self.wei1_u2 = self.wei1_u2.to(self.device)
        self.wei2_u0 = self.wei2_u0.to(self.device)
        self.wei2_u1 = self.wei2_u1.to(self.device)
        self.wei2_u2 = self.wei2_u2.to(self.device)

        self.v0 = self.v0.to(self.device)
        self.v1 = self.v1.to(self.device)
        self.v2 = self.v2.to(self.device)

        self.r0 = self.r0.to(self.device)
        self.r1 = self.r1.to(self.device)
        self.r2 = self.r2.to(self.device)

class TeSet():
    def __init__(self, file_name, nx, dtype):
        self.nx = nx
        self.dtype = dtype
        
        data = pd.read_csv(file_name, header=None)
        data = np.array(data)
        data = torch.tensor(data, dtype=self.dtype)
        
        self.x0 = data[:,0:1].reshape(1,1,self.nx[0],self.nx[1],self.nx[2])
        self.x1 = data[:,1:2].reshape(1,1,self.nx[0],self.nx[1],self.nx[2])
        self.x2 = data[:,2:3].reshape(1,1,self.nx[0],self.nx[1],self.nx[2])
        self.u0a = data[:,3:4].reshape(1,1,self.nx[0],self.nx[1],self.nx[2])
        self.u1a = data[:,4:5].reshape(1,1,self.nx[0],self.nx[1],self.nx[2])
        self.u2a = data[:,5:6].reshape(1,1,self.nx[0],self.nx[1],self.nx[2])
        self.mask = data[:,6:7].reshape(1,1,self.nx[0],self.nx[1],self.nx[2])
        
    def to(self, device):
        self.device = device

        self.x0 = self.x0.to(self.device)
        self.x1 = self.x1.to(self.device)
        self.x2 = self.x2.to(self.device)
        self.u0a = self.u0a.to(self.device)
        self.u1a = self.u1a.to(self.device)
        self.u2a = self.u2a.to(self.device)
        self.mask = self.mask.to(self.device)
