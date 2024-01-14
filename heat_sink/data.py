import torch
import numpy as np
import pandas as pd

class TrSet():
    def __init__(self, geo, mesh, re, dtype, load_intp_coef):
        self.geo = geo
        self.mesh = mesh
        self.re = re
        self.dtype = dtype

        self.nu = 1.0/self.re

        self.parm_size = self.mesh.parm_size
        self.dim = self.mesh.c_x.shape[-1]
        self.nx = self.mesh.nx
        
        self.parm = self.mesh.c_v.reshape(self.parm_size,1,self.nx[0],self.nx[1],self.nx[2])
        self.parm /= self.parm.max()
        self.mask = (self.mesh.c_loc==1).reshape(self.parm_size,1,self.nx[0],self.nx[1],self.nx[2])
        self.mask = self.mask.clone().to(self.dtype)

        """ boundary value on the cell face (if cell face is located on the boundary) """
        self.mesh.fw_v0_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fw_v1_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fw_v2_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fe_v0_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fe_v1_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fe_v2_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fs_v0_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fs_v1_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fs_v2_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fn_v0_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fn_v1_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fn_v2_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fb_v0_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fb_v1_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fb_v2_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.ft_v0_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.ft_v1_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.ft_v2_bd = torch.zeros(self.parm_size,self.mesh.c_size)
        for p in range(self.parm_size):
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    for k in range(self.nx[2]):
                        m = (i*self.nx[1] + j)*self.nx[2] + k
                        if self.mesh.c_loc[p,m]!=1:
                            continue
                        
                        if self.mesh.fw_a_bd[p,m]>0:
                            if i==0:
                                self.mesh.fw_v0_bd[p,m] = 1.0
        
        tol = 1e-4
        self.intp_n_size = 3**3
        self.intp_x = torch.zeros(self.parm_size,self.mesh.c_size,self.intp_n_size,self.dim)
        self.intp_i = torch.zeros(self.parm_size,self.mesh.c_size,self.intp_n_size).long()
        self.intp_v = torch.zeros(self.parm_size,self.mesh.c_size,self.intp_n_size,3)
        for p in range(self.parm_size):
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    for k in range(self.nx[2]):
                        m = (i*self.nx[1] + j)*self.nx[2] + k
                        print(m)
                        if self.mesh.c_loc[p,m]!=1:
                            continue
                            
                        self.intp_x[p,m,:,:], self.intp_i[p,m,:] = self.intp_node([i,j,k], p)

                        ii = self.intp_i[p,m,:]
                        for r in range(3):
                            for s in range(3):
                                for t in range(3):
                                    n = (r*3 + s)*3 + t
                                    if ii[n]==-1 and i==0 and r==0:
                                        self.intp_v[p,m,n,0] = 1.0
        
        self.v0 = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.v1 = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.v2 = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        for p in range(self.parm_size):
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    for k in range(self.nx[2]):
                        m = (i*self.nx[1] + j)*self.nx[2] + k
                        if self.mesh.c_loc[p,m]!=1:
                            continue
                        
                        ii = self.intp_i[p,m,:]
                        vi = self.intp_v[p,m,:,:]
                        
                        for n in range(self.intp_n_size):
                            self.v0[p,n,i,j,k] = vi[n,0] if ii[n]==-1 else 0
                            self.v1[p,n,i,j,k] = vi[n,1] if ii[n]==-1 else 0
                            self.v2[p,n,i,j,k] = vi[n,2] if ii[n]==-1 else 0
        
        self.intp_c_x0_in = torch.zeros(self.parm_size,6,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.intp_c_x1_in = torch.zeros(self.parm_size,6,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.intp_c_x2_in = torch.zeros(self.parm_size,6,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.intp_c_x0_bd = torch.zeros(self.parm_size,6,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.intp_c_x1_bd = torch.zeros(self.parm_size,6,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.intp_c_x2_bd = torch.zeros(self.parm_size,6,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        if load_intp_coef:
            self.load_interpolation_coefficient()
        
        self.r0 = torch.zeros(self.parm_size,1,self.nx[0],self.nx[1],self.nx[2])
        self.r1 = torch.zeros(self.parm_size,1,self.nx[0],self.nx[1],self.nx[2])
        self.r2 = torch.zeros(self.parm_size,1,self.nx[0],self.nx[1],self.nx[2])
        self.r3 = torch.zeros(self.parm_size,1,self.nx[0],self.nx[1],self.nx[2])
        
        self.wei0_u0 = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.wei0_u1 = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.wei0_u2 = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.wei0_p = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.wei1_u0 = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.wei1_u1 = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.wei1_u2 = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.wei1_p = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.wei2_u0 = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.wei2_u1 = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.wei2_u2 = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.wei2_p = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.wei3_u0 = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.wei3_u1 = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.wei3_u2 = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        self.wei3_p = torch.zeros(self.parm_size,self.intp_n_size,self.nx[0],self.nx[1],self.nx[2])
        
        """ interpolation coefficients for regular unit """
        self.re_c_x0_in = torch.zeros(6,self.intp_n_size)
        self.re_c_x1_in = torch.zeros(6,self.intp_n_size)
        self.re_c_x2_in = torch.zeros(6,self.intp_n_size)
        flag = False
        for p in range(self.parm_size):
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    for k in range(self.nx[2]):
                        m = (i*self.nx[1] + j)*self.nx[2] + k
                        if self.mesh.c_loc[p,m]!=1:
                            continue
                        
                        xi = self.intp_x[p,m,:,:]
                        ii = self.intp_i[p,m,:]
                        vi = self.intp_v[p,m,:,:]

                        if (ii!=-1).all():
                            self.re_c_x0_in[0,:], self.re_c_x1_in[0,:], self.re_c_x2_in[0,:] = \
                                self.intp_coef_2(xi, self.mesh.fw_x_in[p,m,:])
                            
                            self.re_c_x0_in[1,:], self.re_c_x1_in[1,:], self.re_c_x2_in[1,:] = \
                                self.intp_coef_2(xi, self.mesh.fe_x_in[p,m,:])
                            
                            self.re_c_x0_in[2,:], self.re_c_x1_in[2,:], self.re_c_x2_in[2,:] = \
                                self.intp_coef_2(xi, self.mesh.fs_x_in[p,m,:])
                            
                            self.re_c_x0_in[3,:], self.re_c_x1_in[3,:], self.re_c_x2_in[3,:] = \
                                self.intp_coef_2(xi, self.mesh.fn_x_in[p,m,:])
                            
                            self.re_c_x0_in[4,:], self.re_c_x1_in[4,:], self.re_c_x2_in[4,:] = \
                                self.intp_coef_2(xi, self.mesh.fb_x_in[p,m,:])
                            
                            self.re_c_x0_in[5,:], self.re_c_x1_in[5,:], self.re_c_x2_in[5,:] = \
                                self.intp_coef_2(xi, self.mesh.ft_x_in[p,m,:])
                            
                            flag = True

                        if flag: break
                    if flag: break
                if flag: break
            if flag: break
        
        # momentum equation
        # volecity
        for p in range(self.parm_size):
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    for k in range(self.nx[2]):
                        m = (i*self.nx[1] + j)*self.nx[2] + k
                        if self.mesh.c_loc[p,m]!=1:
                            continue
                        
                        xi = self.intp_x[p,m,:,:]
                        ii = self.intp_i[p,m,:]
                        vi = self.intp_v[p,m,:,:]
                        
                        # west face
                        if self.mesh.fw_a_in[p,m]>0:
                            if load_intp_coef:
                                c_x0 = self.intp_c_x0_in[p,0,:,i,j,k]
                                c_x1 = self.intp_c_x1_in[p,0,:,i,j,k]
                                c_x2 = self.intp_c_x2_in[p,0,:,i,j,k]
                            else:
                                print(m)
                                if (ii!=-1).all():
                                    c_x0, c_x1, c_x2 = self.re_c_x0_in[0,:], self.re_c_x1_in[0,:], self.re_c_x2_in[0,:]
                                else:
                                    c_x0, c_x1, c_x2 = self.intp_coef_2(xi, self.mesh.fw_x_in[p,m,:])
                                self.intp_c_x0_in[p,0,:,i,j,k] = c_x0
                                self.intp_c_x1_in[p,0,:,i,j,k] = c_x1
                                self.intp_c_x2_in[p,0,:,i,j,k] = c_x2

                            for n in range(self.intp_n_size):
                                self.wei0_u0[p,n,i,j,k] += -self.nu * c_x0[n] * (-1) * self.mesh.fw_a_in[p,m]
                                self.wei1_u1[p,n,i,j,k] += -self.nu * c_x0[n] * (-1) * self.mesh.fw_a_in[p,m]
                                self.wei2_u2[p,n,i,j,k] += -self.nu * c_x0[n] * (-1) * self.mesh.fw_a_in[p,m]

                        if self.mesh.fw_a_bd[p,m]>0:
                            if load_intp_coef:
                                c_x0 = self.intp_c_x0_bd[p,0,:,i,j,k]
                                c_x1 = self.intp_c_x1_bd[p,0,:,i,j,k]
                                c_x2 = self.intp_c_x2_bd[p,0,:,i,j,k]
                            else:
                                c_x0, c_x1, c_x2 = self.intp_coef_2(xi, self.mesh.fw_x_bd[p,m,:])
                                self.intp_c_x0_bd[p,0,:,i,j,k] = c_x0
                                self.intp_c_x1_bd[p,0,:,i,j,k] = c_x1
                                self.intp_c_x2_bd[p,0,:,i,j,k] = c_x2

                            for n in range(self.intp_n_size):
                                self.wei0_u0[p,n,i,j,k] += -self.nu * c_x0[n] * (-1) * self.mesh.fw_a_bd[p,m]
                                self.wei1_u1[p,n,i,j,k] += -self.nu * c_x0[n] * (-1) * self.mesh.fw_a_bd[p,m]
                                self.wei2_u2[p,n,i,j,k] += -self.nu * c_x0[n] * (-1) * self.mesh.fw_a_bd[p,m]

                        # east face
                        if i!=self.nx[0]-1:
                            if self.mesh.fe_a_in[p,m]>0:
                                if load_intp_coef:
                                    c_x0 = self.intp_c_x0_in[p,1,:,i,j,k]
                                    c_x1 = self.intp_c_x1_in[p,1,:,i,j,k]
                                    c_x2 = self.intp_c_x2_in[p,1,:,i,j,k]
                                else:
                                    print(m)
                                    if (ii!=-1).all():
                                        c_x0, c_x1, c_x2 = self.re_c_x0_in[1,:], self.re_c_x1_in[1,:], self.re_c_x2_in[1,:]
                                    else:
                                        c_x0, c_x1, c_x2 = self.intp_coef_2(xi, self.mesh.fe_x_in[p,m,:])
                                    self.intp_c_x0_in[p,1,:,i,j,k] = c_x0
                                    self.intp_c_x1_in[p,1,:,i,j,k] = c_x1
                                    self.intp_c_x2_in[p,1,:,i,j,k] = c_x2

                                for n in range(self.intp_n_size):
                                    self.wei0_u0[p,n,i,j,k] += -self.nu * c_x0[n] * (1) * self.mesh.fe_a_in[p,m]
                                    self.wei1_u1[p,n,i,j,k] += -self.nu * c_x0[n] * (1) * self.mesh.fe_a_in[p,m]
                                    self.wei2_u2[p,n,i,j,k] += -self.nu * c_x0[n] * (1) * self.mesh.fe_a_in[p,m]
                            
                            if self.mesh.fe_a_bd[p,m]>0:
                                if load_intp_coef:
                                    c_x0 = self.intp_c_x0_bd[p,1,:,i,j,k]
                                    c_x1 = self.intp_c_x1_bd[p,1,:,i,j,k]
                                    c_x2 = self.intp_c_x2_bd[p,1,:,i,j,k]
                                else:
                                    c_x0, c_x1, c_x2 = self.intp_coef_2(xi, self.mesh.fe_x_bd[p,m,:])
                                    self.intp_c_x0_bd[p,1,:,i,j,k] = c_x0
                                    self.intp_c_x1_bd[p,1,:,i,j,k] = c_x1
                                    self.intp_c_x2_bd[p,1,:,i,j,k] = c_x2

                                for n in range(self.intp_n_size):
                                    self.wei0_u0[p,n,i,j,k] += -self.nu * c_x0[n] * (1) * self.mesh.fe_a_bd[p,m]
                                    self.wei1_u1[p,n,i,j,k] += -self.nu * c_x0[n] * (1) * self.mesh.fe_a_bd[p,m]
                                    self.wei2_u2[p,n,i,j,k] += -self.nu * c_x0[n] * (1) * self.mesh.fe_a_bd[p,m]

                        # south face
                        if self.mesh.fs_a_in[p,m]>0:
                            if load_intp_coef:
                                c_x0 = self.intp_c_x0_in[p,2,:,i,j,k]
                                c_x1 = self.intp_c_x1_in[p,2,:,i,j,k]
                                c_x2 = self.intp_c_x2_in[p,2,:,i,j,k]
                            else:
                                print(m)
                                if (ii!=-1).all():
                                    c_x0, c_x1, c_x2 = self.re_c_x0_in[2,:], self.re_c_x1_in[2,:], self.re_c_x2_in[2,:]
                                else:
                                    c_x0, c_x1, c_x2 = self.intp_coef_2(xi, self.mesh.fs_x_in[p,m,:])
                                self.intp_c_x0_in[p,2,:,i,j,k] = c_x0
                                self.intp_c_x1_in[p,2,:,i,j,k] = c_x1
                                self.intp_c_x2_in[p,2,:,i,j,k] = c_x2

                            for n in range(self.intp_n_size):
                                self.wei0_u0[p,n,i,j,k] += -self.nu * c_x1[n] * (-1) * self.mesh.fs_a_in[p,m]
                                self.wei1_u1[p,n,i,j,k] += -self.nu * c_x1[n] * (-1) * self.mesh.fs_a_in[p,m]
                                self.wei2_u2[p,n,i,j,k] += -self.nu * c_x1[n] * (-1) * self.mesh.fs_a_in[p,m]
                        
                        if self.mesh.fs_a_bd[p,m]>0:
                            if load_intp_coef:
                                c_x0 = self.intp_c_x0_bd[p,2,:,i,j,k]
                                c_x1 = self.intp_c_x1_bd[p,2,:,i,j,k]
                                c_x2 = self.intp_c_x2_bd[p,2,:,i,j,k]
                            else:
                                c_x0, c_x1, c_x2 = self.intp_coef_2(xi, self.mesh.fs_x_bd[p,m,:])
                                self.intp_c_x0_bd[p,2,:,i,j,k] = c_x0
                                self.intp_c_x1_bd[p,2,:,i,j,k] = c_x1
                                self.intp_c_x2_bd[p,2,:,i,j,k] = c_x2
                                
                            for n in range(self.intp_n_size):
                                self.wei0_u0[p,n,i,j,k] += -self.nu * c_x1[n] * (-1) * self.mesh.fs_a_bd[p,m]
                                self.wei1_u1[p,n,i,j,k] += -self.nu * c_x1[n] * (-1) * self.mesh.fs_a_bd[p,m]
                                self.wei2_u2[p,n,i,j,k] += -self.nu * c_x1[n] * (-1) * self.mesh.fs_a_bd[p,m]

                        # north face
                        if self.mesh.fn_a_in[p,m]>0:
                            if load_intp_coef:
                                c_x0 = self.intp_c_x0_in[p,3,:,i,j,k]
                                c_x1 = self.intp_c_x1_in[p,3,:,i,j,k]
                                c_x2 = self.intp_c_x2_in[p,3,:,i,j,k]
                            else:
                                print(m)
                                if (ii!=-1).all():
                                    c_x0, c_x1, c_x2 = self.re_c_x0_in[3,:], self.re_c_x1_in[3,:], self.re_c_x2_in[3,:]
                                else:
                                    c_x0, c_x1, c_x2 = self.intp_coef_2(xi, self.mesh.fn_x_in[p,m,:])
                                self.intp_c_x0_in[p,3,:,i,j,k] = c_x0
                                self.intp_c_x1_in[p,3,:,i,j,k] = c_x1
                                self.intp_c_x2_in[p,3,:,i,j,k] = c_x2

                            for n in range(self.intp_n_size):
                                self.wei0_u0[p,n,i,j,k] += -self.nu * c_x1[n] * (1) * self.mesh.fn_a_in[p,m]
                                self.wei1_u1[p,n,i,j,k] += -self.nu * c_x1[n] * (1) * self.mesh.fn_a_in[p,m]
                                self.wei2_u2[p,n,i,j,k] += -self.nu * c_x1[n] * (1) * self.mesh.fn_a_in[p,m]
                        
                        if self.mesh.fn_a_bd[p,m]>0:
                            if load_intp_coef:
                                c_x0 = self.intp_c_x0_bd[p,3,:,i,j,k]
                                c_x1 = self.intp_c_x1_bd[p,3,:,i,j,k]
                                c_x2 = self.intp_c_x2_bd[p,3,:,i,j,k]
                            else:
                                c_x0, c_x1, c_x2 = self.intp_coef_2(xi, self.mesh.fn_x_bd[p,m,:])
                                self.intp_c_x0_bd[p,3,:,i,j,k] = c_x0
                                self.intp_c_x1_bd[p,3,:,i,j,k] = c_x1
                                self.intp_c_x2_bd[p,3,:,i,j,k] = c_x2

                            for n in range(self.intp_n_size):
                                self.wei0_u0[p,n,i,j,k] += -self.nu * c_x1[n] * (1) * self.mesh.fn_a_bd[p,m]
                                self.wei1_u1[p,n,i,j,k] += -self.nu * c_x1[n] * (1) * self.mesh.fn_a_bd[p,m]
                                self.wei2_u2[p,n,i,j,k] += -self.nu * c_x1[n] * (1) * self.mesh.fn_a_bd[p,m]
            
                        # bottom face
                        if self.mesh.fb_a_in[p,m]>0:
                            if load_intp_coef:
                                c_x0 = self.intp_c_x0_in[p,4,:,i,j,k]
                                c_x1 = self.intp_c_x1_in[p,4,:,i,j,k]
                                c_x2 = self.intp_c_x2_in[p,4,:,i,j,k]
                            else:
                                print(m)
                                if (ii!=-1).all():
                                    c_x0, c_x1, c_x2 = self.re_c_x0_in[4,:], self.re_c_x1_in[4,:], self.re_c_x2_in[4,:]
                                else:
                                    c_x0, c_x1, c_x2 = self.intp_coef_2(xi, self.mesh.fb_x_in[p,m,:])
                                self.intp_c_x0_in[p,4,:,i,j,k] = c_x0
                                self.intp_c_x1_in[p,4,:,i,j,k] = c_x1
                                self.intp_c_x2_in[p,4,:,i,j,k] = c_x2

                            for n in range(self.intp_n_size):
                                self.wei0_u0[p,n,i,j,k] += -self.nu * c_x2[n] * (-1) * self.mesh.fb_a_in[p,m]
                                self.wei1_u1[p,n,i,j,k] += -self.nu * c_x2[n] * (-1) * self.mesh.fb_a_in[p,m]
                                self.wei2_u2[p,n,i,j,k] += -self.nu * c_x2[n] * (-1) * self.mesh.fb_a_in[p,m]
                        
                        if self.mesh.fb_a_bd[p,m]>0:
                            if load_intp_coef:
                                c_x0 = self.intp_c_x0_bd[p,4,:,i,j,k]
                                c_x1 = self.intp_c_x1_bd[p,4,:,i,j,k]
                                c_x2 = self.intp_c_x2_bd[p,4,:,i,j,k]
                            else:
                                c_x0, c_x1, c_x2 = self.intp_coef_2(xi, self.mesh.fb_x_bd[p,m,:])
                                self.intp_c_x0_bd[p,4,:,i,j,k] = c_x0
                                self.intp_c_x1_bd[p,4,:,i,j,k] = c_x1
                                self.intp_c_x2_bd[p,4,:,i,j,k] = c_x2

                            for n in range(self.intp_n_size):
                                self.wei0_u0[p,n,i,j,k] += -self.nu * c_x2[n] * (-1) * self.mesh.fb_a_bd[p,m]
                                self.wei1_u1[p,n,i,j,k] += -self.nu * c_x2[n] * (-1) * self.mesh.fb_a_bd[p,m]
                                self.wei2_u2[p,n,i,j,k] += -self.nu * c_x2[n] * (-1) * self.mesh.fb_a_bd[p,m]

                        # top face
                        if self.mesh.ft_a_in[p,m]>0:
                            if load_intp_coef:
                                c_x0 = self.intp_c_x0_in[p,5,:,i,j,k]
                                c_x1 = self.intp_c_x1_in[p,5,:,i,j,k]
                                c_x2 = self.intp_c_x2_in[p,5,:,i,j,k]
                            else:
                                print(m)
                                if (ii!=-1).all():
                                    c_x0, c_x1, c_x2 = self.re_c_x0_in[5,:], self.re_c_x1_in[5,:], self.re_c_x2_in[5,:]
                                else:
                                    c_x0, c_x1, c_x2 = self.intp_coef_2(xi, self.mesh.ft_x_in[p,m,:])
                                self.intp_c_x0_in[p,5,:,i,j,k] = c_x0
                                self.intp_c_x1_in[p,5,:,i,j,k] = c_x1
                                self.intp_c_x2_in[p,5,:,i,j,k] = c_x2

                            for n in range(self.intp_n_size):
                                self.wei0_u0[p,n,i,j,k] += -self.nu * c_x2[n] * (1) * self.mesh.ft_a_in[p,m]
                                self.wei1_u1[p,n,i,j,k] += -self.nu * c_x2[n] * (1) * self.mesh.ft_a_in[p,m]
                                self.wei2_u2[p,n,i,j,k] += -self.nu * c_x2[n] * (1) * self.mesh.ft_a_in[p,m]
                        
                        if self.mesh.ft_a_bd[p,m]>0:
                            if load_intp_coef:
                                c_x0 = self.intp_c_x0_bd[p,5,:,i,j,k]
                                c_x1 = self.intp_c_x1_bd[p,5,:,i,j,k]
                                c_x2 = self.intp_c_x2_bd[p,5,:,i,j,k]
                            else:
                                c_x0, c_x1, c_x2 = self.intp_coef_2(xi, self.mesh.ft_x_bd[p,m,:])
                                self.intp_c_x0_bd[p,5,:,i,j,k] = c_x0
                                self.intp_c_x1_bd[p,5,:,i,j,k] = c_x1
                                self.intp_c_x2_bd[p,5,:,i,j,k] = c_x2

                            for n in range(self.intp_n_size):
                                self.wei0_u0[p,n,i,j,k] += -self.nu * c_x2[n] * (1) * self.mesh.ft_a_bd[p,m]
                                self.wei1_u1[p,n,i,j,k] += -self.nu * c_x2[n] * (1) * self.mesh.ft_a_bd[p,m]
                                self.wei2_u2[p,n,i,j,k] += -self.nu * c_x2[n] * (1) * self.mesh.ft_a_bd[p,m]
        
        self.save_interpolation_coefficient()
        
        # pressure
        for p in range(self.parm_size):
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    for k in range(self.nx[2]):
                        m = (i*self.nx[1] + j)*self.nx[2] + k
                        if self.mesh.c_loc[p,m]!=1:
                            continue
                        
                        # west face
                        if self.mesh.nw_loc[p,m]==1 and self.mesh.ne_loc[p,m]==1 and i!=self.nx[0]-1:
                            ji = [4,22]; xi = self.intp_x[p,m,ji,:]
                        if self.mesh.nw_loc[p,m]!=1 and i!=self.nx[0]-1:
                            ji = [13,22]; xi = self.intp_x[p,m,ji,:]
                        if self.mesh.ne_loc[p,m]!=1 or i==self.nx[0]-1:
                            ji = [4,13]; xi = self.intp_x[p,m,ji,:]

                        if self.mesh.fw_a_in[p,m]>tol:
                            c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.fw_x_in[p,m,:])
                            for n in range(2):
                                self.wei0_p[p,ji[n],i,j,k] += c[n] * (-1) * self.mesh.fw_a_in[p,m]
                            
                        if self.mesh.fw_a_bd[p,m]>tol:
                            c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.fw_x_bd[p,m,:])
                            for n in range(2):
                                self.wei0_p[p,ji[n],i,j,k] += c[n] * (-1) * self.mesh.fw_a_bd[p,m]
            
                        # east face
                        if i!=self.nx[0]-1:
                            if self.mesh.fe_a_in[p,m]>tol:
                                c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.fe_x_in[p,m,:])
                                for n in range(2):
                                    self.wei0_p[p,ji[n],i,j,k] += c[n] * (1) * self.mesh.fe_a_in[p,m] 
                            
                            if self.mesh.fe_a_bd[p,m]>tol:
                                c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.fe_x_bd[p,m,:])
                                for n in range(2):
                                    self.wei0_p[p,ji[n],i,j,k] += c[n] * (1) * self.mesh.fe_a_bd[p,m]
                        
                        # south face
                        if self.mesh.ns_loc[p,m]==1 and self.mesh.nn_loc[p,m]==1:
                            ji = [10,16]; xi = self.intp_x[p,m,ji,:]
                        if self.mesh.ns_loc[p,m]!=1:
                            ji = [13,16]; xi = self.intp_x[p,m,ji,:]
                        if self.mesh.nn_loc[p,m]!=1:
                            ji = [10,13]; xi = self.intp_x[p,m,ji,:]

                        if self.mesh.fs_a_in[p,m]>tol:
                            c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.fs_x_in[p,m,:])
                            for n in range(2):
                                self.wei1_p[p,ji[n],i,j,k] += c[n] * (-1) * self.mesh.fs_a_in[p,m]
                                
                        if self.mesh.fs_a_bd[p,m]>tol:
                            c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.fs_x_bd[p,m,:])
                            for n in range(2):
                                self.wei1_p[p,ji[n],i,j,k] += c[n] * (-1) * self.mesh.fs_a_bd[p,m]

                        # north face
                        if self.mesh.fn_a_in[p,m]>tol:
                            c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.fn_x_in[p,m,:])
                            for n in range(2):
                                self.wei1_p[p,ji[n],i,j,k] += c[n] * (1) * self.mesh.fn_a_in[p,m]
                            
                        if self.mesh.fn_a_bd[p,m]>tol:
                            c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.fn_x_bd[p,m,:])
                            for n in range(2):
                                self.wei1_p[p,ji[n],i,j,k] += c[n] * (1) * self.mesh.fn_a_bd[p,m]
                        
                        # north face
                        if self.mesh.nb_loc[p,m]==1 and self.mesh.nt_loc[p,m]==1:
                            ji = [12,14]; xi = self.intp_x[p,m,ji,:]
                        if self.mesh.nb_loc[p,m]!=1:
                            ji = [13,14]; xi = self.intp_x[p,m,ji,:]
                        if self.mesh.nt_loc[p,m]!=1:
                            ji = [12,13]; xi = self.intp_x[p,m,ji,:]

                        if self.mesh.fb_a_in[p,m]>tol:
                            c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.fb_x_in[p,m,:])
                            for n in range(2):
                                self.wei2_p[p,ji[n],i,j,k] += c[n] * (-1) * self.mesh.fb_a_in[p,m]
                                
                        if self.mesh.fb_a_bd[p,m]>tol:
                            c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.fb_x_bd[p,m,:])
                            for n in range(2):
                                self.wei2_p[p,ji[n],i,j,k] += c[n] * (-1) * self.mesh.fb_a_bd[p,m]
                                
                        # top face
                        if self.mesh.ft_a_in[p,m]>tol:
                            c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.ft_x_in[p,m,:])
                            for n in range(2):
                                self.wei2_p[p,ji[n],i,j,k] += c[n] * (1) * self.mesh.ft_a_in[p,m]
                        
                        if self.mesh.ft_a_bd[p,m]>tol:
                            c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.ft_x_bd[p,m,:])
                            for n in range(2):
                                self.wei2_p[p,ji[n],i,j,k] += c[n] * (1) * self.mesh.ft_a_bd[p,m]
        
        # continuity equation
        for p in range(self.parm_size):
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    for k in range(self.nx[2]):
                        m = (i*self.nx[1] + j)*self.nx[2] + k
                        if self.mesh.c_loc[p,m]!=1:
                            continue
                        
                        # west face
                        ji = [4,13]
                        xi = self.intp_x[p,m,ji,:]
                        
                        if self.mesh.fw_a_in[p,m]>tol:
                            c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.fw_x_in[p,m,:])
                            for n in range(2):
                                self.wei3_u0[p,ji[n],i,j,k] += c[n] * (-1) * self.mesh.fw_a_in[p,m]
                        
                        if self.mesh.fw_a_bd[p,m]>tol:
                            self.r3[p,0,i,j,k] -= self.mesh.fw_v0_bd[p,m] * (-1) * self.mesh.fw_a_bd[p,m]
                        
                        # east face
                        ji = [13,22]
                        xi = self.intp_x[p,m,ji,:]
                        if i!=self.nx[0]-1:
                            if self.mesh.fe_a_in[p,m]>tol:
                                c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.fe_x_in[p,m,:])
                                for n in range(2):
                                    self.wei3_u0[p,ji[n],i,j,k] += c[n] * (1) * self.mesh.fe_a_in[p,m]
                            
                            if self.mesh.fe_a_bd[p,m]>tol:
                                self.r3[p,0,i,j,k] -= self.mesh.fe_v0_bd[p,m] * (1) * self.mesh.fe_a_bd[p,m]
                        else:
                            self.wei3_u0[p,13,i,j,k] += self.mesh.fe_a_bd[p,m]
                        
                        # south face
                        ji = [10,13]
                        xi = self.intp_x[p,m,ji,:]
                        
                        if self.mesh.fs_a_in[p,m]>tol:
                            c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.fs_x_in[p,m,:])
                            for n in range(2):
                                self.wei3_u1[p,ji[n],i,j,k] += c[n] * (-1) * self.mesh.fs_a_in[p,m]
                        
                        if self.mesh.fs_a_bd[p,m]>tol:
                            self.r3[p,0,i,j,k] -= self.mesh.fs_v1_bd[p,m] * (-1) * self.mesh.fs_a_bd[p,m]

                        # north face
                        ji = [13,16]
                        xi = self.intp_x[p,m,ji,:]
                        
                        if self.mesh.fn_a_in[p,m]>tol:
                            c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.fn_x_in[p,m,:])
                            for n in range(2):
                                self.wei3_u1[p,ji[n],i,j,k] += c[n] * (1) * self.mesh.fn_a_in[p,m] 
                        
                        if self.mesh.fn_a_bd[p,m]>tol:
                            self.r3[p,0,i,j,k] -= self.mesh.fn_v1_bd[p,m] * (1) * self.mesh.fn_a_bd[p,m]
                        
                        # bottom face
                        ji = [12,13]
                        xi = self.intp_x[p,m,ji,:]
                        
                        if self.mesh.fb_a_in[p,m]>tol:
                            c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.fb_x_in[p,m,:])
                            for n in range(2):
                                self.wei3_u2[p,ji[n],i,j,k] += c[n] * (-1) * self.mesh.fb_a_in[p,m]
                        
                        if self.mesh.fb_a_bd[p,m]>tol:
                            self.r3[p,0,i,j,k] -= self.mesh.fb_v2_bd[p,m] * (-1) * self.mesh.fb_a_bd[p,m]
                            
                        # top face
                        ji = [13,14]
                        xi = self.intp_x[p,m,ji,:]
                        
                        if self.mesh.ft_a_in[p,m]>tol:
                            c, c_x0, c_x1, c_x2 = self.intp_coef_1(xi, self.mesh.ft_x_in[p,m,:])
                            for n in range(2):
                                self.wei3_u2[p,ji[n],i,j,k] += c[n] * (1) * self.mesh.ft_a_in[p,m] 
                        
                        if self.mesh.ft_a_bd[p,m]>tol:
                            self.r3[p,0,i,j,k] -= self.mesh.ft_v2_bd[p,m] * (1) * self.mesh.ft_a_bd[p,m]
        '''
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

                    xi = self.intp_x[i,j,k,:,:]
                    ni = self.n_intp[i,j,k,:,:]
                    ti = self.t_intp[i,j,k,:]
                    ii = self.intp_i[i,j,k,:]
                    vi = self.intp_v[i,j,k,:]
                    
                    for s in range(self.intp_n_size):
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
        '''
    def intp_node(self, idx, p):
        intp_n_size = 3**3
        xi = torch.zeros(intp_n_size,self.dim)
        ii = torch.zeros(intp_n_size).long()

        dir = [[-1,-1,-1],[-1,-1, 0],[-1,-1, 1], [-1, 0,-1],[-1, 0, 0],[-1, 0, 1], [-1, 1,-1],[-1, 1, 0],[-1, 1, 1],
               [ 0,-1,-1],[ 0,-1, 0],[ 0,-1, 1], [ 0, 0,-1],[ 0, 0, 0],[ 0, 0, 1], [ 0, 1,-1],[ 0, 1, 0],[ 0, 1, 1],
               [ 1,-1,-1],[ 1,-1, 0],[ 1,-1, 1], [ 1, 0,-1],[ 1, 0, 0],[ 1, 0, 1], [ 1, 1,-1],[ 1, 1, 0],[ 1, 1, 1]]
        dir = torch.tensor(dir).reshape(intp_n_size,self.dim)

        # regular point
        ix = idx[0]+dir[:,0]; iy = idx[1]+dir[:,1]; iz = idx[2]+dir[:,2]
        m = (ix*self.mesh.nx[1] + iy)*self.mesh.nx[2] + iz
        idx1 = ((ix>=0) & (ix<self.mesh.nx[0]) & (iy>=0) & (iy<self.mesh.nx[1]) & (iz>=0) & (iz<self.mesh.nx[2]))
        idx1[idx1==True] = (self.mesh.c_loc[p,m[idx1]]==1)
        
        xi[idx1,:] = self.mesh.c_x[p,m[idx1],:]
        ii[idx1] = m[idx1]
        
        # irregular point
        idx1 = ~idx1
        m = (idx[0]*self.mesh.nx[1] + idx[1])*self.mesh.nx[2] + idx[2]
        x1 = self.mesh.c_x[p,m,:]
        x1 = x1.reshape(1,self.dim).repeat(intp_n_size,1)
        x2 = x1 + torch.tensor([self.mesh.hx[0],self.mesh.hx[1],self.mesh.hx[2]]) * dir

        xi_tmp = self.geo.intersection(x1[idx1,:], x2[idx1,:], p)
        
        xi[idx1,:] = xi_tmp
        ii[idx1] = -1

        return xi, ii
    
    def intp_coef_1(self, xi, x):
        tol = 1e-4

        intp_n_size = 2
        c = torch.zeros(intp_n_size)
        c_x0 = torch.zeros(intp_n_size); c_x1 = torch.zeros(intp_n_size); c_x2 = torch.zeros(intp_n_size)

        if abs(xi[0,1]-xi[1,1])<tol and abs(xi[0,2]-xi[1,2])<tol:
            c[0] = (xi[1,0]-x[0])/(xi[1,0]-xi[0,0])
            c[1] = (x[0]-xi[0,0])/(xi[1,0]-xi[0,0])
            c_x0[0] = 1/(xi[1,0]-xi[0,0])
            c_x0[1] = 1/(xi[1,0]-xi[0,0])

        if abs(xi[0,0]-xi[1,0])<tol and abs(xi[0,2]-xi[1,2])<tol:
            c[0] = (xi[1,1]-x[1])/(xi[1,1]-xi[0,1])
            c[1] = (x[1]-xi[0,1])/(xi[1,1]-xi[0,1])
            c_x1[0] = 1/(xi[1,1]-xi[0,1])
            c_x1[1] = 1/(xi[1,1]-xi[0,1])

        if abs(xi[0,0]-xi[1,0])<tol and abs(xi[0,1]-xi[1,1])<tol:
            c[0] = (xi[1,2]-x[2])/(xi[1,2]-xi[0,2])
            c[1] = (x[2]-xi[0,2])/(xi[1,2]-xi[0,2])
            c_x2[0] = 1/(xi[1,2]-xi[0,2])
            c_x2[1] = 1/(xi[1,2]-xi[0,2])
        
        return c, c_x0, c_x1, c_x2

    def intp_coef_2(self, xi, x):
        xi = xi.clone().to(torch.float64)
        x = x.clone().to(torch.float64)
        
        intp_n_size = 3**3
        p = torch.zeros(intp_n_size,intp_n_size, dtype=torch.float64)
        for r in range(3):
            for s in range(3):
                for t in range(3):
                    n = (r*3 + s)*3 + t
                    p[:,n] = xi[:,0]**r * xi[:,1]**s * xi[:,2]**t
        b = p
        b = torch.inverse(b)
        
        pp = torch.zeros(1,intp_n_size, dtype=torch.float64)
        pp_x0 = torch.zeros(1,intp_n_size, dtype=torch.float64)
        pp_x1 = torch.zeros(1,intp_n_size, dtype=torch.float64)
        pp_x2 = torch.zeros(1,intp_n_size, dtype=torch.float64)
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
        
        c = (pp @ b).reshape(intp_n_size)
        c_x0 = (pp_x0 @ b).reshape(intp_n_size)
        c_x1 = (pp_x1 @ b).reshape(intp_n_size)
        c_x2 = (pp_x2 @ b).reshape(intp_n_size)
        
        c_x0 = c_x0.clone().to(self.dtype)
        c_x1 = c_x1.clone().to(self.dtype)
        c_x2 = c_x2.clone().to(self.dtype)
        return c_x0, c_x1, c_x2

    def load_interpolation_coefficient(self):
        self.intp_c_x0_in = torch.load('intp_coef/c_x0_in.pt')
        self.intp_c_x1_in = torch.load('intp_coef/c_x1_in.pt')
        self.intp_c_x2_in = torch.load('intp_coef/c_x2_in.pt')
        self.intp_c_x0_bd = torch.load('intp_coef/c_x0_bd.pt')
        self.intp_c_x1_bd = torch.load('intp_coef/c_x1_bd.pt')
        self.intp_c_x2_bd = torch.load('intp_coef/c_x2_bd.pt')
        
    def save_interpolation_coefficient(self):
        torch.save(self.intp_c_x0_in, 'intp_coef/c_x0_in.pt')
        torch.save(self.intp_c_x1_in, 'intp_coef/c_x1_in.pt')
        torch.save(self.intp_c_x2_in, 'intp_coef/c_x2_in.pt')
        torch.save(self.intp_c_x0_bd, 'intp_coef/c_x0_bd.pt')
        torch.save(self.intp_c_x1_bd, 'intp_coef/c_x1_bd.pt')
        torch.save(self.intp_c_x2_bd, 'intp_coef/c_x2_bd.pt')
        
    def to(self, device):
        self.device = device

        self.mesh.c_x = self.mesh.c_x.to(self.device)
        self.parm = self.parm.to(self.device)
        self.mask = self.mask.to(self.device)
        
        self.wei0_u0 = self.wei0_u0.to(self.device)
        self.wei0_u1 = self.wei0_u1.to(self.device)
        self.wei0_u2 = self.wei0_u2.to(self.device)
        self.wei0_p = self.wei0_p.to(self.device)
        self.wei1_u0 = self.wei1_u0.to(self.device)
        self.wei1_u1 = self.wei1_u1.to(self.device)
        self.wei1_u2 = self.wei1_u2.to(self.device)
        self.wei1_p = self.wei1_p.to(self.device)
        self.wei2_u0 = self.wei2_u0.to(self.device)
        self.wei2_u1 = self.wei2_u1.to(self.device)
        self.wei2_u2 = self.wei2_u2.to(self.device)
        self.wei2_p = self.wei2_p.to(self.device)
        self.wei3_u0 = self.wei3_u0.to(self.device)
        self.wei3_u1 = self.wei3_u1.to(self.device)
        self.wei3_u2 = self.wei3_u2.to(self.device)
        self.wei3_p = self.wei3_p.to(self.device)
        
        self.v0 = self.v0.to(self.device)
        self.v1 = self.v1.to(self.device)
        self.v2 = self.v2.to(self.device)

        self.r0 = self.r0.to(self.device)
        self.r1 = self.r1.to(self.device)
        self.r2 = self.r2.to(self.device)
        self.r3 = self.r3.to(self.device)

class TeSet():
    def __init__(self, file_name, parm_size, nx, dtype):
        self.parm_size = parm_size
        self.nx = nx
        self.dtype = dtype
        
        data = pd.read_csv(file_name, header=None)
        data = np.array(data)
        data = torch.tensor(data, dtype=self.dtype)
        
        self.x0 = data[:,0:1].reshape(self.parm_size,1,self.nx[0],self.nx[1],self.nx[2])
        self.x1 = data[:,1:2].reshape(self.parm_size,1,self.nx[0],self.nx[1],self.nx[2])
        self.x2 = data[:,2:3].reshape(self.parm_size,1,self.nx[0],self.nx[1],self.nx[2])
        self.parm = data[:,3:4].reshape(self.parm_size,1,self.nx[0],self.nx[1],self.nx[2])
        self.u0a = data[:,4:5].reshape(self.parm_size,1,self.nx[0],self.nx[1],self.nx[2])
        self.u1a = data[:,5:6].reshape(self.parm_size,1,self.nx[0],self.nx[1],self.nx[2])
        self.u2a = data[:,6:7].reshape(self.parm_size,1,self.nx[0],self.nx[1],self.nx[2])
        self.pa = data[:,7:8].reshape(self.parm_size,1,self.nx[0],self.nx[1],self.nx[2])
        self.mask = data[:,8:9].reshape(self.parm_size,1,self.nx[0],self.nx[1],self.nx[2])
        
        self.parm /= self.parm.max()

    def to(self, device):
        self.device = device

        self.x0 = self.x0.to(self.device)
        self.x1 = self.x1.to(self.device)
        self.x2 = self.x2.to(self.device)
        self.parm = self.parm.to(self.device)
        self.u0a = self.u0a.to(self.device)
        self.u1a = self.u1a.to(self.device)
        self.u2a = self.u2a.to(self.device)
        self.pa = self.pa.to(self.device)
        self.mask = self.mask.to(self.device)
