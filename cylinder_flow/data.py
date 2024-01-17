import torch
import numpy as np
import pandas as pd

class TrSet():
    def __init__(self, geo, mesh, re, dtype):
        self.geo = geo
        self.mesh = mesh
        self.re = re
        self.dtype = dtype

        self.nu = 1.0/self.re

        self.parm_size = self.mesh.parm_size
        self.dim = self.mesh.c_x.shape[-1]
        self.nx = self.mesh.nx
        
        self.parm = self.mesh.c_a.reshape(self.parm_size,1,self.nx[0],self.nx[1])
        self.parm /= self.parm.max()
        self.mask = (self.mesh.c_loc==1).reshape(self.parm_size,1,self.nx[0],self.nx[1])
        self.mask = self.mask.clone().to(self.dtype)

        """ boundary value on the cell face (if cell face is located on the boundary) """
        print('Generating boundary value ...')
        self.mesh.fw_v0 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fe_v0 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fs_v0 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fn_v0 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fw_v1 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fe_v1 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fs_v1 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fn_v1 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fw_v2 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fe_v2 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fs_v2 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fn_v2 = torch.zeros(self.parm_size,self.mesh.c_size)
        for p in range(self.parm_size):
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    m = i*self.nx[1] + j
                    if self.mesh.c_loc[p,m]!=1:
                        continue

                    if i==0:
                        self.mesh.fw_v0[p,m] = 1.0
        
        self.mesh.fws_v0 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fwn_v0 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fes_v0 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fen_v0 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fws_v1 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fwn_v1 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fes_v1 = torch.zeros(self.parm_size,self.mesh.c_size)
        self.mesh.fen_v1 = torch.zeros(self.parm_size,self.mesh.c_size)
        
        """ boundary value on the interpolation node (if node is located on the boundary) """
        self.mesh.intp_v = torch.zeros(self.parm_size,self.mesh.c_size,self.mesh.intp_n_size,3)
        for p in range(self.parm_size):
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    m = i*self.nx[1] + j
                    if self.mesh.c_loc[p,m]!=1:
                        continue

                    ii = self.mesh.intp_i[p,m,:]
                    for r in range(3):
                        for s in range(3):
                            n = r*3 + s
                            if ii[n]==-1 and i==0 and r==0:
                                self.mesh.intp_v[p,m,n,0] = 1.0

        # Reshape for calculating the value of loss function
        self.v0 = (self.mesh.intp_v[:,:,:,0].permute(0,2,1)).reshape(
            self.parm_size,self.mesh.intp_n_size,self.nx[0],self.nx[1])
        self.v1 = (self.mesh.intp_v[:,:,:,1].permute(0,2,1)).reshape(
            self.parm_size,self.mesh.intp_n_size,self.nx[0],self.nx[1])
        self.v2 = (self.mesh.intp_v[:,:,:,2].permute(0,2,1)).reshape(
            self.parm_size,self.mesh.intp_n_size,self.nx[0],self.nx[1])
        
        """ right hand side """
        self.r0 = torch.zeros(self.parm_size,1,self.nx[0],self.nx[1])
        self.r1 = torch.zeros(self.parm_size,1,self.nx[0],self.nx[1])
        self.r2 = torch.zeros(self.parm_size,1,self.nx[0],self.nx[1])

        """ weight for calculating the value of loss function """
        self.wei0_u0 = torch.zeros(self.parm_size,self.mesh.intp_n_size,self.nx[0],self.nx[1])
        self.wei0_u1 = torch.zeros(self.parm_size,self.mesh.intp_n_size,self.nx[0],self.nx[1])
        self.wei0_p = torch.zeros(self.parm_size,self.mesh.intp_n_size,self.nx[0],self.nx[1])
        self.wei1_u0 = torch.zeros(self.parm_size,self.mesh.intp_n_size,self.nx[0],self.nx[1])
        self.wei1_u1 = torch.zeros(self.parm_size,self.mesh.intp_n_size,self.nx[0],self.nx[1])
        self.wei1_p = torch.zeros(self.parm_size,self.mesh.intp_n_size,self.nx[0],self.nx[1])
        self.wei2_u0 = torch.zeros(self.parm_size,self.mesh.intp_n_size,self.nx[0],self.nx[1])
        self.wei2_u1 = torch.zeros(self.parm_size,self.mesh.intp_n_size,self.nx[0],self.nx[1])
        self.wei2_p = torch.zeros(self.parm_size,self.mesh.intp_n_size,self.nx[0],self.nx[1])

        """ interpolation coefficients for regular unit """
        self.re_c = torch.zeros(4,self.mesh.intp_n_size)
        self.re_c_x0 = torch.zeros(4,self.mesh.intp_n_size)
        self.re_c_x1 = torch.zeros(4,self.mesh.intp_n_size)
        flag = False
        for p in range(self.parm_size):
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    m = i*self.nx[1] + j
                    if self.mesh.c_loc[p,m]!=1:
                        continue
                    
                    xi = self.mesh.intp_x[p,m,:,:]
                    ii = self.mesh.intp_i[p,m,:]
                    vi = self.mesh.intp_v[p,m,:,:]

                    if (ii!=-1).all():
                        self.re_c[0,:], self.re_c_x0[0,:], self.re_c_x1[0,:] = \
                            self.intp_coef_2(xi, self.mesh.fw_x[p,m,:])
                        self.re_c[1,:], self.re_c_x0[1,:], self.re_c_x1[1,:] = \
                            self.intp_coef_2(xi, self.mesh.fe_x[p,m,:])
                        self.re_c[2,:], self.re_c_x0[2,:], self.re_c_x1[2,:] = \
                            self.intp_coef_2(xi, self.mesh.fs_x[p,m,:])
                        self.re_c[3,:], self.re_c_x0[3,:], self.re_c_x1[3,:] = \
                            self.intp_coef_2(xi, self.mesh.fn_x[p,m,:])
                        flag = True

                    if flag: break
                if flag: break
            if flag: break
        
        # momentum equation
        print('Generating weight for evaluating the residual of momentum equation ...')
        for p in range(self.parm_size):
            print('for parameter: c = [{:.2f},{:.2f}]'.format(self.geo.center[p,0],self.geo.center[p,1]))
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    m = i*self.nx[1] + j
                    # print(m)
                    if self.mesh.c_loc[p,m]!=1:
                        continue
                    
                    # volecity
                    # west face
                    if i!=self.mesh.nx[0]-1:
                        intp_n_size = self.mesh.intp_n_size
                        ji = [0,1,2,3,4,5,6,7,8]
                        xi = self.mesh.intp_x[p,m,ji,:]; ii = self.mesh.intp_i[p,m,ji]
                        if (ii!=-1).all():
                            c, c_x0, c_x1 = self.re_c[0,:], self.re_c_x0[0,:], self.re_c_x1[0,:]
                        else:
                            c, c_x0, c_x1 = self.intp_coef_2(xi, self.mesh.fw_x[p,m,:])
                    else:
                        intp_n_size = 2
                        ji = [1,4]; xi = self.mesh.intp_x[p,m,ji,:]
                        c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fw_x[p,m,:])

                    for n in range(intp_n_size):
                        diff = -self.nu * (c_x0[n]*self.mesh.fw_n[p,m,0] + c_x1[n]*self.mesh.fw_n[p,m,1])
                        self.wei0_u0[p,ji[n],i,j] += diff * self.mesh.fw_l[p,m]
                        self.wei1_u1[p,ji[n],i,j] += diff * self.mesh.fw_l[p,m]
                    
                    # east face
                    if i!=self.mesh.nx[0]-1:
                        intp_n_size = self.mesh.intp_n_size
                        ji = [0,1,2,3,4,5,6,7,8]
                        xi = self.mesh.intp_x[p,m,ji,:]; ii = self.mesh.intp_i[p,m,ji]
                        if (ii!=-1).all():
                            c, c_x0, c_x1 = self.re_c[1,:], self.re_c_x0[1,:], self.re_c_x1[1,:]
                        else:
                            c, c_x0, c_x1 = self.intp_coef_2(xi, self.mesh.fe_x[p,m,:])
                    else:
                        intp_n_size = 2
                        ji = [1,4]; xi = self.mesh.intp_x[p,m,ji,:]
                        c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fe_x[p,m,:])

                    for n in range(intp_n_size):
                        diff = -self.nu * (c_x0[n]*self.mesh.fe_n[p,m,0] + c_x1[n]*self.mesh.fe_n[p,m,1])
                        self.wei0_u0[p,ji[n],i,j] += diff * self.mesh.fe_l[p,m]
                        self.wei1_u1[p,ji[n],i,j] += diff * self.mesh.fe_l[p,m]
                    
                    # south face
                    intp_n_size = self.mesh.intp_n_size
                    ji = [0,1,2,3,4,5,6,7,8]
                    xi = self.mesh.intp_x[p,m,ji,:]; ii = self.mesh.intp_i[p,m,ji]
                    if (ii!=-1).all():
                        c, c_x0, c_x1 = self.re_c[2,:], self.re_c_x0[2,:], self.re_c_x1[2,:]
                    else:
                        c, c_x0, c_x1 = self.intp_coef_2(xi, self.mesh.fs_x[p,m,:])

                    for n in range(intp_n_size):
                        diff = -self.nu * (c_x0[n]*self.mesh.fs_n[p,m,0] + c_x1[n]*self.mesh.fs_n[p,m,1])
                        self.wei0_u0[p,ji[n],i,j] += diff * self.mesh.fs_l[p,m]
                        self.wei1_u1[p,ji[n],i,j] += diff * self.mesh.fs_l[p,m]
                    
                    # north face
                    intp_n_size = self.mesh.intp_n_size
                    ji = [0,1,2,3,4,5,6,7,8]
                    xi = self.mesh.intp_x[p,m,ji,:]; ii = self.mesh.intp_i[p,m,ji]
                    if (ii!=-1).all():
                        c, c_x0, c_x1 = self.re_c[3,:], self.re_c_x0[3,:], self.re_c_x1[3,:]
                    else:
                        c, c_x0, c_x1 = self.intp_coef_2(xi, self.mesh.fn_x[p,m,:])

                    for n in range(intp_n_size):
                        diff = -self.nu * (c_x0[n]*self.mesh.fn_n[p,m,0] + c_x1[n]*self.mesh.fn_n[p,m,1])
                        self.wei0_u0[p,ji[n],i,j] += diff * self.mesh.fn_l[p,m]
                        self.wei1_u1[p,ji[n],i,j] += diff * self.mesh.fn_l[p,m]
                    
                    # west south face
                    tol = 1e-4
                    if mesh.fws_l[p,m]>tol:
                        intp_n_size = self.mesh.intp_n_size
                        ji = [0,1,2,3,4,5,6,7,8]
                        xi = self.mesh.intp_x[p,m,ji,:]
                        c, c_x0, c_x1 = self.intp_coef_2(xi, self.mesh.fws_x[p,m,:])

                    for n in range(intp_n_size):
                        diff = -self.nu * (c_x0[n]*self.mesh.fws_n[p,m,0] + c_x1[n]*self.mesh.fws_n[p,m,1])
                        self.wei0_u0[p,ji[n],i,j] += diff * self.mesh.fws_l[p,m]
                        self.wei1_u1[p,ji[n],i,j] += diff * self.mesh.fws_l[p,m]

                    # west north face
                    if mesh.fwn_l[p,m]>tol:
                        intp_n_size = self.mesh.intp_n_size
                        ji = [0,1,2,3,4,5,6,7,8]
                        xi = self.mesh.intp_x[p,m,ji,:]
                        c, c_x0, c_x1 = self.intp_coef_2(xi, self.mesh.fwn_x[p,m,:])

                    for n in range(intp_n_size):
                        diff = -self.nu * (c_x0[n]*self.mesh.fwn_n[p,m,0] + c_x1[n]*self.mesh.fwn_n[p,m,1])
                        self.wei0_u0[p,ji[n],i,j] += diff * self.mesh.fwn_l[p,m]
                        self.wei1_u1[p,ji[n],i,j] += diff * self.mesh.fwn_l[p,m]
                    
                    # east south face
                    if mesh.fes_l[p,m]>tol:
                        intp_n_size = self.mesh.intp_n_size
                        ji = [0,1,2,3,4,5,6,7,8]
                        xi = self.mesh.intp_x[p,m,ji,:]
                        c, c_x0, c_x1 = self.intp_coef_2(xi, self.mesh.fes_x[p,m,:])

                    for n in range(intp_n_size):
                        diff = -self.nu * (c_x0[n]*self.mesh.fes_n[p,m,0] + c_x1[n]*self.mesh.fes_n[p,m,1])
                        self.wei0_u0[p,ji[n],i,j] += diff * self.mesh.fes_l[p,m]
                        self.wei1_u1[p,ji[n],i,j] += diff * self.mesh.fes_l[p,m]
                    
                    # east north face
                    if mesh.fen_l[p,m]>tol:
                        intp_n_size = self.mesh.intp_n_size
                        ji = [0,1,2,3,4,5,6,7,8]
                        xi = self.mesh.intp_x[p,m,ji,:]
                        c, c_x0, c_x1 = self.intp_coef_2(xi, self.mesh.fen_x[p,m,:])

                    for n in range(intp_n_size):
                        diff = -self.nu * (c_x0[n]*self.mesh.fen_n[p,m,0] + c_x1[n]*self.mesh.fen_n[p,m,1])
                        self.wei0_u0[p,ji[n],i,j] += diff * self.mesh.fen_l[p,m]
                        self.wei1_u1[p,ji[n],i,j] += diff * self.mesh.fen_l[p,m]

                    # presure dp/dx
                    intp_n_size = 2
                    if mesh.nw_loc[p,m]==1 and mesh.ne_loc[p,m]==1:
                        ji = [1,7]; xi = self.mesh.intp_x[p,m,ji,:]
                    if mesh.nw_loc[p,m]!=1:
                        ji = [4,7]; xi = self.mesh.intp_x[p,m,ji,:]
                    if mesh.ne_loc[p,m]!=1:
                        ji = [1,4]; xi = self.mesh.intp_x[p,m,ji,:]
                    if i==mesh.nx[0]-1:
                        ji = [1,4]; xi = self.mesh.intp_x[p,m,ji,:]
                    
                    # west face
                    c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fw_x[p,m,:])
                    for n in range(intp_n_size):
                        self.wei0_p[p,ji[n],i,j] += c[n]*self.mesh.fw_n[p,m,0] * self.mesh.fw_l[p,m]
                    
                    # east face
                    if i!=self.mesh.nx[0]-1:
                        c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fe_x[p,m,:])
                        for n in range(intp_n_size):
                            self.wei0_p[p,ji[n],i,j] += c[n]*self.mesh.fe_n[p,m,0] * self.mesh.fe_l[p,m]
                    else:
                        self.r0[p,0,i,j] -= self.mesh.fe_v2[p,m] * self.mesh.fe_n[p,m,0] * self.mesh.fe_l[p,m]
                    
                    # south face
                    c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fs_x[p,m,:])
                    for n in range(intp_n_size):
                        self.wei0_p[p,ji[n],i,j] += c[n]*self.mesh.fs_n[p,m,0] * self.mesh.fs_l[p,m]
                    
                    # north face
                    c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fn_x[p,m,:])
                    for n in range(intp_n_size):
                        self.wei0_p[p,ji[n],i,j] += c[n]*self.mesh.fn_n[p,m,0] * self.mesh.fn_l[p,m]
                    
                    # west south face
                    if self.mesh.fws_l[p,m]>tol:
                        c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fws_x[p,m,:])
                        for n in range(intp_n_size):
                            self.wei0_p[p,ji[n],i,j] += c[n]*self.mesh.fws_n[p,m,0] * self.mesh.fws_l[p,m]
                    
                    # west north face
                    if self.mesh.fwn_l[p,m]>tol:
                        c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fwn_x[p,m,:])
                        for n in range(intp_n_size):
                            self.wei0_p[p,ji[n],i,j] += c[n]*self.mesh.fwn_n[p,m,0] * self.mesh.fwn_l[p,m]
                    
                    # east south face
                    if self.mesh.fes_l[p,m]>tol:
                        c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fes_x[p,m,:])
                        for n in range(intp_n_size):
                            self.wei0_p[p,ji[n],i,j] += c[n]*self.mesh.fes_n[p,m,0] * self.mesh.fes_l[p,m]
                    
                    # east north face
                    if self.mesh.fen_l[p,m]>tol:
                        c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fen_x[p,m,:])
                        for n in range(intp_n_size):
                            self.wei0_p[p,ji[n],i,j] += c[n]*self.mesh.fen_n[p,m,0] * self.mesh.fen_l[p,m]

                    # presure dp/dy
                    intp_n_size = 2
                    if mesh.ns_loc[p,m]==1 and mesh.nn_loc[p,m]==1:
                        ji = [3,5]; xi = self.mesh.intp_x[p,m,ji,:]
                    if mesh.ns_loc[p,m]!=1:
                        ji = [4,5]; xi = self.mesh.intp_x[p,m,ji,:]
                    if mesh.nn_loc[p,m]!=1:
                        ji = [3,4]; xi = self.mesh.intp_x[p,m,ji,:]
                    
                    # west face
                    c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fw_x[p,m,:])
                    for n in range(intp_n_size):
                        self.wei1_p[p,ji[n],i,j] += c[n]*self.mesh.fw_n[p,m,1] * self.mesh.fw_l[p,m]
                    
                    # east face
                    if i!=self.mesh.nx[0]-1:
                        c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fe_x[p,m,:])
                        for n in range(intp_n_size):
                            self.wei1_p[p,ji[n],i,j] += c[n]*self.mesh.fe_n[p,m,1] * self.mesh.fe_l[p,m]
                    
                    # south face
                    c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fs_x[p,m,:])
                    for n in range(intp_n_size):
                        self.wei1_p[p,ji[n],i,j] += c[n]*self.mesh.fs_n[p,m,1] * self.mesh.fs_l[p,m]
                    
                    # north face
                    c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fn_x[p,m,:])
                    for n in range(intp_n_size):
                        self.wei1_p[p,ji[n],i,j] += c[n]*self.mesh.fn_n[p,m,1] * self.mesh.fn_l[p,m]
                    
                    # west south face
                    if self.mesh.fws_l[p,m]>tol:
                        c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fws_x[p,m,:])
                        for n in range(intp_n_size):
                            self.wei1_p[p,ji[n],i,j] += c[n]*self.mesh.fws_n[p,m,1] * self.mesh.fws_l[p,m]
                    
                    # west north face
                    if self.mesh.fwn_l[p,m]>tol:
                        c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fwn_x[p,m,:])
                        for n in range(intp_n_size):
                            self.wei1_p[p,ji[n],i,j] += c[n]*self.mesh.fwn_n[p,m,1] * self.mesh.fwn_l[p,m]
                    
                    # east south face
                    if self.mesh.fes_l[p,m]>tol:
                        c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fes_x[p,m,:])
                        for n in range(intp_n_size):
                            self.wei1_p[p,ji[n],i,j] += c[n]*self.mesh.fes_n[p,m,1] * self.mesh.fes_l[p,m]
                    
                    # east north face
                    if self.mesh.fen_l[p,m]>tol:
                        c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fen_x[p,m,:])
                        for n in range(intp_n_size):
                            self.wei1_p[p,ji[n],i,j] += c[n]*self.mesh.fen_n[p,m,1] * self.mesh.fen_l[p,m]
        
        # continuity equation
        print('Generating weight for evaluating the residual of continuity equation ...')
        for p in range(self.parm_size):
            print('for parameter: c = [{:.2f},{:.2f}]'.format(self.geo.center[p,0],self.geo.center[p,1]))
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    m = i*self.nx[1] + j
                    # print(m)
                    if self.mesh.c_loc[p,m]!=1:
                        continue
                    
                    # west face
                    if i!=self.mesh.nx[0]-1:
                        intp_n_size = self.mesh.intp_n_size
                        ji = [0,1,2,3,4,5,6,7,8]
                        xi = self.mesh.intp_x[p,m,ji,:]; ii = self.mesh.intp_i[p,m,ji]
                        if (ii!=-1).all():
                            c, c_x0, c_x1 = self.re_c[0,:], self.re_c_x0[0,:], self.re_c_x1[0,:]
                        else:
                            c, c_x0, c_x1 = self.intp_coef_2(xi, self.mesh.fw_x[p,m,:])
                    else:
                        intp_n_size = 2
                        ji = [1,4]; xi = self.mesh.intp_x[p,m,ji,:]
                        c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fw_x[p,m,:])

                    if self.mesh.fw_loc[p,m]==1:
                        for n in range(intp_n_size):
                            self.wei2_u0[p,ji[n],i,j] += c[n]*self.mesh.fw_n[p,m,0] * self.mesh.fw_l[p,m]
                            self.wei2_u1[p,ji[n],i,j] += c[n]*self.mesh.fw_n[p,m,1] * self.mesh.fw_l[p,m]
                    if self.mesh.fw_loc[p,m]==0:
                        self.r2[p,0,i,j] -= (self.mesh.fw_v0[p,m] * self.mesh.fw_n[p,m,0] + 
                                             self.mesh.fw_v1[p,m] * self.mesh.fw_n[p,m,1]) * self.mesh.fw_l[p,m]
                    
                    # east face
                    if i!=self.mesh.nx[0]-1:
                        intp_n_size = self.mesh.intp_n_size
                        ji = [0,1,2,3,4,5,6,7,8]
                        xi = self.mesh.intp_x[p,m,ji,:]; ii = self.mesh.intp_i[p,m,ji]
                        if (ii!=-1).all():
                            c, c_x0, c_x1 = self.re_c[1,:], self.re_c_x0[1,:], self.re_c_x1[1,:]
                        else:
                            c, c_x0, c_x1 = self.intp_coef_2(xi, self.mesh.fe_x[p,m,:])
                    else:
                        intp_n_size = 2
                        ji = [1,4]; xi = self.mesh.intp_x[p,m,ji,:]
                        c, c_x0, c_x1 = self.intp_coef_1(xi, self.mesh.fe_x[p,m,:])
                    
                    if i!=self.nx[0]-1:
                        if self.mesh.fe_loc[p,m]==1:
                            for n in range(intp_n_size):
                                self.wei2_u0[p,ji[n],i,j] += c[n]*self.mesh.fe_n[p,m,0] * self.mesh.fe_l[p,m]
                                self.wei2_u1[p,ji[n],i,j] += c[n]*self.mesh.fe_n[p,m,1] * self.mesh.fe_l[p,m]
                        if self.mesh.fe_loc[p,m]==0:
                            self.r2[p,0,i,j] -= (self.mesh.fe_v0[p,m] * self.mesh.fe_n[p,m,0] + 
                                                 self.mesh.fe_v1[p,m] * self.mesh.fe_n[p,m,1]) * self.mesh.fe_l[p,m]
                    else:
                        self.wei2_u0[p,4,i,j] += self.mesh.fe_n[p,m,0] * self.mesh.fe_l[p,m]
                        self.wei2_u1[p,4,i,j] += self.mesh.fe_n[p,m,1] * self.mesh.fe_l[p,m]

                    # south face
                    intp_n_size = self.mesh.intp_n_size
                    ji = [0,1,2,3,4,5,6,7,8]
                    xi = self.mesh.intp_x[p,m,ji,:]; ii = self.mesh.intp_i[p,m,ji]
                    if (ii!=-1).all():
                        c, c_x0, c_x1 = self.re_c[2,:], self.re_c_x0[2,:], self.re_c_x1[2,:]
                    else:
                        c, c_x0, c_x1 = self.intp_coef_2(xi, self.mesh.fs_x[p,m,:])

                    if self.mesh.fs_loc[p,m]==1:
                        for n in range(intp_n_size):
                            self.wei2_u0[p,ji[n],i,j] += c[n]*self.mesh.fs_n[p,m,0] * self.mesh.fs_l[p,m]
                            self.wei2_u1[p,ji[n],i,j] += c[n]*self.mesh.fs_n[p,m,1] * self.mesh.fs_l[p,m]
                    if self.mesh.fs_loc[p,m]==0:
                        self.r2[p,0,i,j] -= (self.mesh.fs_v0[p,m] * self.mesh.fs_n[p,m,0] + 
                                             self.mesh.fs_v1[p,m] * self.mesh.fs_n[p,m,1]) * self.mesh.fs_l[p,m]

                    # north face
                    intp_n_size = self.mesh.intp_n_size
                    ji = [0,1,2,3,4,5,6,7,8]
                    xi = self.mesh.intp_x[p,m,ji,:]; ii = self.mesh.intp_i[p,m,ji]
                    if (ii!=-1).all():
                        c, c_x0, c_x1 = self.re_c[3,:], self.re_c_x0[3,:], self.re_c_x1[3,:]
                    else:
                        c, c_x0, c_x1 = self.intp_coef_2(xi, self.mesh.fn_x[p,m,:])

                    if self.mesh.fn_loc[p,m]==1:
                        for n in range(intp_n_size):
                            self.wei2_u0[p,ji[n],i,j] += c[n]*self.mesh.fn_n[p,m,0] * self.mesh.fn_l[p,m]
                            self.wei2_u1[p,ji[n],i,j] += c[n]*self.mesh.fn_n[p,m,1] * self.mesh.fn_l[p,m]
                    if self.mesh.fn_loc[p,m]==0:
                        self.r2[p,0,i,j] -= (self.mesh.fn_v0[p,m] * self.mesh.fn_n[p,m,0] + 
                                             self.mesh.fn_v1[p,m] * self.mesh.fn_n[p,m,1]) * self.mesh.fn_l[p,m]
        '''
        print(self.wei2_u0[0,:,0,0])
        print(self.wei2_u1[0,:,0,0])
        '''

    def intp_coef_1(self, xi, x):
        tol = 1e-4

        intp_n_size = 2
        c = torch.zeros(intp_n_size)
        c_x0 = torch.zeros(intp_n_size); c_x1 = torch.zeros(intp_n_size)

        if abs(xi[0,1]-xi[1,1])<tol:
            c[0] = (xi[1,0]-x[0])/(xi[1,0]-xi[0,0])
            c[1] = (x[0]-xi[0,0])/(xi[1,0]-xi[0,0])
            c_x0[0] = 1/(xi[1,0]-xi[0,0])
            c_x0[1] = 1/(xi[1,0]-xi[0,0])

        if abs(xi[0,0]-xi[1,0])<tol:
            c[0] = (xi[1,1]-x[1])/(xi[1,1]-xi[0,1])
            c[1] = (x[1]-xi[0,1])/(xi[1,1]-xi[0,1])
            c_x1[0] = 1/(xi[1,1]-xi[0,1])
            c_x1[1] = 1/(xi[1,1]-xi[0,1])

        return c, c_x0, c_x1
    
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

    def to(self, device):
        self.device = device

        self.parm = self.parm.to(self.device)
        self.mask = self.mask.to(self.device)
        
        self.wei0_u0 = self.wei0_u0.to(self.device)
        self.wei0_u1 = self.wei0_u1.to(self.device)
        self.wei0_p = self.wei0_p.to(self.device)
        self.wei1_u0 = self.wei1_u0.to(self.device)
        self.wei1_u1 = self.wei1_u1.to(self.device)
        self.wei1_p = self.wei1_p.to(self.device)
        self.wei2_u0 = self.wei2_u0.to(self.device)
        self.wei2_u1 = self.wei2_u1.to(self.device)
        self.wei2_p = self.wei2_p.to(self.device)
        
        self.v0 = self.v0.to(self.device)
        self.v1 = self.v1.to(self.device)
        self.v2 = self.v2.to(self.device)

        self.r0 = self.r0.to(self.device)
        self.r1 = self.r1.to(self.device)
        self.r2 = self.r2.to(self.device)

class TeSet():
    def __init__(self, file_name, parm_size, nx, dtype):
        self.parm_size = parm_size
        self.nx = nx
        self.dtype = dtype
        
        data = pd.read_csv(file_name, header=None)
        data = np.array(data)
        data = torch.tensor(data, dtype=self.dtype)
        
        self.x0 = data[:,0:1].reshape(self.parm_size,1,self.nx[0],self.nx[1])
        self.x1 = data[:,1:2].reshape(self.parm_size,1,self.nx[0],self.nx[1])
        self.parm = data[:,2:3].reshape(self.parm_size,1,self.nx[0],self.nx[1])
        self.u0a = data[:,3:4].reshape(self.parm_size,1,self.nx[0],self.nx[1])
        self.u1a = data[:,4:5].reshape(self.parm_size,1,self.nx[0],self.nx[1])
        self.pa = data[:,5:6].reshape(self.parm_size,1,self.nx[0],self.nx[1])
        self.mask = data[:,6:7].reshape(self.parm_size,1,self.nx[0],self.nx[1])
        
        self.parm /= self.parm.max()

    def to(self, device):
        self.device = device

        self.x0 = self.x0.to(self.device)
        self.x1 = self.x1.to(self.device)
        self.parm = self.parm.to(self.device)
        self.u0a = self.u0a.to(self.device)
        self.u1a = self.u1a.to(self.device)
        self.pa = self.pa.to(self.device)
        self.mask = self.mask.to(self.device)
