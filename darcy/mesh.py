import torch
import numpy as np

class Mesh():
    def __init__(self, geo, bounds, nx):
        self.geo = geo
        self.bounds = bounds
        self.nx = nx
        
        self.dim = self.bounds.shape[0]
        self.hx = (self.bounds[:,1]-self.bounds[:,0])/self.nx
        self.xx = torch.linspace(self.bounds[0,0]+self.hx[0]/2,self.bounds[0,1]-self.hx[0]/2,self.nx[0])
        self.yy = torch.linspace(self.bounds[1,0]+self.hx[1]/2,self.bounds[1,1]-self.hx[1]/2,self.nx[1])
        
        """ cell center """
        print('Genrating mesh ...')
        print('Genrating cell center ...')
        self.c_size = self.nx[0]*self.nx[1]
        self.c_x = torch.zeros(self.c_size,self.dim)
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                m = i*self.nx[1] + j
                self.c_x[m,0] = self.bounds[0,0] + (i+0.5)*self.hx[0] 
                self.c_x[m,1] = self.bounds[1,0] + (j+0.5)*self.hx[1]
        
        self.c_loc = self.geo.location(self.c_x)
        
        """ cell corner
            w: west; e: east; s: south; n: north
        """
        print('Genrating cell corner ...')
        self.cws_x = self.c_x + torch.tensor([-0.5*self.hx[0],-0.5*self.hx[1]])
        self.cwn_x = self.c_x + torch.tensor([-0.5*self.hx[0], 0.5*self.hx[1]])
        self.ces_x = self.c_x + torch.tensor([ 0.5*self.hx[0],-0.5*self.hx[1]])
        self.cen_x = self.c_x + torch.tensor([ 0.5*self.hx[0], 0.5*self.hx[1]])
        self.cws_loc = self.geo.location(self.cws_x)
        self.cwn_loc = self.geo.location(self.cwn_x)
        self.ces_loc = self.geo.location(self.ces_x)
        self.cen_loc = self.geo.location(self.cen_x)

        """ neighbor cell """
        print('Genrating neighbor cell ...')
        self.nw_x = self.c_x + torch.tensor([-self.hx[0],0])
        self.ne_x = self.c_x + torch.tensor([ self.hx[0],0])
        self.ns_x = self.c_x + torch.tensor([0,-self.hx[1]])
        self.nn_x = self.c_x + torch.tensor([0, self.hx[1]])
        self.nw_loc = self.geo.location(self.nw_x)
        self.ne_loc = self.geo.location(self.ne_x)
        self.ns_loc = self.geo.location(self.ns_x)
        self.nn_loc = self.geo.location(self.nn_x)
        
        """ cell face """
        print('Genrating cell face ...')
        self.fw_st = torch.zeros(self.c_size); self.fw_ed = torch.ones(self.c_size)
        self.fe_st = torch.zeros(self.c_size); self.fe_ed = torch.ones(self.c_size)
        self.fs_st = torch.zeros(self.c_size); self.fs_ed = torch.ones(self.c_size)
        self.fn_st = torch.zeros(self.c_size); self.fn_ed = torch.ones(self.c_size)
        self.fw_l = torch.zeros(self.c_size); self.fe_l = torch.zeros(self.c_size)
        self.fs_l = torch.zeros(self.c_size); self.fn_l = torch.zeros(self.c_size)
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                m = i*self.nx[1] + j
                if self.c_loc[m]!=1:
                    continue
                
                if self.cws_loc[m]==-1 and self.cwn_loc[m]==1:
                    tmp_x = self.geo.intersection(self.cws_x[m:m+1,:],self.cwn_x[m:m+1,:])
                    self.fw_st[m] = (tmp_x[0,1]-self.cws_x[m,1])/self.hx[1]
                if self.cws_loc[m]==1 and self.cwn_loc[m]==-1:
                    tmp_x = self.geo.intersection(self.cws_x[m:m+1,:],self.cwn_x[m:m+1,:])
                    self.fw_ed[m] = (tmp_x[0,1]-self.cws_x[m,1])/self.hx[1]
                if self.cws_loc[m]==-1 and self.cwn_loc[m]==-1:
                    self.fw_st[m] = 0.0; self.fw_ed[m] = 0.0
                
                if self.ces_loc[m]==-1 and self.cen_loc[m]==1:
                    tmp_x = self.geo.intersection(self.ces_x[m:m+1,:],self.cen_x[m:m+1,:])
                    self.fe_st[m] = (tmp_x[0,1]-self.ces_x[m,1])/self.hx[1]
                if self.ces_loc[m]==1 and self.cen_loc[m]==-1:
                    tmp_x = self.geo.intersection(self.ces_x[m:m+1,:],self.cen_x[m:m+1,:])
                    self.fe_ed[m] = (tmp_x[0,1]-self.ces_x[m,1])/self.hx[1]
                if self.ces_loc[m]==-1 and self.cen_loc[m]==-1:
                    self.fe_st[m] = 0.0; self.fe_ed[m] = 0.0
                
                if self.cws_loc[m]==-1 and self.ces_loc[m]==1:
                    tmp_x = self.geo.intersection(self.cws_x[m:m+1,:],self.ces_x[m:m+1,:])
                    self.fs_st[m] = (tmp_x[0,0]-self.cws_x[m,0])/self.hx[0]
                if self.cws_loc[m]==1 and self.ces_loc[m]==-1:
                    tmp_x = self.geo.intersection(self.cws_x[m:m+1,:],self.ces_x[m:m+1,:])
                    self.fs_ed[m] = (tmp_x[0,0]-self.cws_x[m,0])/self.hx[0]
                if self.cws_loc[m]==-1 and self.ces_loc[m]==-1:
                    self.fs_st[m] = 0.0; self.fs_ed[m] = 0.0
                
                if self.cwn_loc[m]==-1 and self.cen_loc[m]==1:
                    tmp_x = self.geo.intersection(self.cwn_x[m:m+1,:],self.cen_x[m:m+1,:])
                    self.fn_st[m] = (tmp_x[0,0]-self.cwn_x[m,0])/self.hx[0]
                if self.cwn_loc[m]==1 and self.cen_loc[m]==-1:
                    tmp_x = self.geo.intersection(self.cwn_x[m:m+1,:],self.cen_x[m:m+1,:])
                    self.fn_ed[m] = (tmp_x[0,0]-self.cwn_x[m,0])/self.hx[0]
                if self.cwn_loc[m]==-1 and self.cen_loc[m]==-1:
                    self.fn_st[m] = 0.0; self.fn_ed[m] = 0.0
                
                self.fw_l[m] = (self.fw_ed[m]-self.fw_st[m]) * self.hx[1]
                self.fe_l[m] = (self.fe_ed[m]-self.fe_st[m]) * self.hx[1]
                self.fs_l[m] = (self.fs_ed[m]-self.fs_st[m]) * self.hx[0]
                self.fn_l[m] = (self.fn_ed[m]-self.fn_st[m]) * self.hx[0]
                if self.cws_loc[m]==-1 and self.cwn_loc[m]==-1:
                    tmp_x0 = self.cws_x[m:m+1,:] + self.fs_st[m]*(self.ces_x[m:m+1,:]-self.cws_x[m:m+1,:])
                    tmp_x1 = self.cwn_x[m:m+1,:] + self.fn_st[m]*(self.cen_x[m:m+1,:]-self.cwn_x[m:m+1,:])
                    self.fw_l[m] = (((tmp_x0-tmp_x1)**2).sum())**0.5
                
                if self.ces_loc[m]==-1 and self.cen_loc[m]==-1:
                    tmp_x0 = self.cws_x[m:m+1,:] + self.fs_ed[m]*(self.ces_x[m:m+1,:]-self.cws_x[m:m+1,:])
                    tmp_x1 = self.cwn_x[m:m+1,:] + self.fn_ed[m]*(self.cen_x[m:m+1,:]-self.cwn_x[m:m+1,:])
                    self.fe_l[m] = (((tmp_x0-tmp_x1)**2).sum())**0.5
                
                if self.cws_loc[m]==-1 and self.ces_loc[m]==-1:
                    tmp_x0 = self.cws_x[m:m+1,:] + self.fw_st[m]*(self.cwn_x[m:m+1,:]-self.cws_x[m:m+1,:])
                    tmp_x1 = self.ces_x[m:m+1,:] + self.fe_st[m]*(self.cen_x[m:m+1,:]-self.ces_x[m:m+1,:])
                    self.fs_l[m] = (((tmp_x0-tmp_x1)**2).sum())**0.5
                
                if self.cwn_loc[m]==-1 and self.cen_loc[m]==-1:
                    tmp_x0 = self.cws_x[m:m+1,:] + self.fw_ed[m]*(self.cwn_x[m:m+1,:]-self.cws_x[m:m+1,:])
                    tmp_x1 = self.ces_x[m:m+1,:] + self.fe_ed[m]*(self.cen_x[m:m+1,:]-self.ces_x[m:m+1,:])
                    self.fn_l[m] = (((tmp_x0-tmp_x1)**2).sum())**0.5
            
        self.fw_x = torch.zeros(self.c_size,self.dim); self.fe_x = torch.zeros(self.c_size,self.dim)
        self.fs_x = torch.zeros(self.c_size,self.dim); self.fn_x = torch.zeros(self.c_size,self.dim)
        self.fw_n = torch.zeros(self.c_size,self.dim); self.fe_n = torch.zeros(self.c_size,self.dim)
        self.fs_n = torch.zeros(self.c_size,self.dim); self.fn_n = torch.zeros(self.c_size,self.dim)
        self.fw_loc = torch.zeros(self.c_size); self.fe_loc = torch.zeros(self.c_size)
        self.fs_loc = torch.zeros(self.c_size); self.fn_loc = torch.zeros(self.c_size)
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                m = i*self.nx[1] + j
                if self.c_loc[m]!=1:
                    continue
                    
                if self.cws_loc[m]==-1 and self.cwn_loc[m]==-1:
                    tmp_x0 = self.cws_x[m:m+1,:] + self.fs_st[m]*(self.ces_x[m:m+1,:]-self.cws_x[m:m+1,:])
                    tmp_x1 = self.cwn_x[m:m+1,:] + self.fn_st[m]*(self.cen_x[m:m+1,:]-self.cwn_x[m:m+1,:])
                    self.fw_loc[m] = 0
                else:
                    tmp_x0 = self.cws_x[m:m+1,:] + self.fw_st[m]*(self.cwn_x[m:m+1,:]-self.cws_x[m:m+1,:])
                    tmp_x1 = self.cws_x[m:m+1,:] + self.fw_ed[m]*(self.cwn_x[m:m+1,:]-self.cws_x[m:m+1,:])
                    self.fw_loc[m] = self.geo.location(0.5*(tmp_x0+tmp_x1))
                
                self.fw_x[m:m+1,:] = 0.5*(tmp_x0+tmp_x1)
                self.fw_n[m,:] = torch.tensor([-(tmp_x1[0,1]-tmp_x0[0,1]),tmp_x1[0,0]-tmp_x0[0,0]])
                self.fw_n[m,:] = self.fw_n[m,:]/((self.fw_n[m,:]**2).sum())**0.5
                
                if self.ces_loc[m]==-1 and self.cen_loc[m]==-1:
                    tmp_x0 = self.cws_x[m:m+1,:] + self.fs_ed[m]*(self.ces_x[m:m+1,:]-self.cws_x[m:m+1,:])
                    tmp_x1 = self.cwn_x[m:m+1,:] + self.fn_ed[m]*(self.cen_x[m:m+1,:]-self.cwn_x[m:m+1,:])
                    self.fe_loc[m] = 0
                else:
                    tmp_x0 = self.ces_x[m:m+1,:] + self.fe_st[m]*(self.cen_x[m:m+1,:]-self.ces_x[m:m+1,:])
                    tmp_x1 = self.ces_x[m:m+1,:] + self.fe_ed[m]*(self.cen_x[m:m+1,:]-self.ces_x[m:m+1,:])
                    self.fe_loc[m] = self.geo.location(0.5*(tmp_x0+tmp_x1))
                
                self.fe_x[m:m+1,:] = 0.5*(tmp_x0+tmp_x1)
                self.fe_n[m,:] = torch.tensor([tmp_x1[0,1]-tmp_x0[0,1],-(tmp_x1[0,0]-tmp_x0[0,0])])
                self.fe_n[m,:] = self.fe_n[m,:]/((self.fe_n[m,:]**2).sum())**0.5

                if self.cws_loc[m]==-1 and self.ces_loc[m]==-1:
                    tmp_x0 = self.cws_x[m:m+1,:] + self.fw_st[m]*(self.cwn_x[m:m+1,:]-self.cws_x[m:m+1,:])
                    tmp_x1 = self.ces_x[m:m+1,:] + self.fe_st[m]*(self.cen_x[m:m+1,:]-self.ces_x[m:m+1,:])
                    self.fs_loc[m] = 0
                else:
                    tmp_x0 = self.cws_x[m:m+1,:] + self.fs_st[m]*(self.ces_x[m:m+1,:]-self.cws_x[m:m+1,:])
                    tmp_x1 = self.cws_x[m:m+1,:] + self.fs_ed[m]*(self.ces_x[m:m+1,:]-self.cws_x[m:m+1,:])
                    self.fs_loc[m] = self.geo.location(0.5*(tmp_x0+tmp_x1))
                
                self.fs_x[m:m+1,:] = 0.5*(tmp_x0+tmp_x1)
                self.fs_n[m,:] = torch.tensor([tmp_x1[0,1]-tmp_x0[0,1],-(tmp_x1[0,0]-tmp_x0[0,0])])
                self.fs_n[m,:] = self.fs_n[m,:]/((self.fs_n[m,:]**2).sum())**0.5
                
                if self.cwn_loc[m]==-1 and self.cen_loc[m]==-1:
                    tmp_x0 = self.cws_x[m:m+1,:] + self.fw_ed[m]*(self.cwn_x[m:m+1,:]-self.cws_x[m:m+1,:])
                    tmp_x1 = self.ces_x[m:m+1,:] + self.fe_ed[m]*(self.cen_x[m:m+1,:]-self.ces_x[m:m+1,:])
                    self.fn_loc[m] = 0
                else:
                    tmp_x0 = self.cwn_x[m:m+1,:] + self.fn_st[m]*(self.cen_x[m:m+1,:]-self.cwn_x[m:m+1,:])
                    tmp_x1 = self.cwn_x[m:m+1,:] + self.fn_ed[m]*(self.cen_x[m:m+1,:]-self.cwn_x[m:m+1,:])
                    self.fn_loc[m] = self.geo.location(0.5*(tmp_x0+tmp_x1))
                
                self.fn_x[m:m+1,:] = 0.5*(tmp_x0+tmp_x1)
                self.fn_n[m,:] = torch.tensor([-(tmp_x1[0,1]-tmp_x0[0,1]),tmp_x1[0,0]-tmp_x0[0,0]])
                self.fn_n[m,:] = self.fn_n[m,:]/((self.fn_n[m,:]**2).sum())**0.5    
        
        self.fws_l = torch.zeros(self.c_size); self.fwn_l = torch.zeros(self.c_size)
        self.fes_l = torch.zeros(self.c_size); self.fen_l = torch.zeros(self.c_size)
        self.fws_x = torch.zeros(self.c_size,self.dim); self.fwn_x = torch.zeros(self.c_size,self.dim)
        self.fes_x = torch.zeros(self.c_size,self.dim); self.fen_x = torch.zeros(self.c_size,self.dim)
        self.fws_n = torch.zeros(self.c_size,self.dim); self.fwn_n = torch.zeros(self.c_size,self.dim)
        self.fes_n = torch.zeros(self.c_size,self.dim); self.fen_n = torch.zeros(self.c_size,self.dim)
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                m = i*self.nx[1] + j
                if self.c_loc[m]!=1:
                    continue

                if self.cws_loc[m]==-1 and self.cwn_loc[m]==1 and self.ces_loc[m]==1:
                    tmp = torch.tensor([self.fs_st[m]*self.hx[0],self.fw_st[m]*self.hx[1]])
                    self.fws_l[m] = ((tmp**2).sum())**0.5
                    self.fws_x[m:m+1,:] = self.cws_x[m:m+1,:] + 0.5*torch.tensor([tmp[0],tmp[1]])
                    self.fws_n[m,:] = torch.tensor([-tmp[1],-tmp[0]])
                    self.fws_n[m,:] = self.fws_n[m,:]/(((self.fws_n[m,:]**2).sum())**0.5)
                
                if self.cwn_loc[m]==-1 and self.cws_loc[m]==1 and self.cen_loc[m]==1:
                    tmp = torch.tensor([self.fn_st[m]*self.hx[0],(1-self.fw_ed[m])*self.hx[1]])
                    self.fwn_l[m] = (sum(tmp**2))**0.5
                    self.fwn_x[m:m+1,:] = self.cwn_x[m:m+1,:] + 0.5*torch.tensor([tmp[0],-tmp[1]])
                    self.fwn_n[m,:] = torch.tensor([-tmp[1],tmp[0]])
                    self.fwn_n[m,:] = self.fwn_n[m,:]/(((self.fwn_n[m,:]**2).sum())**0.5)
                
                if self.ces_loc[m]==-1 and self.cen_loc[m]==1 and self.cws_loc[m]==1:
                    tmp = torch.tensor([(1-self.fs_ed[m])*self.hx[0],self.fe_st[m]*self.hx[1]])
                    self.fes_l[m] = ((tmp**2).sum())**0.5
                    self.fes_x[m:m+1,:] = self.ces_x[m:m+1,:] + 0.5*torch.tensor([-tmp[0],tmp[1]])
                    self.fes_n[m,:] = torch.tensor([tmp[1],-tmp[0]])
                    self.fes_n[m,:] = self.fes_n[m,:]/(((self.fes_n[m,:]**2).sum())**0.5)
                
                if self.cen_loc[m]==-1 and self.ces_loc[m]==1 and self.cwn_loc[m]==1:
                    tmp = torch.tensor([(1-self.fn_ed[m])*self.hx[0],(1-self.fe_ed[m])*self.hx[1]])
                    self.fen_l[m] = ((tmp**2).sum())**0.5
                    self.fen_x[m:m+1,:] = self.cen_x[m:m+1,:] + 0.5*torch.tensor([-tmp[0],-tmp[1]])
                    self.fen_n[m,:] = torch.tensor([tmp[1],tmp[0]])
                    self.fen_n[m,:] = self.fen_n[m,:]/(((self.fen_n[m,:]**2).sum())**0.5)
        
        """ cell area """
        print('Genrating cell area ...')
        self.c_a = torch.zeros(self.c_size)
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                m = i*self.nx[1] + j
                if self.c_loc[m]!=1:
                    continue

                self.c_a[m] = self.hx[0]*self.hx[1]
                if self.cws_loc[m]==-1 and self.cwn_loc[m]==1 and self.ces_loc[m]==1:
                    self.c_a[m] = self.c_a[m] - 0.5*(self.fw_st[m]*self.hx[1])*(self.fs_st[m]*self.hx[0])
                if self.cwn_loc[m]==-1 and self.cws_loc[m]==1 and self.cen_loc[m]==1:
                    self.c_a[m] = self.c_a[m] - 0.5*((1.0-self.fw_ed[m])*self.hx[1])*(self.fn_st[m]*self.hx[0])
                if self.ces_loc[m]==-1 and self.cen_loc[m]==1 and self.cws_loc[m]==1:
                    self.c_a[m] = self.c_a[m] - 0.5*(self.fe_st[m]*self.hx[1])*((1.0-self.fs_ed[m])*self.hx[0])
                if self.cen_loc[m]==-1 and self.ces_loc[m]==1 and self.cwn_loc[m]==1:
                    self.c_a[m] = self.c_a[m] - 0.5*((1.0-self.fe_ed[m])*self.hx[1])*((1.0-self.fn_ed[m])*self.hx[0])
                if self.cws_loc[m]==-1 and self.cwn_loc[m]==-1:
                    self.c_a[m] = self.c_a[m] - 0.5*self.hx[1]*(self.fs_st[m]+self.fn_st[m])*self.hx[0]
                if self.ces_loc[m]==-1 and self.cen_loc[m]==-1:
                    self.c_a[m] = self.c_a[m] - 0.5*self.hx[1]*(1.0-self.fs_ed[m]+1.0-self.fn_ed[m])*self.hx[0]
                if self.cws_loc[m]==-1 and self.ces_loc[m]==-1:
                    self.c_a[m] = self.c_a[m] - 0.5*(self.fw_st[m]+self.fe_st[m])*self.hx[1]*self.hx[0]
                if self.cwn_loc[m]==-1 and self.cen_loc[m]==-1:
                    self.c_a[m] = self.c_a[m] - 0.5*(1.0-self.fw_ed[m]+1.0-self.fe_ed[m])*self.hx[1]*self.hx[0]

        """ interpolation node """
        print('Genrating interpolation node ...')
        self.intp_n_size = 3**2
        self.intp_x = torch.zeros(self.c_size,self.intp_n_size,self.dim)
        self.intp_i = torch.zeros(self.c_size,self.intp_n_size).long()
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                m = i*self.nx[1] + j
                if self.c_loc[m]!=1:
                    continue
                
                self.intp_x[m,:,:], self.intp_i[m,:] = self.intp_node([i,j])
    
    def intp_node(self, idx):
        intp_n_size = 3**2
        xi = torch.zeros(intp_n_size,self.dim)
        ii = torch.zeros(intp_n_size).long()

        dir = [[-1,-1],[-1, 0],[-1, 1], [0,-1],[0, 0],[0, 1], [1,-1],[1, 0],[1, 1]]
        dir = torch.tensor(dir).reshape(intp_n_size,self.dim)

        # regular point
        ix = idx[0]+dir[:,0]; iy = idx[1]+dir[:,1]
        m = ix*self.nx[1] + iy
        idx1 = ((ix>=0) & (ix<self.nx[0]) & (iy>=0) & (iy<self.nx[1]))
        idx1[idx1==True] = (self.c_loc[m[idx1]]==1)
        
        xi[idx1,:] = self.c_x[m[idx1],:]
        ii[idx1] = m[idx1]
        
        # irregular point
        idx1 = ~idx1
        m = idx[0]*self.nx[1] + idx[1]
        x1 = self.c_x[m,:]
        x1 = x1.reshape(1,self.dim).repeat(intp_n_size,1)
        x2 = x1 + torch.tensor([self.hx[0],self.hx[1]]) * dir

        xi_tmp = self.geo.intersection(x1[idx1,:], x2[idx1,:])

        xi[idx1,:] = xi_tmp
        ii[idx1] = -1

        return xi, ii