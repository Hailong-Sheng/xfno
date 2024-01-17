import torch

class Mesh():
    def __init__(self, geo, bounds, nx):
        self.geo = geo
        self.bounds = bounds
        self.nx = nx

        self.parm_size = geo.parm_size
        self.dim = self.bounds.shape[0]
        self.hx = (self.bounds[:,1]-self.bounds[:,0])/self.nx
        self.xx = torch.linspace(self.bounds[0,0]+self.hx[0]/2,self.bounds[0,1]-self.hx[0]/2,self.nx[0])
        self.yy = torch.linspace(self.bounds[1,0]+self.hx[1]/2,self.bounds[1,1]-self.hx[1]/2,self.nx[1])
        
        """ cell center """
        print('Genrating mesh ...')
        print('Genrating cell center ...')
        self.c_size = self.nx[0]*self.nx[1]
        self.c_x = torch.zeros(self.parm_size,self.c_size,self.dim)
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                m = i*self.nx[1] + j
                self.c_x[:,m,0] = self.bounds[0,0] + (i+0.5)*self.hx[0] 
                self.c_x[:,m,1] = self.bounds[1,0] + (j+0.5)*self.hx[1]
        
        self.c_loc = torch.zeros(self.parm_size,self.c_size)
        for p in range(self.parm_size):
            self.c_loc[p,:] = self.geo.location(self.c_x[p,:,:], p)
        
        """ cell corner
            w: west; e: east; s: south; n: north
        """
        print('Genrating cell corner ...')
        self.cws_x = self.c_x + torch.tensor([-0.5*self.hx[0],-0.5*self.hx[1]])
        self.cwn_x = self.c_x + torch.tensor([-0.5*self.hx[0], 0.5*self.hx[1]])
        self.ces_x = self.c_x + torch.tensor([ 0.5*self.hx[0],-0.5*self.hx[1]])
        self.cen_x = self.c_x + torch.tensor([ 0.5*self.hx[0], 0.5*self.hx[1]])
        self.cws_loc = torch.zeros(self.parm_size,self.c_size)
        self.cwn_loc = torch.zeros(self.parm_size,self.c_size)
        self.ces_loc = torch.zeros(self.parm_size,self.c_size)
        self.cen_loc = torch.zeros(self.parm_size,self.c_size)
        for p in range(self.parm_size):
            self.cws_loc[p,:] = self.geo.location(self.cws_x[p,:,:], p)
            self.cwn_loc[p,:] = self.geo.location(self.cwn_x[p,:,:], p)
            self.ces_loc[p,:] = self.geo.location(self.ces_x[p,:,:], p)
            self.cen_loc[p,:] = self.geo.location(self.cen_x[p,:,:], p)
        
        """ neighbor cell """
        print('Genrating neighbor cell ...')
        self.nw_x = self.c_x + torch.tensor([-self.hx[0],0])
        self.ne_x = self.c_x + torch.tensor([ self.hx[0],0])
        self.ns_x = self.c_x + torch.tensor([0,-self.hx[1]])
        self.nn_x = self.c_x + torch.tensor([0, self.hx[1]])
        self.nw_loc = torch.zeros(self.parm_size,self.c_size)
        self.ne_loc = torch.zeros(self.parm_size,self.c_size)
        self.ns_loc = torch.zeros(self.parm_size,self.c_size)
        self.nn_loc = torch.zeros(self.parm_size,self.c_size)
        for p in range(self.parm_size):
            self.nw_loc[p,:] = self.geo.location(self.nw_x[p,:,:], p)
            self.ne_loc[p,:] = self.geo.location(self.ne_x[p,:,:], p)
            self.ns_loc[p,:] = self.geo.location(self.ns_x[p,:,:], p)
            self.nn_loc[p,:] = self.geo.location(self.nn_x[p,:,:], p)
        
        """ cell face """
        print('Genrating cell face ...')
        self.fw_st = torch.zeros(self.parm_size,self.c_size); self.fw_ed = torch.ones(self.parm_size,self.c_size)
        self.fe_st = torch.zeros(self.parm_size,self.c_size); self.fe_ed = torch.ones(self.parm_size,self.c_size)
        self.fs_st = torch.zeros(self.parm_size,self.c_size); self.fs_ed = torch.ones(self.parm_size,self.c_size)
        self.fn_st = torch.zeros(self.parm_size,self.c_size); self.fn_ed = torch.ones(self.parm_size,self.c_size)
        self.fw_l = torch.zeros(self.parm_size,self.c_size); self.fe_l = torch.zeros(self.parm_size,self.c_size)
        self.fs_l = torch.zeros(self.parm_size,self.c_size); self.fn_l = torch.zeros(self.parm_size,self.c_size)
        for p in range(self.parm_size):
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    m = i*self.nx[1] + j
                    if self.c_loc[p,m]!=1:
                        continue
                    
                    if self.cws_loc[p,m]==-1 and self.cwn_loc[p,m]==1:
                        tmp_x = self.geo.intersection(self.cws_x[p,m:m+1,:],self.cwn_x[p,m:m+1,:],p)
                        self.fw_st[p,m] = (tmp_x[0,1]-self.cws_x[p,m,1])/self.hx[1]
                    if self.cws_loc[p,m]==1 and self.cwn_loc[p,m]==-1:
                        tmp_x = self.geo.intersection(self.cws_x[p,m:m+1,:],self.cwn_x[p,m:m+1,:],p)
                        self.fw_ed[p,m] = (tmp_x[0,1]-self.cws_x[p,m,1])/self.hx[1]
                    if self.cws_loc[p,m]==-1 and self.cwn_loc[p,m]==-1:
                        self.fw_st[p,m] = 0.0; self.fw_ed[p,m] = 0.0
                    
                    if self.ces_loc[p,m]==-1 and self.cen_loc[p,m]==1:
                        tmp_x = self.geo.intersection(self.ces_x[p,m:m+1,:],self.cen_x[p,m:m+1,:],p)
                        self.fe_st[p,m] = (tmp_x[0,1]-self.ces_x[p,m,1])/self.hx[1]
                    if self.ces_loc[p,m]==1 and self.cen_loc[p,m]==-1:
                        tmp_x = self.geo.intersection(self.ces_x[p,m:m+1,:],self.cen_x[p,m:m+1,:],p)
                        self.fe_ed[p,m] = (tmp_x[0,1]-self.ces_x[p,m,1])/self.hx[1]
                    if self.ces_loc[p,m]==-1 and self.cen_loc[p,m]==-1:
                        self.fe_st[p,m] = 0.0; self.fe_ed[p,m] = 0.0
                    
                    if self.cws_loc[p,m]==-1 and self.ces_loc[p,m]==1:
                        tmp_x = self.geo.intersection(self.cws_x[p,m:m+1,:],self.ces_x[p,m:m+1,:],p)
                        self.fs_st[p,m] = (tmp_x[0,0]-self.cws_x[p,m,0])/self.hx[0]
                    if self.cws_loc[p,m]==1 and self.ces_loc[p,m]==-1:
                        tmp_x = self.geo.intersection(self.cws_x[p,m:m+1,:],self.ces_x[p,m:m+1,:],p)
                        self.fs_ed[p,m] = (tmp_x[0,0]-self.cws_x[p,m,0])/self.hx[0]
                    if self.cws_loc[p,m]==-1 and self.ces_loc[p,m]==-1:
                        self.fs_st[p,m] = 0.0; self.fs_ed[p,m] = 0.0
                    
                    if self.cwn_loc[p,m]==-1 and self.cen_loc[p,m]==1:
                        tmp_x = self.geo.intersection(self.cwn_x[p,m:m+1,:],self.cen_x[p,m:m+1,:],p)
                        self.fn_st[p,m] = (tmp_x[0,0]-self.cwn_x[p,m,0])/self.hx[0]
                    
                    if self.cwn_loc[p,m]==1 and self.cen_loc[p,m]==-1:
                        tmp_x = self.geo.intersection(self.cwn_x[p,m:m+1,:],self.cen_x[p,m:m+1,:],p)
                        self.fn_ed[p,m] = (tmp_x[0,0]-self.cwn_x[p,m,0])/self.hx[0]
                    
                    if self.cwn_loc[p,m]==-1 and self.cen_loc[p,m]==-1:
                        self.fn_st[p,m] = 0.0; self.fn_ed[p,m] = 0.0
                    
                    self.fw_l[p,m] = (self.fw_ed[p,m]-self.fw_st[p,m]) * self.hx[1]
                    self.fe_l[p,m] = (self.fe_ed[p,m]-self.fe_st[p,m]) * self.hx[1]
                    self.fs_l[p,m] = (self.fs_ed[p,m]-self.fs_st[p,m]) * self.hx[0]
                    self.fn_l[p,m] = (self.fn_ed[p,m]-self.fn_st[p,m]) * self.hx[0]
                    if self.cws_loc[p,m]==-1 and self.cwn_loc[p,m]==-1:
                        tmp_x0 = self.cws_x[p,m:m+1,:] + self.fs_st[p,m]*(self.ces_x[p,m:m+1,:]-self.cws_x[p,m:m+1,:])
                        tmp_x1 = self.cwn_x[p,m:m+1,:] + self.fn_st[p,m]*(self.cen_x[p,m:m+1,:]-self.cwn_x[p,m:m+1,:])
                        self.fw_l[p,m] = (((tmp_x0-tmp_x1)**2).sum())**0.5
                    
                    if self.ces_loc[p,m]==-1 and self.cen_loc[p,m]==-1:
                        tmp_x0 = self.cws_x[p,m:m+1,:] + self.fs_ed[p,m]*(self.ces_x[p,m:m+1,:]-self.cws_x[p,m:m+1,:])
                        tmp_x1 = self.cwn_x[p,m:m+1,:] + self.fn_ed[p,m]*(self.cen_x[p,m:m+1,:]-self.cwn_x[p,m:m+1,:])
                        self.fe_l[p,m] = (((tmp_x0-tmp_x1)**2).sum())**0.5
                    
                    if self.cws_loc[p,m]==-1 and self.ces_loc[p,m]==-1:
                        tmp_x0 = self.cws_x[p,m:m+1,:] + self.fw_st[p,m]*(self.cwn_x[p,m:m+1,:]-self.cws_x[p,m:m+1,:])
                        tmp_x1 = self.ces_x[p,m:m+1,:] + self.fe_st[p,m]*(self.cen_x[p,m:m+1,:]-self.ces_x[p,m:m+1,:])
                        self.fs_l[p,m] = (((tmp_x0-tmp_x1)**2).sum())**0.5
                    
                    if self.cwn_loc[p,m]==-1 and self.cen_loc[p,m]==-1:
                        tmp_x0 = self.cws_x[p,m:m+1,:] + self.fw_ed[p,m]*(self.cwn_x[p,m:m+1,:]-self.cws_x[p,m:m+1,:])
                        tmp_x1 = self.ces_x[p,m:m+1,:] + self.fe_ed[p,m]*(self.cen_x[p,m:m+1,:]-self.ces_x[p,m:m+1,:])
                        self.fn_l[p,m] = (((tmp_x0-tmp_x1)**2).sum())**0.5
            
        self.fw_x = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fe_x = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fs_x = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fn_x = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fw_n = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fe_n = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fs_n = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fn_n = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fw_loc = torch.zeros(self.parm_size,self.c_size); self.fe_loc = torch.zeros(self.parm_size,self.c_size)
        self.fs_loc = torch.zeros(self.parm_size,self.c_size); self.fn_loc = torch.zeros(self.parm_size,self.c_size)
        for p in range(self.parm_size):
            print('for parameter: c = [{:.2f},{:.2f}]'.format(self.geo.center[p,0],self.geo.center[p,1]))
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    m = i*self.nx[1] + j
                    if self.c_loc[p,m]!=1:
                        continue
                    
                    if self.cws_loc[p,m]==-1 and self.cwn_loc[p,m]==-1:
                        tmp_x0 = self.cws_x[p,m:m+1,:] + self.fs_st[p,m]*(self.ces_x[p,m:m+1,:]-self.cws_x[p,m:m+1,:])
                        tmp_x1 = self.cwn_x[p,m:m+1,:] + self.fn_st[p,m]*(self.cen_x[p,m:m+1,:]-self.cwn_x[p,m:m+1,:])
                        self.fw_loc[p,m] = 0
                    else:
                        tmp_x0 = self.cws_x[p,m:m+1,:] + self.fw_st[p,m]*(self.cwn_x[p,m:m+1,:]-self.cws_x[p,m:m+1,:])
                        tmp_x1 = self.cws_x[p,m:m+1,:] + self.fw_ed[p,m]*(self.cwn_x[p,m:m+1,:]-self.cws_x[p,m:m+1,:])
                        self.fw_loc[p,m] = self.geo.location(0.5*(tmp_x0+tmp_x1), p)
                    
                    self.fw_x[p,m:m+1,:] = 0.5*(tmp_x0+tmp_x1)
                    self.fw_n[p,m,:] = torch.tensor([-(tmp_x1[0,1]-tmp_x0[0,1]),tmp_x1[0,0]-tmp_x0[0,0]])
                    self.fw_n[p,m,:] = self.fw_n[p,m,:]/((self.fw_n[p,m,:]**2).sum())**0.5
                    
                    if self.ces_loc[p,m]==-1 and self.cen_loc[p,m]==-1:
                        tmp_x0 = self.cws_x[p,m:m+1,:] + self.fs_ed[p,m]*(self.ces_x[p,m:m+1,:]-self.cws_x[p,m:m+1,:])
                        tmp_x1 = self.cwn_x[p,m:m+1,:] + self.fn_ed[p,m]*(self.cen_x[p,m:m+1,:]-self.cwn_x[p,m:m+1,:])
                        self.fe_loc[p,m] = 0
                    else:
                        tmp_x0 = self.ces_x[p,m:m+1,:] + self.fe_st[p,m]*(self.cen_x[p,m:m+1,:]-self.ces_x[p,m:m+1,:])
                        tmp_x1 = self.ces_x[p,m:m+1,:] + self.fe_ed[p,m]*(self.cen_x[p,m:m+1,:]-self.ces_x[p,m:m+1,:])
                        self.fe_loc[p,m] = self.geo.location(0.5*(tmp_x0+tmp_x1), p)
                    
                    self.fe_x[p,m:m+1,:] = 0.5*(tmp_x0+tmp_x1)
                    self.fe_n[p,m,:] = torch.tensor([tmp_x1[0,1]-tmp_x0[0,1],-(tmp_x1[0,0]-tmp_x0[0,0])])
                    self.fe_n[p,m,:] = self.fe_n[p,m,:]/((self.fe_n[p,m,:]**2).sum())**0.5

                    if self.cws_loc[p,m]==-1 and self.ces_loc[p,m]==-1:
                        tmp_x0 = self.cws_x[p,m:m+1,:] + self.fw_st[p,m]*(self.cwn_x[p,m:m+1,:]-self.cws_x[p,m:m+1,:])
                        tmp_x1 = self.ces_x[p,m:m+1,:] + self.fe_st[p,m]*(self.cen_x[p,m:m+1,:]-self.ces_x[p,m:m+1,:])
                        self.fs_loc[p,m] = 0
                    else:
                        tmp_x0 = self.cws_x[p,m:m+1,:] + self.fs_st[p,m]*(self.ces_x[p,m:m+1,:]-self.cws_x[p,m:m+1,:])
                        tmp_x1 = self.cws_x[p,m:m+1,:] + self.fs_ed[p,m]*(self.ces_x[p,m:m+1,:]-self.cws_x[p,m:m+1,:])
                        self.fs_loc[p,m] = self.geo.location(0.5*(tmp_x0+tmp_x1), p)
                    
                    self.fs_x[p,m:m+1,:] = 0.5*(tmp_x0+tmp_x1)
                    self.fs_n[p,m,:] = torch.tensor([tmp_x1[0,1]-tmp_x0[0,1],-(tmp_x1[0,0]-tmp_x0[0,0])])
                    self.fs_n[p,m,:] = self.fs_n[p,m,:]/((self.fs_n[p,m,:]**2).sum())**0.5
                    
                    if self.cwn_loc[p,m]==-1 and self.cen_loc[p,m]==-1:
                        tmp_x0 = self.cws_x[p,m:m+1,:] + self.fw_ed[p,m]*(self.cwn_x[p,m:m+1,:]-self.cws_x[p,m:m+1,:])
                        tmp_x1 = self.ces_x[p,m:m+1,:] + self.fe_ed[p,m]*(self.cen_x[p,m:m+1,:]-self.ces_x[p,m:m+1,:])
                        self.fn_loc[p,m] = 0
                    else:
                        tmp_x0 = self.cwn_x[p,m:m+1,:] + self.fn_st[p,m]*(self.cen_x[p,m:m+1,:]-self.cwn_x[p,m:m+1,:])
                        tmp_x1 = self.cwn_x[p,m:m+1,:] + self.fn_ed[p,m]*(self.cen_x[p,m:m+1,:]-self.cwn_x[p,m:m+1,:])
                        self.fn_loc[p,m] = self.geo.location(0.5*(tmp_x0+tmp_x1), p)
                    
                    self.fn_x[p,m:m+1,:] = 0.5*(tmp_x0+tmp_x1)
                    self.fn_n[p,m,:] = torch.tensor([-(tmp_x1[0,1]-tmp_x0[0,1]),tmp_x1[0,0]-tmp_x0[0,0]])
                    self.fn_n[p,m,:] = self.fn_n[p,m,:]/((self.fn_n[p,m,:]**2).sum())**0.5    
        
        self.fws_l = torch.zeros(self.parm_size,self.c_size); self.fwn_l = torch.zeros(self.parm_size,self.c_size)
        self.fes_l = torch.zeros(self.parm_size,self.c_size); self.fen_l = torch.zeros(self.parm_size,self.c_size)
        self.fws_x = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fwn_x = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fes_x = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fen_x = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fws_n = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fwn_n = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fes_n = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fen_n = torch.zeros(self.parm_size,self.c_size,self.dim)
        for p in range(self.parm_size):
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    m = i*self.nx[1] + j
                    if self.c_loc[p,m]!=1:
                        continue

                    if self.cws_loc[p,m]==-1 and self.cwn_loc[p,m]==1 and self.ces_loc[p,m]==1:
                        tmp = torch.tensor([self.fs_st[p,m]*self.hx[0],self.fw_st[p,m]*self.hx[1]])
                        self.fws_l[p,m] = ((tmp**2).sum())**0.5
                        self.fws_x[p,m:m+1,:] = self.cws_x[p,m:m+1,:] + 0.5*torch.tensor([tmp[0],tmp[1]])
                        self.fws_n[p,m,:] = torch.tensor([-tmp[1],-tmp[0]])
                        self.fws_n[p,m,:] = self.fws_n[p,m,:]/(((self.fws_n[p,m,:]**2).sum())**0.5)
                    
                    if self.cwn_loc[p,m]==-1 and self.cws_loc[p,m]==1 and self.cen_loc[p,m]==1:
                        tmp = torch.tensor([self.fn_st[p,m]*self.hx[0],(1-self.fw_ed[p,m])*self.hx[1]])
                        self.fwn_l[p,m] = (sum(tmp**2))**0.5
                        self.fwn_x[p,m:m+1,:] = self.cwn_x[p,m:m+1,:] + 0.5*torch.tensor([tmp[0],-tmp[1]])
                        self.fwn_n[p,m,:] = torch.tensor([-tmp[1],tmp[0]])
                        self.fwn_n[p,m,:] = self.fwn_n[p,m,:]/(((self.fwn_n[p,m,:]**2).sum())**0.5)
                    
                    if self.ces_loc[p,m]==-1 and self.cen_loc[p,m]==1 and self.cws_loc[p,m]==1:
                        tmp = torch.tensor([(1-self.fs_ed[p,m])*self.hx[0],self.fe_st[p,m]*self.hx[1]])
                        self.fes_l[p,m] = ((tmp**2).sum())**0.5
                        self.fes_x[p,m:m+1,:] = self.ces_x[p,m:m+1,:] + 0.5*torch.tensor([-tmp[0],tmp[1]])
                        self.fes_n[p,m,:] = torch.tensor([tmp[1],-tmp[0]])
                        self.fes_n[p,m,:] = self.fes_n[p,m,:]/(((self.fes_n[p,m,:]**2).sum())**0.5)
                    
                    if self.cen_loc[p,m]==-1 and self.ces_loc[p,m]==1 and self.cwn_loc[p,m]==1:
                        tmp = torch.tensor([(1-self.fn_ed[p,m])*self.hx[0],(1-self.fe_ed[p,m])*self.hx[1]])
                        self.fen_l[p,m] = ((tmp**2).sum())**0.5
                        self.fen_x[p,m:m+1,:] = self.cen_x[p,m:m+1,:] + 0.5*torch.tensor([-tmp[0],-tmp[1]])
                        self.fen_n[p,m,:] = torch.tensor([tmp[1],tmp[0]])
                        self.fen_n[p,m,:] = self.fen_n[p,m,:]/(((self.fen_n[p,m,:]**2).sum())**0.5)
        
        """ cell area """
        print('Genrating cell area ...')
        self.c_a = torch.zeros(self.parm_size,self.c_size)
        for p in range(self.parm_size):
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    m = i*self.nx[1] + j
                    if self.c_loc[p,m]!=1:
                        continue

                    self.c_a[p,m] = self.hx[0]*self.hx[1]
                    if self.cws_loc[p,m]==-1 and self.cwn_loc[p,m]==1 and self.ces_loc[p,m]==1:
                        self.c_a[p,m] = self.c_a[p,m] - 0.5*(self.fw_st[p,m]*self.hx[1])*(self.fs_st[p,m]*self.hx[0])
                    if self.cwn_loc[p,m]==-1 and self.cws_loc[p,m]==1 and self.cen_loc[p,m]==1:
                        self.c_a[p,m] = self.c_a[p,m] - 0.5*((1.0-self.fw_ed[p,m])*self.hx[1])*(self.fn_st[p,m]*self.hx[0])
                    if self.ces_loc[p,m]==-1 and self.cen_loc[p,m]==1 and self.cws_loc[p,m]==1:
                        self.c_a[p,m] = self.c_a[p,m] - 0.5*(self.fe_st[p,m]*self.hx[1])*((1.0-self.fs_ed[p,m])*self.hx[0])
                    if self.cen_loc[p,m]==-1 and self.ces_loc[p,m]==1 and self.cwn_loc[p,m]==1:
                        self.c_a[p,m] = self.c_a[p,m] - 0.5*((1.0-self.fe_ed[p,m])*self.hx[1])*((1.0-self.fn_ed[p,m])*self.hx[0])
                    if self.cws_loc[p,m]==-1 and self.cwn_loc[p,m]==-1:
                        self.c_a[p,m] = self.c_a[p,m] - 0.5*self.hx[1]*(self.fs_st[p,m]+self.fn_st[p,m])*self.hx[0]
                    if self.ces_loc[p,m]==-1 and self.cen_loc[p,m]==-1:
                        self.c_a[p,m] = self.c_a[p,m] - 0.5*self.hx[1]*(1.0-self.fs_ed[p,m]+1.0-self.fn_ed[p,m])*self.hx[0]
                    if self.cws_loc[p,m]==-1 and self.ces_loc[p,m]==-1:
                        self.c_a[p,m] = self.c_a[p,m] - 0.5*(self.fw_st[p,m]+self.fe_st[p,m])*self.hx[1]*self.hx[0]
                    if self.cwn_loc[p,m]==-1 and self.cen_loc[p,m]==-1:
                        self.c_a[p,m] = self.c_a[p,m] - 0.5*(1.0-self.fw_ed[p,m]+1.0-self.fe_ed[p,m])*self.hx[1]*self.hx[0]

        """ interpolation node """
        print('Genrating interpolation node ...')
        self.intp_n_size = 3**2
        self.intp_x = torch.zeros(self.parm_size,self.c_size,self.intp_n_size,self.dim)
        self.intp_i = torch.zeros(self.parm_size,self.c_size,self.intp_n_size).long()
        self.intp_v = torch.zeros(self.parm_size,self.c_size,self.intp_n_size,3)
        for p in range(self.parm_size):
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    m = i*self.nx[1] + j
                    if self.c_loc[p,m]!=1:
                        continue
                    
                    self.intp_x[p,m,:,:], self.intp_i[p,m,:] = self.intp_node([i,j], p)
        
    def intp_node(self, idx, p):
        intp_n_size = 3**2
        xi = torch.zeros(intp_n_size,self.dim)
        ii = torch.zeros(intp_n_size).long()

        dir = [[-1,-1],[-1, 0],[-1, 1], [0,-1],[0, 0],[0, 1], [1,-1],[1, 0],[1, 1]]
        dir = torch.tensor(dir).reshape(intp_n_size,self.dim)

        # regular point
        ix = idx[0]+dir[:,0]; iy = idx[1]+dir[:,1]
        m = ix*self.nx[1] + iy
        idx1 = ((ix>=0) & (ix<self.nx[0]) & (iy>=0) & (iy<self.nx[1]))
        idx1[idx1==True] = (self.c_loc[p,m[idx1]]==1)
        
        xi[idx1,:] = self.c_x[p,m[idx1],:]
        ii[idx1] = m[idx1]
        
        # irregular point
        idx1 = ~idx1
        m = idx[0]*self.nx[1] + idx[1]
        x1 = self.c_x[p,m,:]
        x1 = x1.reshape(1,self.dim).repeat(intp_n_size,1)
        x2 = x1 + torch.tensor([self.hx[0],self.hx[1]]) * dir

        xi_tmp = self.geo.intersection(x1[idx1,:], x2[idx1,:], p)
        
        xi[idx1,:] = xi_tmp
        ii[idx1] = -1

        return xi, ii