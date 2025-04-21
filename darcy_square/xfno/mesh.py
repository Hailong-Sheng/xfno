import numpy as np

class Mesh():
    """ Cartesian mesh """
    def __init__(self, geo, nx):
        """ initialization
        args:
            geo: geometry
            bounds: lower and upper bounds of the domain
            nx: size of the mesh
        """
        self.geo = geo
        self.nx = nx
        
        self.bounds = self.geo.bounds
        self.dim = self.bounds.shape[0]
        self.hx = (self.bounds[:,1]-self.bounds[:,0])/self.nx
        self.xx = np.linspace(self.bounds[0,0]+self.hx[0]/2,self.bounds[0,1]-self.hx[0]/2,self.nx[0])
        self.yy = np.linspace(self.bounds[1,0]+self.hx[1]/2,self.bounds[1,1]-self.hx[1]/2,self.nx[1])
        
        print('Genrating mesh ...')

        # cell center
        print('Genrating cell center ...')
        self.c_size = self.nx[0]*self.nx[1]
        self.c_x = np.zeros([self.c_size,self.dim])
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                m = i*self.nx[1] + j
                self.c_x[m,0] = self.bounds[0,0] + (i+0.5)*self.hx[0] 
                self.c_x[m,1] = self.bounds[1,0] + (j+0.5)*self.hx[1]
        
        self.c_loc = self.geo.location(self.c_x)
        
        # cell corner
        # w: west; e: east; s: south; n: north
        print('Genrating cell corner ...')
        self.cws_x = self.c_x + np.array([-0.5*self.hx[0],-0.5*self.hx[1]])
        self.cwn_x = self.c_x + np.array([-0.5*self.hx[0], 0.5*self.hx[1]])
        self.ces_x = self.c_x + np.array([ 0.5*self.hx[0],-0.5*self.hx[1]])
        self.cen_x = self.c_x + np.array([ 0.5*self.hx[0], 0.5*self.hx[1]])
        self.cws_loc = self.geo.location(self.cws_x)
        self.cwn_loc = self.geo.location(self.cwn_x)
        self.ces_loc = self.geo.location(self.ces_x)
        self.cen_loc = self.geo.location(self.cen_x)

        # neighbor cell
        print('Genrating neighbor cell ...')
        self.nw_x = self.c_x + np.array([-self.hx[0],0])
        self.ne_x = self.c_x + np.array([ self.hx[0],0])
        self.ns_x = self.c_x + np.array([0,-self.hx[1]])
        self.nn_x = self.c_x + np.array([0, self.hx[1]])
        self.nw_loc = self.geo.location(self.nw_x)
        self.ne_loc = self.geo.location(self.ne_x)
        self.ns_loc = self.geo.location(self.ns_x)
        self.nn_loc = self.geo.location(self.nn_x)
        
        # cell face
        print('Genrating cell face ...')
        self.fw_st = np.zeros(self.c_size); self.fw_ed = np.ones(self.c_size)
        self.fe_st = np.zeros(self.c_size); self.fe_ed = np.ones(self.c_size)
        self.fs_st = np.zeros(self.c_size); self.fs_ed = np.ones(self.c_size)
        self.fn_st = np.zeros(self.c_size); self.fn_ed = np.ones(self.c_size)
        self.fw_l = np.zeros(self.c_size); self.fe_l = np.zeros(self.c_size)
        self.fs_l = np.zeros(self.c_size); self.fn_l = np.zeros(self.c_size)
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
            
        self.fw_x = np.zeros([self.c_size,self.dim]); self.fe_x = np.zeros([self.c_size,self.dim])
        self.fs_x = np.zeros([self.c_size,self.dim]); self.fn_x = np.zeros([self.c_size,self.dim])
        self.fw_n = np.zeros([self.c_size,self.dim]); self.fe_n = np.zeros([self.c_size,self.dim])
        self.fs_n = np.zeros([self.c_size,self.dim]); self.fn_n = np.zeros([self.c_size,self.dim])
        self.fw_loc = np.zeros(self.c_size); self.fe_loc = np.zeros(self.c_size)
        self.fs_loc = np.zeros(self.c_size); self.fn_loc = np.zeros(self.c_size)
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
                self.fw_n[m,:] = np.array([-(tmp_x1[0,1]-tmp_x0[0,1]),tmp_x1[0,0]-tmp_x0[0,0]])
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
                self.fe_n[m,:] = np.array([tmp_x1[0,1]-tmp_x0[0,1],-(tmp_x1[0,0]-tmp_x0[0,0])])
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
                self.fs_n[m,:] = np.array([tmp_x1[0,1]-tmp_x0[0,1],-(tmp_x1[0,0]-tmp_x0[0,0])])
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
                self.fn_n[m,:] = np.array([-(tmp_x1[0,1]-tmp_x0[0,1]),tmp_x1[0,0]-tmp_x0[0,0]])
                self.fn_n[m,:] = self.fn_n[m,:]/((self.fn_n[m,:]**2).sum())**0.5    
        
        self.fws_l = np.zeros(self.c_size); self.fwn_l = np.zeros(self.c_size)
        self.fes_l = np.zeros(self.c_size); self.fen_l = np.zeros(self.c_size)
        self.fws_x = np.zeros([self.c_size,self.dim]); self.fwn_x = np.zeros([self.c_size,self.dim])
        self.fes_x = np.zeros([self.c_size,self.dim]); self.fen_x = np.zeros([self.c_size,self.dim])
        self.fws_n = np.zeros([self.c_size,self.dim]); self.fwn_n = np.zeros([self.c_size,self.dim])
        self.fes_n = np.zeros([self.c_size,self.dim]); self.fen_n = np.zeros([self.c_size,self.dim])
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                m = i*self.nx[1] + j
                if self.c_loc[m]!=1:
                    continue

                if self.cws_loc[m]==-1 and self.cwn_loc[m]==1 and self.ces_loc[m]==1:
                    tmp = np.array([self.fs_st[m]*self.hx[0],self.fw_st[m]*self.hx[1]])
                    self.fws_l[m] = ((tmp**2).sum())**0.5
                    self.fws_x[m:m+1,:] = self.cws_x[m:m+1,:] + 0.5*np.array([tmp[0],tmp[1]])
                    self.fws_n[m,:] = np.array([-tmp[1],-tmp[0]])
                    self.fws_n[m,:] = self.fws_n[m,:]/(((self.fws_n[m,:]**2).sum())**0.5)
                
                if self.cwn_loc[m]==-1 and self.cws_loc[m]==1 and self.cen_loc[m]==1:
                    tmp = np.array([self.fn_st[m]*self.hx[0],(1-self.fw_ed[m])*self.hx[1]])
                    self.fwn_l[m] = (sum(tmp**2))**0.5
                    self.fwn_x[m:m+1,:] = self.cwn_x[m:m+1,:] + 0.5*np.array([tmp[0],-tmp[1]])
                    self.fwn_n[m,:] = np.array([-tmp[1],tmp[0]])
                    self.fwn_n[m,:] = self.fwn_n[m,:]/(((self.fwn_n[m,:]**2).sum())**0.5)
                
                if self.ces_loc[m]==-1 and self.cen_loc[m]==1 and self.cws_loc[m]==1:
                    tmp = np.array([(1-self.fs_ed[m])*self.hx[0],self.fe_st[m]*self.hx[1]])
                    self.fes_l[m] = ((tmp**2).sum())**0.5
                    self.fes_x[m:m+1,:] = self.ces_x[m:m+1,:] + 0.5*np.array([-tmp[0],tmp[1]])
                    self.fes_n[m,:] = np.array([tmp[1],-tmp[0]])
                    self.fes_n[m,:] = self.fes_n[m,:]/(((self.fes_n[m,:]**2).sum())**0.5)
                
                if self.cen_loc[m]==-1 and self.ces_loc[m]==1 and self.cwn_loc[m]==1:
                    tmp = np.array([(1-self.fn_ed[m])*self.hx[0],(1-self.fe_ed[m])*self.hx[1]])
                    self.fen_l[m] = ((tmp**2).sum())**0.5
                    self.fen_x[m:m+1,:] = self.cen_x[m:m+1,:] + 0.5*np.array([-tmp[0],-tmp[1]])
                    self.fen_n[m,:] = np.array([tmp[1],tmp[0]])
                    self.fen_n[m,:] = self.fen_n[m,:]/(((self.fen_n[m,:]**2).sum())**0.5)
        
        # cell area
        print('Genrating cell area ...')
        self.c_a = np.zeros(self.c_size)
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