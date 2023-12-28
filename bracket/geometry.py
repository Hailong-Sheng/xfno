import torch

class Geometry():
    def __init__(self, bounds_p1, bounds_p2, center, radius):
        self.bounds_p1 = bounds_p1
        self.bounds_p2 = bounds_p2
        self.center = center
        self.radius = radius

        self.dim = self.bounds_p1.shape[0]
    
    def location(self, x):
        x0 = x[...,0]
        x1 = x[...,1]
        x2 = x[...,2]
        
        loc = torch.zeros(x[...,0].shape)
        tol = 1e-4
        
        r = ((x0-self.center[0])**2 + (x1-self.center[1])**2) ** 0.5

        idx = ((r>self.radius+tol) & 
               (x0>self.bounds_p1[0,0]+tol) & (x0<self.bounds_p1[0,1]-tol) & 
               (x1>self.bounds_p1[1,0]+tol) & (x1<self.bounds_p1[1,1]-tol) & 
               (x2>self.bounds_p1[2,0]+tol) & (x2<self.bounds_p1[2,1]-tol))
        loc[idx] = 1

        idx = ((x0>self.bounds_p2[0,0]+tol) & (x0<self.bounds_p2[0,1]-tol) & 
               (x1>self.bounds_p2[1,0]+tol) & (x1<self.bounds_p2[1,1]-tol) & 
               (x2>self.bounds_p2[2,0]+tol) & (x2<self.bounds_p2[2,1]-tol))
        loc[idx] = 1

        idx = ((r<self.radius-tol) | 
               (x0<self.bounds_p2[0,0]-tol) | (x0>self.bounds_p2[0,1]+tol) | 
               (x1<self.bounds_p2[1,0]-tol) | (x1>self.bounds_p2[1,1]+tol) | 
               (x2<self.bounds_p1[2,0]-tol) | (x2>self.bounds_p1[2,1]+tol) | 
               ((x0>self.bounds_p1[0,1]+tol) & (x2<self.bounds_p2[2,0]-tol)) | 
               ((x0>self.bounds_p1[0,1]+tol) & (x2>self.bounds_p2[2,1]+tol)))
        loc[idx] = -1
        return loc

    def intersection(self, x0, x1):
        x0 = x0.clone()
        x1 = x1.clone()

        x = torch.zeros(x0.shape[0],self.dim)

        tol = 1e-4
        loc = self.location(x0)
        idx = (loc==-1)
        tmp = x0[idx,:]; x0[idx,:] = x1[idx,:]; x1[idx,:] = tmp

        idx = ((x1[:,0]-self.center[0])**2 + (x1[:,1]-self.center[1])**2) **0.5 < (self.radius+tol)
        a = (x1[idx,0:1]-x0[idx,0:1])**2 + (x1[idx,1:2]-x0[idx,1:2])**2
        b = 2*x0[idx,0:1]*(x1[idx,0:1]-x0[idx,0:1]) + 2*x0[idx,1:2]*(x1[idx,1:2]-x0[idx,1:2])
        c = x0[idx,0:1]**2 + x0[idx,1:2]**2 - self.radius**2
        t1 = (-b+(b**2-4*a*c)**0.5)/(2*a)
        t2 = (-b-(b**2-4*a*c)**0.5)/(2*a)

        if idx.sum()!=0:
            mask = ((t1>0-tol) & (t1<1+tol))
            if mask.sum()!=0:
                x[idx,:] = (x0[idx,:] + t1*(x1[idx,:]-x0[idx,:])) * mask
            
            mask = ((t2>0-tol) & (t2<1+tol))
            if mask.sum()!=0:
                x[idx,:] = (x0[idx,:] + t2*(x1[idx,:]-x0[idx,:])) * mask
        mask = (self.location(x)!=0)

        mask0 = (x0[:,0]<=self.bounds_p1[0,1])
        idx = (mask & mask0 & (x1[:,0] < (self.bounds_p1[0,0]+tol)))
        if idx.sum()!=0:
            t = (self.bounds_p1[0,0]-x0[idx,0:1]) / (x1[idx,0:1]-x0[idx,0:1])
            x[idx,:] = x0[idx,:] + t*(x1[idx,:]-x0[idx,:])

        idx = (mask & mask0 & (x1[:,0] > (self.bounds_p1[0,1]+tol)))
        if idx.sum()!=0:
            t = (self.bounds_p1[0,1]-x0[idx,0:1]) / (x1[idx,0:1]-x0[idx,0:1])
            x[idx,:] = x0[idx,:] + t*(x1[idx,:]-x0[idx,:])
        
        idx = (mask & mask0 & (x1[:,1] < (self.bounds_p1[1,0]+tol)))
        if idx.sum()!=0:
            t = (self.bounds_p1[1,0]-x0[idx,1:2]) / (x1[idx,1:2]-x0[idx,1:2])
            x[idx,:] = x0[idx,:] + t*(x1[idx,:]-x0[idx,:])
        
        idx = (mask & mask0 & (x1[:,1] > (self.bounds_p1[1,1]+tol)))
        if idx.sum()!=0:
            t = (self.bounds_p1[1,1]-x0[idx,1:2]) / (x1[idx,1:2]-x0[idx,1:2])
            x[idx,:] = x0[idx,:] + t*(x1[idx,:]-x0[idx,:])
        
        idx = (mask & mask0 & (x1[:,2] < (self.bounds_p1[2,0]+tol)))
        if idx.sum()!=0:
            t = (self.bounds_p1[2,0]-x0[idx,2:3]) / (x1[idx,2:3]-x0[idx,2:3])
            x[idx,:] = x0[idx,:] + t*(x1[idx,:]-x0[idx,:])
        
        idx = (mask & mask0 & (x1[:,2] > (self.bounds_p1[2,1]+tol)))
        if idx.sum()!=0:
            t = (self.bounds_p1[2,1]-x0[idx,2:3]) / (x1[idx,2:3]-x0[idx,2:3])
            x[idx,:] = x0[idx,:] + t*(x1[idx,:]-x0[idx,:])
        
        mask1 = (x0[:,0]>self.bounds_p1[0,1])
        idx = (mask & mask1 & (x1[:,0] > (self.bounds_p2[0,1]+tol)))
        if idx.sum()!=0:
            t = (self.bounds_p2[0,1]-x0[idx,0:1]) / (x1[idx,0:1]-x0[idx,0:1])
            x[idx,:] = x0[idx,:] + t*(x1[idx,:]-x0[idx,:])
        
        idx = (mask & mask1 & (x1[:,1] < (self.bounds_p2[1,0]+tol)))
        if idx.sum()!=0:
            t = (self.bounds_p2[1,0]-x0[idx,1:2]) / (x1[idx,1:2]-x0[idx,1:2])
            x[idx,:] = x0[idx,:] + t*(x1[idx,:]-x0[idx,:])
        
        idx = (mask & mask1 & (x1[:,1] > (self.bounds_p2[1,1]+tol)))
        if idx.sum()!=0:
            t = (self.bounds_p2[1,1]-x0[idx,1:2]) / (x1[idx,1:2]-x0[idx,1:2])
            x[idx,:] = x0[idx,:] + t*(x1[idx,:]-x0[idx,:])
        
        idx = (mask & mask1 & (x1[:,2] < (self.bounds_p2[2,0]+tol)))
        if idx.sum()!=0:
            t = (self.bounds_p2[2,0]-x0[idx,2:3]) / (x1[idx,2:3]-x0[idx,2:3])
            x[idx,:] = x0[idx,:] + t*(x1[idx,:]-x0[idx,:])
        
        idx = (mask & mask1 & (x1[:,2] > (self.bounds_p2[2,1]+tol)))
        if idx.sum()!=0:
            t = (self.bounds_p2[2,1]-x0[idx,2:3]) / (x1[idx,2:3]-x0[idx,2:3])
            x[idx,:] = x0[idx,:] + t*(x1[idx,:]-x0[idx,:])
        
        return x

class Mesh():
    def __init__(self, geo, bounds, nx):
        self.geo = geo
        self.bounds = bounds
        self.nx = nx

        tol = 1e-4
        
        self.dim = self.bounds.shape[0]
        self.hx = (self.bounds[:,1]-self.bounds[:,0])/self.nx
        self.xx = torch.linspace(self.bounds[0,0]+self.hx[0]/2,self.bounds[0,1]-self.hx[0]/2,self.nx[0])
        self.yy = torch.linspace(self.bounds[1,0]+self.hx[1]/2,self.bounds[1,1]-self.hx[1]/2,self.nx[1])
        self.zz = torch.linspace(self.bounds[2,0]+self.hx[2]/2,self.bounds[2,1]-self.hx[2]/2,self.nx[2])

        self.c_size = self.nx[0]*self.nx[1]*self.nx[2]
        self.c_loc = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.c_x = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim)
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    self.c_x[i,j,k,0] = self.bounds[0,0] + (i+0.5)*self.hx[0] 
                    self.c_x[i,j,k,1] = self.bounds[1,0] + (j+0.5)*self.hx[1]
                    self.c_x[i,j,k,2] = self.bounds[2,0] + (k+0.5)*self.hx[2]
                    self.c_loc[i,j,k] = self.geo.location(self.c_x[i,j,k,:])
        
        self.cwsb_loc = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.cwst_loc = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.cwnb_loc = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.cwnt_loc = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.cesb_loc = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.cest_loc = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.cenb_loc = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.cent_loc = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.cwsb_x = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim); self.cwst_x = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim)
        self.cwnb_x = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim); self.cwnt_x = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim)
        self.cesb_x = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim); self.cest_x = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim)
        self.cenb_x = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim); self.cent_x = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim)
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    self.cwsb_x[i,j,k,:] = torch.tensor([self.c_x[i,j,k,0]-self.hx[0]/2, self.c_x[i,j,k,1]-self.hx[1]/2, self.c_x[i,j,k,2]-self.hx[2]/2])
                    self.cwst_x[i,j,k,:] = torch.tensor([self.c_x[i,j,k,0]-self.hx[0]/2, self.c_x[i,j,k,1]-self.hx[1]/2, self.c_x[i,j,k,2]+self.hx[2]/2])
                    self.cwnb_x[i,j,k,:] = torch.tensor([self.c_x[i,j,k,0]-self.hx[0]/2, self.c_x[i,j,k,1]+self.hx[1]/2, self.c_x[i,j,k,2]-self.hx[2]/2])
                    self.cwnt_x[i,j,k,:] = torch.tensor([self.c_x[i,j,k,0]-self.hx[0]/2, self.c_x[i,j,k,1]+self.hx[1]/2, self.c_x[i,j,k,2]+self.hx[2]/2])
                    self.cesb_x[i,j,k,:] = torch.tensor([self.c_x[i,j,k,0]+self.hx[0]/2, self.c_x[i,j,k,1]-self.hx[1]/2, self.c_x[i,j,k,2]-self.hx[2]/2])
                    self.cest_x[i,j,k,:] = torch.tensor([self.c_x[i,j,k,0]+self.hx[0]/2, self.c_x[i,j,k,1]-self.hx[1]/2, self.c_x[i,j,k,2]+self.hx[2]/2])
                    self.cenb_x[i,j,k,:] = torch.tensor([self.c_x[i,j,k,0]+self.hx[0]/2, self.c_x[i,j,k,1]+self.hx[1]/2, self.c_x[i,j,k,2]-self.hx[2]/2])
                    self.cent_x[i,j,k,:] = torch.tensor([self.c_x[i,j,k,0]+self.hx[0]/2, self.c_x[i,j,k,1]+self.hx[1]/2, self.c_x[i,j,k,2]+self.hx[2]/2])
                    
                    self.cwsb_loc[i,j,k] = self.geo.location(self.cwsb_x[i,j,k,:])
                    self.cwst_loc[i,j,k] = self.geo.location(self.cwst_x[i,j,k,:])
                    self.cwnb_loc[i,j,k] = self.geo.location(self.cwnb_x[i,j,k,:])
                    self.cwnt_loc[i,j,k] = self.geo.location(self.cwnt_x[i,j,k,:])
                    self.cesb_loc[i,j,k] = self.geo.location(self.cesb_x[i,j,k,:])
                    self.cest_loc[i,j,k] = self.geo.location(self.cest_x[i,j,k,:])
                    self.cenb_loc[i,j,k] = self.geo.location(self.cenb_x[i,j,k,:])
                    self.cent_loc[i,j,k] = self.geo.location(self.cent_x[i,j,k,:])
        
        self.fw_st = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.fw_ed = torch.ones(self.nx[0],self.nx[1],self.nx[2])
        self.fe_st = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.fe_ed = torch.ones(self.nx[0],self.nx[1],self.nx[2])
        self.fs_st = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.fs_ed = torch.ones(self.nx[0],self.nx[1],self.nx[2])
        self.fn_st = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.fn_ed = torch.ones(self.nx[0],self.nx[1],self.nx[2])
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    if self.cwsb_loc[i,j,k]==-1 and self.cwnb_loc[i,j,k]!=-1:
                        tmp_x = self.geo.intersection(self.cwsb_x[i,j,k:k+1,:],self.cwnb_x[i,j,k:k+1,:])
                        self.fw_st[i,j,k] = (tmp_x[0,1]-self.cwsb_x[i,j,k,1])/self.hx[1]
                    
                    if self.cwsb_loc[i,j,k]!=-1 and self.cwnb_loc[i,j,k]==-1:
                        tmp_x = self.geo.intersection(self.cwsb_x[i,j,k:k+1,:],self.cwnb_x[i,j,k:k+1,:])
                        self.fw_ed[i,j,k] = (tmp_x[0,1]-self.cwsb_x[i,j,k,1])/self.hx[1]
                    
                    if self.cwsb_loc[i,j,k]==-1 and self.cwnb_loc[i,j,k]==-1:
                        tmp_x1 = abs(self.cwsb_x[i,j,k,:]-geo.center)
                        tmp_x2 = abs(self.cwnb_x[i,j,k,:]-geo.center)
                        if sum(tmp_x1**2) < sum(tmp_x2**2):
                            self.fw_st[i,j,k] = 1.0; self.fw_ed[i,j,k] = 1.0
                        else:
                            self.fw_st[i,j,k] = 0.0; self.fw_ed[i,j,k] = 0.0

                    if self.cesb_loc[i,j,k]==-1 and self.cenb_loc[i,j,k]!=-1:
                        tmp_x = self.geo.intersection(self.cesb_x[i,j,k:k+1,:],self.cenb_x[i,j,k:k+1,:])
                        self.fe_st[i,j,k] = (tmp_x[0,1]-self.cesb_x[i,j,k,1])/self.hx[1]
                    
                    if self.cesb_loc[i,j,k]!=-1 and self.cenb_loc[i,j,k]==-1:
                        tmp_x = self.geo.intersection(self.cesb_x[i,j,k:k+1,:],self.cenb_x[i,j,k:k+1,:])
                        self.fe_ed[i,j,k] = (tmp_x[0,1]-self.cesb_x[i,j,k,1])/self.hx[1]
                    
                    if self.cesb_loc[i,j,k]==-1 and self.cenb_loc[i,j,k]==-1:
                        tmp_x1 = abs(self.cesb_x[i,j,k,:]-geo.center)
                        tmp_x2 = abs(self.cenb_x[i,j,k,:]-geo.center)
                        if sum(tmp_x1**2) < sum(tmp_x2**2):
                            self.fe_st[i,j,k] = 1.0; self.fe_ed[i,j,k] = 1.0
                        else:
                            self.fe_st[i,j,k] = 0.0; self.fe_ed[i,j,k] = 0.0
                    
                    if self.cwsb_loc[i,j,k]==-1 and self.cesb_loc[i,j,k]!=-1:
                        tmp_x = self.geo.intersection(self.cwsb_x[i,j,k:k+1,:],self.cesb_x[i,j,k:k+1,:])
                        self.fs_st[i,j,k] = (tmp_x[0,0]-self.cwsb_x[i,j,k,0])/self.hx[0]
                    
                    if self.cwsb_loc[i,j,k]!=-1 and self.cesb_loc[i,j,k]==-1:
                        tmp_x = self.geo.intersection(self.cwsb_x[i,j,k:k+1,:],self.cesb_x[i,j,k:k+1,:])
                        self.fs_ed[i,j,k] = (tmp_x[0,0]-self.cwsb_x[i,j,k,0])/self.hx[0]

                    if self.cwsb_loc[i,j,k]==-1 and self.cesb_loc[i,j,k]==-1:
                        tmp_x1 = abs(self.cwsb_x[i,j,k,:]-geo.center)
                        tmp_x2 = abs(self.cesb_x[i,j,k,:]-geo.center)
                        if sum(tmp_x1**2) < sum(tmp_x2**2):
                            self.fs_st[i,j,k] = 1.0; self.fs_ed[i,j,k] = 1.0
                        else:
                            self.fs_st[i,j,k] = 0.0; self.fs_ed[i,j,k] = 0.0

                    if self.cwnb_loc[i,j,k]==-1 and self.cenb_loc[i,j,k]!=-1:
                        tmp_x = self.geo.intersection(self.cwnb_x[i,j,k:k+1,:],self.cenb_x[i,j,k:k+1,:])
                        self.fn_st[i,j,k] = (tmp_x[0,0]-self.cwnb_x[i,j,k,0])/self.hx[0]
                    
                    if self.cwnb_loc[i,j,k]!=-1 and self.cenb_loc[i,j,k]==-1:
                        tmp_x = self.geo.intersection(self.cwnb_x[i,j,k:k+1,:],self.cenb_x[i,j,k:k+1,:])
                        self.fn_ed[i,j,k] = (tmp_x[0,0]-self.cwnb_x[i,j,k,0])/self.hx[0]
                    
                    if self.cwnb_loc[i,j,k]==-1 and self.cenb_loc[i,j,k]==-1:
                        tmp_x1 = abs(self.cwnb_x[i,j,k,:]-geo.center)
                        tmp_x2 = abs(self.cenb_x[i,j,k,:]-geo.center)
                        if sum(tmp_x1**2) < sum(tmp_x2**2):
                            self.fn_st[i,j,k] = 1.0; self.fn_ed[i,j,k] = 1.0
                        else:
                            self.fn_st[i,j,k] = 0.0; self.fn_ed[i,j,k] = 0.0
        
        self.fw_x = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim); self.fe_x = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim)
        self.fs_x = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim); self.fn_x = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim)
        self.fw_n = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim); self.fe_n = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim)
        self.fs_n = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim); self.fn_n = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim)
        self.fw_a = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.fe_a = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.fs_a = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.fn_a = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.fw_loc = -torch.ones(self.nx[0],self.nx[1],self.nx[2]); self.fe_loc = -torch.ones(self.nx[0],self.nx[1],self.nx[2])
        self.fs_loc = -torch.ones(self.nx[0],self.nx[1],self.nx[2]); self.fn_loc = -torch.ones(self.nx[0],self.nx[1],self.nx[2])
        self.fw_t = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.fe_t = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.fs_t = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.fn_t = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    if self.c_loc[i,j,k]!=1:
                        continue

                    if self.cwsb_loc[i,j,k]==-1 and self.cwnb_loc[i,j,k]==-1:
                        tmp_x1 = 0.5 * (self.cwsb_x[i,j,k,:] + self.fs_st[i,j,k]*(self.cesb_x[i,j,k,:]-self.cwsb_x[i,j,k,:]) + 
                                        self.cwst_x[i,j,k,:] + self.fs_st[i,j,k]*(self.cest_x[i,j,k,:]-self.cwst_x[i,j,k,:]))
                        tmp_x2 = 0.5 * (self.cwnb_x[i,j,k,:] + self.fn_st[i,j,k]*(self.cenb_x[i,j,k,:]-self.cwnb_x[i,j,k,:]) + 
                                        self.cwnt_x[i,j,k,:] + self.fn_st[i,j,k]*(self.cent_x[i,j,k,:]-self.cwnt_x[i,j,k,:]))
                        self.fw_loc[i,j,k] = 0
                    else:
                        tmp_x1 = 0.5 * (self.cwsb_x[i,j,k,:] + self.fw_st[i,j,k]*(self.cwnb_x[i,j,k,:]-self.cwsb_x[i,j,k,:]) + 
                                        self.cwst_x[i,j,k,:] + self.fw_st[i,j,k]*(self.cwnt_x[i,j,k,:]-self.cwst_x[i,j,k,:]))
                        tmp_x2 = 0.5 * (self.cwsb_x[i,j,k,:] + self.fw_ed[i,j,k]*(self.cwnb_x[i,j,k,:]-self.cwsb_x[i,j,k,:]) + 
                                        self.cwst_x[i,j,k,:] + self.fw_ed[i,j,k]*(self.cwnt_x[i,j,k,:]-self.cwst_x[i,j,k,:]))
                        self.fw_loc[i,j,k] = self.geo.location(0.5*(tmp_x1+tmp_x2))
                    
                    self.fw_x[i,j,k,:] = 0.5*(tmp_x1+tmp_x2)
                    self.fw_n[i,j,k,:] = torch.tensor([tmp_x1[1]-tmp_x2[1],tmp_x2[0]-tmp_x1[0],0])
                    self.fw_a[i,j,k] = (sum((tmp_x2-tmp_x1)**2))**0.5
                    if self.fw_a[i,j,k]>tol:
                        self.fw_n[i,j,k,:] = self.fw_n[i,j,k,:]/self.fw_a[i,j,k]
                    
                    self.fw_a[i,j,k] = self.fw_a[i,j,k] * self.hx[2]
                    if self.fw_loc[i,j,k]==0:
                        if i==0:
                            self.fw_t[i,j,k] = 1
                        else:
                            self.fw_t[i,j,k] = 2
                    
                    if self.cesb_loc[i,j,k]==-1 and self.cenb_loc[i,j,k]==-1:
                        tmp_x1 = 0.5 * (self.cwsb_x[i,j,k,:] + self.fs_ed[i,j,k]*(self.cesb_x[i,j,k,:]-self.cwsb_x[i,j,k,:]) + 
                                        self.cwst_x[i,j,k,:] + self.fs_ed[i,j,k]*(self.cest_x[i,j,k,:]-self.cwst_x[i,j,k,:]))
                        tmp_x2 = 0.5 * (self.cwnb_x[i,j,k,:] + self.fn_ed[i,j,k]*(self.cenb_x[i,j,k,:]-self.cwnb_x[i,j,k,:]) + 
                                        self.cwnt_x[i,j,k,:] + self.fn_ed[i,j,k]*(self.cent_x[i,j,k,:]-self.cwnt_x[i,j,k,:]))
                        self.fe_loc[i,j,k] = 0
                    else:
                        tmp_x1 = 0.5 * (self.cesb_x[i,j,k,:] + self.fe_st[i,j,k]*(self.cenb_x[i,j,k,:]-self.cesb_x[i,j,k,:]) + 
                                        self.cest_x[i,j,k,:] + self.fe_st[i,j,k]*(self.cent_x[i,j,k,:]-self.cest_x[i,j,k,:]))
                        tmp_x2 = 0.5 * (self.cesb_x[i,j,k,:] + self.fe_ed[i,j,k]*(self.cenb_x[i,j,k,:]-self.cesb_x[i,j,k,:]) + 
                                        self.cest_x[i,j,k,:] + self.fe_ed[i,j,k]*(self.cent_x[i,j,k,:]-self.cest_x[i,j,k,:]))
                        self.fe_loc[i,j,k] = self.geo.location(0.5*(tmp_x1+tmp_x2))
                    
                    self.fe_x[i,j,k,:] = 0.5*(tmp_x1+tmp_x2)
                    self.fe_n[i,j,k,:] = torch.tensor([tmp_x2[1]-tmp_x1[1],tmp_x1[0]-tmp_x2[0],0])
                    self.fe_a[i,j,k] = (sum((tmp_x2-tmp_x1)**2))**0.5
                    if self.fe_a[i,j,k]>tol:
                        self.fe_n[i,j,k,:] = self.fe_n[i,j,k,:]/self.fe_a[i,j,k]
                    
                    self.fe_a[i,j,k] = self.fe_a[i,j,k] * self.hx[2]
                    if self.fe_loc[i,j,k]==0:
                        self.fe_t[i,j,k] = 2

                    if self.cwsb_loc[i,j,k]==-1 and self.cesb_loc[i,j,k]==-1:
                        tmp_x1 = 0.5 * (self.cwsb_x[i,j,k,:] + self.fw_st[i,j,k]*(self.cwnb_x[i,j,k,:]-self.cwsb_x[i,j,k,:]) + 
                                        self.cwst_x[i,j,k,:] + self.fw_st[i,j,k]*(self.cwnt_x[i,j,k,:]-self.cwst_x[i,j,k,:]))
                        tmp_x2 = 0.5 * (self.cesb_x[i,j,k,:] + self.fe_st[i,j,k]*(self.cenb_x[i,j,k,:]-self.cesb_x[i,j,k,:]) + 
                                        self.cest_x[i,j,k,:] + self.fe_st[i,j,k]*(self.cent_x[i,j,k,:]-self.cest_x[i,j,k,:]))
                        self.fs_loc[i,j,k] = 0
                    else:
                        tmp_x1 = 0.5 * (self.cwsb_x[i,j,k,:] + self.fs_st[i,j,k]*(self.cesb_x[i,j,k,:]-self.cwsb_x[i,j,k,:]) + 
                                        self.cwst_x[i,j,k,:] + self.fs_st[i,j,k]*(self.cest_x[i,j,k,:]-self.cwst_x[i,j,k,:]))
                        tmp_x2 = 0.5 * (self.cwsb_x[i,j,k,:] + self.fs_ed[i,j,k]*(self.cesb_x[i,j,k,:]-self.cwsb_x[i,j,k,:]) + 
                                        self.cwst_x[i,j,k,:] + self.fs_ed[i,j,k]*(self.cest_x[i,j,k,:]-self.cwst_x[i,j,k,:]))
                        self.fs_loc[i,j,k] = self.geo.location(0.5*(tmp_x1+tmp_x2))
                    
                    self.fs_x[i,j,k,:] = 0.5*(tmp_x1+tmp_x2)
                    self.fs_n[i,j,k,:] = torch.tensor([tmp_x2[1]-tmp_x1[1],tmp_x1[0]-tmp_x2[0],0])
                    self.fs_a[i,j,k] = (sum((tmp_x2-tmp_x1)**2))**0.5
                    if self.fs_a[i,j,k]>tol:
                        self.fs_n[i,j,k,:] = self.fs_n[i,j,k,:]/self.fs_a[i,j,k]
                    
                    self.fs_a[i,j,k] = self.fs_a[i,j,k] * self.hx[2]
                    if self.fs_loc[i,j,k]==0:
                        self.fs_t[i,j,k] = 2

                    if self.cwnb_loc[i,j,k]==-1 and self.cenb_loc[i,j,k]==-1:
                        tmp_x1 = 0.5 * (self.cwsb_x[i,j,k,:] + self.fw_ed[i,j,k]*(self.cwnb_x[i,j,k,:]-self.cwsb_x[i,j,k,:]) + 
                                        self.cwst_x[i,j,k,:] + self.fw_ed[i,j,k]*(self.cwnt_x[i,j,k,:]-self.cwst_x[i,j,k,:]))
                        tmp_x2 = 0.5 * (self.cesb_x[i,j,k,:] + self.fe_ed[i,j,k]*(self.cenb_x[i,j,k,:]-self.cesb_x[i,j,k,:]) + 
                                        self.cest_x[i,j,k,:] + self.fe_ed[i,j,k]*(self.cent_x[i,j,k,:]-self.cest_x[i,j,k,:]))
                        self.fn_loc[i,j,k] = 0
                    else:
                        tmp_x1 = 0.5 * (self.cwnb_x[i,j,k,:] + self.fn_st[i,j,k]*(self.cenb_x[i,j,k,:]-self.cwnb_x[i,j,k,:]) + 
                                        self.cwnt_x[i,j,k,:] + self.fn_st[i,j,k]*(self.cent_x[i,j,k,:]-self.cwnt_x[i,j,k,:]))
                        tmp_x2 = 0.5 * (self.cwnb_x[i,j,k,:] + self.fn_ed[i,j,k]*(self.cenb_x[i,j,k,:]-self.cwnb_x[i,j,k,:]) + 
                                        self.cwnt_x[i,j,k,:] + self.fn_ed[i,j,k]*(self.cent_x[i,j,k,:]-self.cwnt_x[i,j,k,:]))
                        self.fn_loc[i,j,k] = self.geo.location(0.5*(tmp_x1+tmp_x2))
                    
                    self.fn_x[i,j,k,:] = 0.5*(tmp_x1+tmp_x2)
                    self.fn_n[i,j,k,:] = torch.tensor([tmp_x1[1]-tmp_x2[1],tmp_x2[0]-tmp_x1[0],0])
                    self.fn_a[i,j,k] = (sum((tmp_x2-tmp_x1)**2))**0.5
                    if self.fn_a[i,j,k]>tol:
                        self.fn_n[i,j,k,:] = self.fn_n[i,j,k,:]/self.fn_a[i,j,k]
                    
                    self.fn_a[i,j,k] = self.fn_a[i,j,k] * self.hx[2]
                    if self.fn_loc[i,j,k]==0:
                        self.fn_t[i,j,k] = 2
        
        self.fws_x = torch.ones(self.nx[0],self.nx[1],self.nx[2],self.dim); self.fwn_x = torch.ones(self.nx[0],self.nx[1],self.nx[2],self.dim)
        self.fes_x = torch.ones(self.nx[0],self.nx[1],self.nx[2],self.dim); self.fen_x = torch.ones(self.nx[0],self.nx[1],self.nx[2],self.dim)
        self.fws_n = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim); self.fwn_n = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim)
        self.fes_n = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim); self.fen_n = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim)
        self.fws_a = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.fwn_a = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.fes_a = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.fen_a = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.fws_loc = torch.ones(self.nx[0],self.nx[1],self.nx[2]); self.fwn_loc = torch.ones(self.nx[0],self.nx[1],self.nx[2])
        self.fes_loc = torch.ones(self.nx[0],self.nx[1],self.nx[2]); self.fen_loc = torch.ones(self.nx[0],self.nx[1],self.nx[2])
        self.fws_t = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.fwn_t = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.fes_t = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.fen_t = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    if self.c_loc[i,j,k]!=1:
                        continue

                    if self.cwsb_loc[i,j,k]==-1 and self.cwnb_loc[i,j,k]!=-1 and self.cesb_loc[i,j,k]!=-1:
                        self.fws_loc[i,j,k] = 0; self.fws_t[i,j,k] = 2
                        tmp_x1 = 0.5 * (self.cwsb_x[i,j,k,:] + self.fw_st[i,j,k]*(self.cwnb_x[i,j,k,:]-self.cwsb_x[i,j,k,:]) + 
                                        self.cwst_x[i,j,k,:] + self.fw_st[i,j,k]*(self.cwnt_x[i,j,k,:]-self.cwst_x[i,j,k,:]))
                        tmp_x2 = 0.5 * (self.cwsb_x[i,j,k,:] + self.fs_st[i,j,k]*(self.cesb_x[i,j,k,:]-self.cwsb_x[i,j,k,:]) + 
                                        self.cwst_x[i,j,k,:] + self.fs_st[i,j,k]*(self.cest_x[i,j,k,:]-self.cwst_x[i,j,k,:]))
                        self.fws_x[i,j,k,:] = 0.5*(tmp_x1+tmp_x2)
                        self.fws_a[i,j,k] = torch.norm(torch.tensor([self.fw_st[i,j,k]*self.hx[1],self.fs_st[i,j,k]*self.hx[0]])) * self.hx[2];
                        self.fws_n[i,j,k,0] = -self.fw_st[i,j,k]; self.fws_n[i,j,k,1] = -self.fs_st[i,j,k]
                        self.fws_n[i,j,k,:] = self.fws_n[i,j,k,:]/(sum(self.fws_n[i,j,k,:]**2))**0.5
                    
                    if self.cwnb_loc[i,j,k]==-1 and self.cwsb_loc[i,j,k]!=-1 and self.cenb_loc[i,j,k]!=-1:
                        self.fwn_loc[i,j,k] = 0; self.fwn_t[i,j,k] = 2
                        tmp_x1 = 0.5 * (self.cwsb_x[i,j,k,:] + self.fw_ed[i,j,k]*(self.cwnb_x[i,j,k,:]-self.cwsb_x[i,j,k,:]) + 
                                        self.cwst_x[i,j,k,:] + self.fw_ed[i,j,k]*(self.cwnt_x[i,j,k,:]-self.cwst_x[i,j,k,:]))
                        tmp_x2 = 0.5 * (self.cwnb_x[i,j,k,:] + self.fn_st[i,j,k]*(self.cenb_x[i,j,k,:]-self.cwnb_x[i,j,k,:]) + 
                                        self.cwnt_x[i,j,k,:] + self.fn_st[i,j,k]*(self.cent_x[i,j,k,:]-self.cwnt_x[i,j,k,:]))
                        self.fwn_x[i,j,k,:] = 0.5*(tmp_x1+tmp_x2)
                        self.fwn_a[i,j,k] = torch.norm(torch.tensor([(1-self.fw_ed[i,j,k])*self.hx[1],self.fn_st[i,j,k]*self.hx[0]])) * self.hx[2]
                        self.fwn_n[i,j,k,0] = -(1-self.fw_ed[i,j,k]); self.fwn_n[i,j,k,1] = self.fn_st[i,j,k]
                        self.fwn_n[i,j,k,:] = self.fwn_n[i,j,k,:]/(sum(self.fwn_n[i,j,k,:]**2))**0.5
                    
                    if self.cesb_loc[i,j,k]==-1 and self.cwsb_loc[i,j,k]!=-1 and self.cenb_loc[i,j,k]!=-1:
                        self.fes_loc[i,j,k] = 0; self.fes_t[i,j,k] = 2
                        tmp_x1 = 0.5 * (self.cesb_x[i,j,k,:] + self.fe_st[i,j,k]*(self.cenb_x[i,j,k,:]-self.cesb_x[i,j,k,:]) + 
                                        self.cest_x[i,j,k,:] + self.fe_st[i,j,k]*(self.cent_x[i,j,k,:]-self.cest_x[i,j,k,:]))
                        tmp_x2 = 0.5 * (self.cwsb_x[i,j,k,:] + self.fs_ed[i,j,k]*(self.cesb_x[i,j,k,:]-self.cwsb_x[i,j,k,:]) + 
                                        self.cwst_x[i,j,k,:] + self.fs_ed[i,j,k]*(self.cest_x[i,j,k,:]-self.cwst_x[i,j,k,:]))
                        self.fes_x[i,j,k,:] = 0.5*(tmp_x1+tmp_x2)
                        self.fes_a[i,j,k] = torch.norm(torch.tensor([self.fe_st[i,j,k]*self.hx[1],(1-self.fs_ed[i,j,k])*self.hx[0]])) * self.hx[2]
                        self.fes_n[i,j,k,0] = self.fe_st[i,j,k]; self.fes_n[i,j,k,1] = -(1-self.fs_ed[i,j,k])
                        self.fes_n[i,j,k,:] = self.fes_n[i,j,k,:]/(sum(self.fes_n[i,j,k,:]**2))**0.5
                    
                    if self.cenb_loc[i,j,k]==-1 and self.cwnb_loc[i,j,k]!=-1 and self.cesb_loc[i,j,k]!=-1:
                        self.fen_loc[i,j,k] = 0; self.fen_t[i,j,k] = 2
                        tmp_x1 = 0.5 * (self.cesb_x[i,j,k,:] + self.fe_ed[i,j,k]*(self.cenb_x[i,j,k,:]-self.cesb_x[i,j,k,:]) + 
                                        self.cest_x[i,j,k,:] + self.fe_ed[i,j,k]*(self.cent_x[i,j,k,:]-self.cest_x[i,j,k,:]))
                        tmp_x2 = 0.5 * (self.cwnb_x[i,j,k,:] + self.fn_ed[i,j,k]*(self.cenb_x[i,j,k,:]-self.cwnb_x[i,j,k,:]) + 
                                        self.cwnt_x[i,j,k,:] + self.fn_ed[i,j,k]*(self.cent_x[i,j,k,:]-self.cwnt_x[i,j,k,:]))
                        self.fen_x[i,j,k,:] = 0.5*(tmp_x1+tmp_x2)
                        self.fen_a[i,j,k] = torch.norm(torch.tensor([(1-self.fe_ed[i,j,k])*self.hx[1],(1-self.fn_ed[i,j,k])*self.hx[0]])) * self.hx[2]
                        self.fen_n[i,j,k,0] = 1-self.fe_ed[i,j,k]; self.fen_n[i,j,k,1] = 1-self.fn_ed[i,j,k]
                        self.fen_n[i,j,k,:] = self.fen_n[i,j,k,:]/(sum(self.fen_n[i,j,k,:]**2))**0.5
        
        self.fb_x = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim); self.ft_x = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim)
        self.fb_n = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim); self.ft_n = torch.zeros(self.nx[0],self.nx[1],self.nx[2],self.dim)
        self.fb_a = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.ft_a = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        self.fb_loc = -torch.ones(self.nx[0],self.nx[1],self.nx[2]); self.ft_loc = -torch.ones(self.nx[0],self.nx[1],self.nx[2])
        self.fb_t = torch.zeros(self.nx[0],self.nx[1],self.nx[2]); self.ft_t = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    if self.c_loc[i,j,k]!=1:
                        continue
                    
                    self.fb_x[i,j,k,:] = 1/4 * (self.fw_x[i,j,k,:] + self.fe_x[i,j,k,:] + self.fs_x[i,j,k,:] + self.fn_x[i,j,k,:])
                    self.fb_x[i,j,k,2] = self.zz[k] - 0.5*self.hx[2]
                    self.fb_n[i,j,k,:] = torch.tensor([0,0,-1])
                    self.fb_a[i,j,k] = self.hx[0]*self.hx[1]
                    if self.cwsb_loc[i,j,k]==-1 and self.cwnb_loc[i,j,k]!=-1 and self.cesb_loc[i,j,k]!=-1:
                        self.fb_a[i,j,k] = self.fb_a[i,j,k] - 0.5*(self.fw_st[i,j,k]*self.hx[1])*(self.fs_st[i,j,k]*self.hx[0])
                    
                    if self.cwnb_loc[i,j,k]==-1 and self.cwsb_loc[i,j,k]!=-1 and self.cenb_loc[i,j,k]!=-1:
                        self.fb_a[i,j,k] = self.fb_a[i,j,k] - 0.5*((1.0-self.fw_ed[i,j,k])*self.hx[1])*(self.fn_st[i,j,k]*self.hx[0])
                    
                    if self.cesb_loc[i,j,k]==-1 and self.cenb_loc[i,j,k]!=-1 and self.cwsb_loc[i,j,k]!=-1:
                        self.fb_a[i,j,k] = self.fb_a[i,j,k] - 0.5*(self.fe_st[i,j,k]*self.hx[1])*((1.0-self.fs_ed[i,j,k])*self.hx[0])
                    
                    if self.cenb_loc[i,j,k]==-1 and self.cesb_loc[i,j,k]!=-1 and self.cwnb_loc[i,j,k]!=-1:
                        self.fb_a[i,j,k] = self.fb_a[i,j,k] - 0.5*((1.0-self.fe_ed[i,j,k])*self.hx[1])*((1.0-self.fn_ed[i,j,k])*self.hx[0])
                    
                    if self.cwsb_loc[i,j,k]==-1 and self.cwnb_loc[i,j,k]==-1:
                        self.fb_a[i,j,k] = self.fb_a[i,j,k] - 0.5*self.hx[1]*(self.fs_st[i,j,k]+self.fn_st[i,j,k])*self.hx[0]
                    
                    if self.cesb_loc[i,j,k]==-1 and self.cenb_loc[i,j,k]==-1:
                        self.fb_a[i,j,k] = self.fb_a[i,j,k] - 0.5*self.hx[1]*(1.0-self.fs_ed[i,j,k]+1.0-self.fn_ed[i,j,k])*self.hx[0]
                    
                    if self.cwsb_loc[i,j,k]==-1 and self.cesb_loc[i,j,k]==-1:
                        self.fb_a[i,j,k] = self.fb_a[i,j,k] - 0.5*(self.fw_st[i,j,k]+self.fe_st[i,j,k])*self.hx[1]*self.hx[0]
                    
                    if self.cwnb_loc[i,j,k]==-1 and self.cenb_loc[i,j,k]==-1:
                        self.fb_a[i,j,k] = self.fb_a[i,j,k] - 0.5*(1.0-self.fw_ed[i,j,k]+1.0-self.fe_ed[i,j,k])*self.hx[1]*self.hx[0]
                    
                    if k==0:
                        self.fb_loc[i,j,k] = 0; self.fb_t[i,j,k] = 2
                    else:
                        self.fb_loc[i,j,k] = 1
                    
                    self.ft_x[i,j,k,:] = 1/4 * (self.fw_x[i,j,k,:] + self.fe_x[i,j,k,:] + self.fs_x[i,j,k,:] + self.fn_x[i,j,k,:])
                    self.ft_x[i,j,k,2] = self.zz[k] + 0.5*self.hx[2]
                    self.ft_n[i,j,k,:] = torch.tensor([0,0,1])
                    self.ft_a[i,j,k] = self.fb_a[i,j,k]
                    if k==self.nx[2]-1:
                        self.ft_loc[i,j,k] = 0; self.ft_t[i,j,k] = 2
                    else:
                        self.ft_loc[i,j,k] = 1
        
        self.c_v = torch.zeros(self.nx[0],self.nx[1],self.nx[2])
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    if self.c_loc[i,j,k]!=1:
                        continue
                    
                    self.c_v[i,j,k] = self.fb_a[i,j,k] * self.hx[2]
