import torch

class Geometry():
    def __init__(self, bounds, center, radius):
        self.bounds = bounds
        self.center = center
        self.radius = radius

        self.dim = self.bounds.shape[0]
        
    def location(self, x):
        """ location of data point
            1: in the domain; 0: on the boundary; -1: out of the boundary
        """
        x0 = x[...,0]
        x1 = x[...,1]
        
        loc = torch.zeros(x[...,0].shape)
        tol = 1e-6

        r = ((x0-self.center[0])**2 + (x1-self.center[1])**2) ** 0.5

        # in the domain
        idx = ((r>self.radius+tol) & 
               (x0>self.bounds[0,0]+tol) & (x0<self.bounds[0,1]-tol) & 
               (x1>self.bounds[1,0]+tol) & (x1<self.bounds[1,1]-tol))
        loc[idx] = 1

        # out of the domain
        idx = ((r<self.radius-tol) | 
               (x0<self.bounds[0,0]-tol) | (x0>self.bounds[0,1]+tol) | 
               (x1<self.bounds[1,0]-tol) | (x1>self.bounds[1,1]+tol))
        loc[idx] = -1
        
        return loc
        
    def intersection(self, x0, x1):
        """ Compute the intersection between the segement x0x1 and the domain boundary. """
        x0 = x0.clone()
        x1 = x1.clone()
        x = torch.zeros(x0.shape)
        
        tol = 1e-4
        loc = self.location(x0)
        idx = (loc==-1)
        tmp = x0[idx,:]; x0[idx,:] = x1[idx,:]; x1[idx,:] = tmp

        idx = ((x1[:,0]-self.center[0])**2 + (x1[:,1]-self.center[1])**2) **0.5 < (self.radius+tol)
        a = (x1[:,0]-x0[:,0])**2 + (x1[:,1]-x0[:,1])**2
        b = 2*(x0[:,0]-self.center[0])*(x1[:,0]-x0[:,0]) + 2*(x0[:,1]-self.center[1])*(x1[:,1]-x0[:,1])
        c = (x0[:,0]-self.center[0])**2 + (x0[:,1]-self.center[1])**2 - self.radius**2
        t0 = (-b+(b**2-4*a*c)**0.5)/(2*a)
        t1 = (-b-(b**2-4*a*c)**0.5)/(2*a)
        
        idx1 = (idx & ((t0>0-tol) & (t0<1+tol)))
        x[idx1,:] = (x0 + t0.view(-1,1)*(x1-x0))[idx1,:]
        idx2 = (idx & (~((t0>0-tol) & (t0<1+tol))))
        x[idx2,:] = (x0 + t1.view(-1,1)*(x1-x0))[idx2,:]
        
        loc = self.location(x)
        idx = ((loc!=0) & (x1[:,0] < (self.bounds[0,0]+tol)))
        t = (self.bounds[0,0]-x0[:,0]) / (x1[:,0]-x0[:,0])
        x[idx,:] = (x0 + t.view(-1,1)*(x1-x0))[idx,:]
        
        loc = self.location(x)
        idx = ((loc!=0) & (x1[:,0] > (self.bounds[0,1]+tol)))
        t = (self.bounds[0,1]-x0[:,0]) / (x1[:,0]-x0[:,0])
        x[idx,:] = (x0 + t.view(-1,1)*(x1-x0))[idx,:]
        
        loc = self.location(x)
        idx = ((loc!=0) & (x1[:,1] < (self.bounds[1,0]+tol)))
        t = (self.bounds[1,0]-x0[:,1]) / (x1[:,1]-x0[:,1])
        x[idx,:] = (x0 + t.view(-1,1)*(x1-x0))[idx,:]
        
        loc = self.location(x)
        idx = ((loc!=0) & (x1[:,1] > (self.bounds[1,1]+tol)))
        t = (self.bounds[1,1]-x0[:,1]) / (x1[:,1]-x0[:,1])
        x[idx,:] = (x0 + t.view(-1,1)*(x1-x0))[idx,:]
        
        return x
