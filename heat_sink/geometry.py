import torch

class Geometry():
    def __init__(self, bounds, bounds_base, fins_wid, fins_hei, fins_num):
        self.bounds = bounds
        self.bounds_base = bounds_base
        self.fins_wid = fins_wid
        self.fins_hei = fins_hei
        self.fins_num = fins_num

        self.parm_size = self.bounds_base.shape[0]
        self.dim = self.bounds.shape[0]

        self.fins_wid = self.fins_wid*torch.ones(self.parm_size)
        self.fins_hei = self.fins_hei*torch.ones(self.parm_size)
        self.fins_gap = (self.bounds_base[:,1,1]-self.bounds_base[:,1,0]-
                         self.fins_num*self.fins_wid)/(self.fins_num-1)
        
        self.bounds_fins = torch.zeros(self.parm_size,self.fins_num,self.dim,2)
        for p in range(self.parm_size):
            for i in range(self.fins_num):
                self.bounds_fins[p,i,0,0] = self.bounds_base[p,0,0]
                self.bounds_fins[p,i,0,1] = self.bounds_base[p,0,1]
                self.bounds_fins[p,i,1,0] = self.bounds_base[p,1,0] + i*self.fins_wid[p] + i*self.fins_gap[p]
                self.bounds_fins[p,i,1,1] = self.bounds_base[p,1,0] + (i+1)*self.fins_wid[p] + i*self.fins_gap[p]
                self.bounds_fins[p,i,2,0] = self.bounds_base[p,2,1]
                self.bounds_fins[p,i,2,1] = self.bounds_base[p,2,1] + self.fins_hei[p]
        
        self.faces = []
        self.faces_idx = []
        for p in range(self.parm_size):
            faces, faces_idx = self.genrate_faces(p)
            self.faces.append(faces)
            self.faces_idx.append(faces_idx)

    def location(self, x, p):
        """ location of data point
            1: in the domain; 0: on the boundary; -1: out of the boundary
        """
        x0 = x[...,0]
        x1 = x[...,1]
        x2 = x[...,2]
        
        loc = torch.zeros(x[...,0].shape)
        tol = 1e-4
        
        # in the domain
        idx = ((x0>self.bounds[0,0]+tol) & (x0<self.bounds_base[p,0,0]-tol) & 
               (x1>self.bounds[1,0]+tol) & (x1<self.bounds[1,1]-tol) & 
               (x2>self.bounds[2,0]+tol) & (x2<self.bounds[2,1]-tol))
        loc[idx] = 1
        
        idx = ((x0>self.bounds_base[p,0,1]+tol) & (x0<self.bounds[0,1]-tol) & 
               (x1>self.bounds[1,0]+tol) & (x1<self.bounds[1,1]-tol) & 
               (x2>self.bounds[2,0]+tol) & (x2<self.bounds[2,1]-tol))
        loc[idx] = 1
        
        idx = ((x0>self.bounds[0,0]+tol) & (x0<self.bounds[0,1]-tol) & 
               (x1>self.bounds[1,0]+tol) & (x1<self.bounds_base[p,1,0]-tol) & 
               (x2>self.bounds[2,0]+tol) & (x2<self.bounds[2,1]-tol))
        loc[idx] = 1

        idx = ((x0>self.bounds[0,0]+tol) & (x0<self.bounds[0,1]-tol) & 
               (x1>self.bounds_base[p,1,1]+tol) & (x1<self.bounds[1,1]-tol) & 
               (x2>self.bounds[2,0]+tol) & (x2<self.bounds[2,1]-tol))
        loc[idx] = 1
        
        idx = ((x0>self.bounds[0,0]+tol) & (x0<self.bounds[0,1]-tol) & 
               (x1>self.bounds[1,0]+tol) & (x1<self.bounds[1,1]-tol) & 
               (x2>self.bounds_base[p,2,1]+self.fins_hei[p]+tol) & (x2<self.bounds[2,1]-tol))
        loc[idx] = 1
        
        for i in range(self.fins_num-1):
            idx = ((x0>self.bounds[0,0]+tol) & (x0<self.bounds[0,1]-tol) & 
                   (x1>self.bounds_fins[p,i,1,1]+tol) & (x1<self.bounds_fins[p,i+1,1,0]-tol) & 
                   (x2>self.bounds_fins[p,i,2,0]+tol) & (x2<self.bounds[2,1]-tol))
            loc[idx] = 1
        
        # out of the domain
        idx = ((x0<self.bounds[0,0]-tol) | (x0>self.bounds[0,1]+tol) | 
               (x1<self.bounds[1,0]-tol) | (x1>self.bounds[1,1]+tol) | 
               (x2<self.bounds[2,0]-tol) | (x2>self.bounds[2,1]+tol))
        loc[idx] = -1

        idx = ((x0>self.bounds_base[p,0,0]+tol) & (x0<self.bounds_base[p,0,1]-tol) & 
               (x1>self.bounds_base[p,1,0]+tol) & (x1<self.bounds_base[p,1,1]-tol) & 
               (x2<self.bounds_base[p,2,1]-tol))
        loc[idx] = -1

        for i in range(self.fins_num):
            idx = ((x0>self.bounds_fins[p,i,0,0]+tol) & (x0<self.bounds_fins[p,i,0,1]-tol) & 
                   (x1>self.bounds_fins[p,i,1,0]+tol) & (x1<self.bounds_fins[p,i,1,1]-tol) & 
                   (x2<self.bounds_fins[p,i,2,1]-tol))
            loc[idx] = -1
        
        return loc

    def genrate_faces(self, p):
        xx = torch.tensor([self.bounds[0,0],self.bounds_base[p,0,0],
                           self.bounds_base[p,0,1],self.bounds[0,1]])
        zz = torch.tensor([self.bounds[2,0],self.bounds_base[p,2,1],
                           self.bounds_base[p,2,1]+self.fins_hei[p],self.bounds[2,1]])
        
        yy = torch.zeros(2*self.fins_num+2)
        yy[0] = self.bounds[1,0]
        for i in range(self.fins_num):
            yy[2*i+1] = self.bounds_base[p,1,0] + i*self.fins_wid[p] + i*self.fins_gap[p]
            yy[2*i+2] = self.bounds_base[p,1,0] + (i+1)*self.fins_wid[p] + i*self.fins_gap[p]
        yy[-1] = self.bounds[1,1]
        
        nx = xx.shape[0]; ny = yy.shape[0]; nz = zz.shape[0]
        x = torch.zeros(nx*ny*nz,self.dim)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    m = (i*ny + j)*nz + k
                    x[m,0] = xx[i]; x[m,1] = yy[j]; x[m,2] = zz[k]
        
        faces = torch.zeros(0,3*self.dim)
        faces_idx = torch.zeros(0,3).long()
        idx0 = torch.zeros((nx-1)*(ny-1)*(nz-1)).long()
        idx1 = torch.zeros((nx-1)*(ny-1)*(nz-1)).long()
        idx2 = torch.zeros((nx-1)*(ny-1)*(nz-1)).long()
        idx3 = torch.zeros((nx-1)*(ny-1)*(nz-1)).long()
        idx4 = torch.zeros((nx-1)*(ny-1)*(nz-1)).long()
        idx5 = torch.zeros((nx-1)*(ny-1)*(nz-1)).long()
        idx6 = torch.zeros((nx-1)*(ny-1)*(nz-1)).long()
        idx7 = torch.zeros((nx-1)*(ny-1)*(nz-1)).long()
        for i in range(nx-1):
            for j in range(ny-1):
                for k in range(nz-1):
                    m = (i*(ny-1) + j)*(nz-1) + k
                    idx0[m] = (i*ny + j)*nz + k
                    idx1[m] = (i*ny + j)*nz + k+1
                    idx2[m] = (i*ny + j+1)*nz + k
                    idx3[m] = (i*ny + j+1)*nz + k+1
                    idx4[m] = ((i+1)*ny + j)*nz + k
                    idx5[m] = ((i+1)*ny + j)*nz + k+1
                    idx6[m] = ((i+1)*ny + j+1)*nz + k
                    idx7[m] = ((i+1)*ny + j+1)*nz + k+1

        faces = torch.cat([faces, torch.cat([x[idx0,:],x[idx1,:],x[idx2,:]],1)])
        faces = torch.cat([faces, torch.cat([x[idx1,:],x[idx2,:],x[idx3,:]],1)])
        faces = torch.cat([faces, torch.cat([x[idx4,:],x[idx5,:],x[idx6,:]],1)])
        faces = torch.cat([faces, torch.cat([x[idx5,:],x[idx6,:],x[idx7,:]],1)])
        faces = torch.cat([faces, torch.cat([x[idx0,:],x[idx2,:],x[idx4,:]],1)])
        faces = torch.cat([faces, torch.cat([x[idx2,:],x[idx4,:],x[idx6,:]],1)])
        faces = torch.cat([faces, torch.cat([x[idx1,:],x[idx3,:],x[idx5,:]],1)])
        faces = torch.cat([faces, torch.cat([x[idx3,:],x[idx5,:],x[idx7,:]],1)])
        faces = torch.cat([faces, torch.cat([x[idx0,:],x[idx1,:],x[idx4,:]],1)])
        faces = torch.cat([faces, torch.cat([x[idx1,:],x[idx4,:],x[idx5,:]],1)])
        faces = torch.cat([faces, torch.cat([x[idx2,:],x[idx3,:],x[idx6,:]],1)])
        faces = torch.cat([faces, torch.cat([x[idx3,:],x[idx6,:],x[idx7,:]],1)])
        faces_idx = torch.cat([faces_idx, torch.stack([idx0,idx1,idx2],1)])
        faces_idx = torch.cat([faces_idx, torch.stack([idx1,idx2,idx3],1)])
        faces_idx = torch.cat([faces_idx, torch.stack([idx4,idx5,idx6],1)])
        faces_idx = torch.cat([faces_idx, torch.stack([idx5,idx6,idx7],1)])
        faces_idx = torch.cat([faces_idx, torch.stack([idx0,idx2,idx4],1)])
        faces_idx = torch.cat([faces_idx, torch.stack([idx2,idx4,idx6],1)])
        faces_idx = torch.cat([faces_idx, torch.stack([idx1,idx3,idx5],1)])
        faces_idx = torch.cat([faces_idx, torch.stack([idx3,idx5,idx7],1)])
        faces_idx = torch.cat([faces_idx, torch.stack([idx0,idx1,idx4],1)])
        faces_idx = torch.cat([faces_idx, torch.stack([idx1,idx4,idx5],1)])
        faces_idx = torch.cat([faces_idx, torch.stack([idx2,idx3,idx6],1)])
        faces_idx = torch.cat([faces_idx, torch.stack([idx3,idx6,idx7],1)])

        faces_cen = 1/3*(faces[:,0:3]+faces[:,3:6]+faces[:,6:9])
        tol = 1e-4
        i = 0
        while i < faces.shape[0]-1:
            j = i+1
            while j < faces.shape[0]-1:
                dis = (((faces_cen[i,:]-faces_cen[j,:])**2).sum())**0.5
                if dis < tol:
                    faces = torch.cat([faces[:j,:],faces[j+1:,:]])
                    faces_idx = torch.cat([faces_idx[:j,:],faces_idx[j+1:,:]])
                    faces_cen = torch.cat([faces_cen[:j,:],faces_cen[j+1:,:]])
                    j -= 1
                j += 1
            i += 1
        
        faces_cen = 1/3*(faces[:,0:3]+faces[:,3:6]+faces[:,6:9])
        faces_new = torch.zeros(0,3*self.dim)
        faces_idx_new = torch.zeros(0,3).long()
        idx = (((faces_cen[:,2]-self.bounds[2,0]).abs()<tol) & 
               (~((faces_cen[:,0]>self.bounds_base[p,0,0]) & (faces_cen[:,0]<self.bounds_base[p,0,1]) & 
                  (faces_cen[:,1]>self.bounds_base[p,1,0]) & (faces_cen[:,1]<self.bounds_base[p,1,1]))))
        faces_idx_new = torch.cat([faces_idx_new, faces_idx[idx,:]])
        faces_new = torch.cat([faces_new, faces[idx,:]])
        
        idx = (((faces_cen[:,0]>self.bounds_base[p,0,0]-tol) & (faces_cen[:,0]<self.bounds_base[p,0,1]+tol) & 
                (faces_cen[:,1]>self.bounds_base[p,1,0]-tol) & (faces_cen[:,1]<self.bounds_base[p,1,1]+tol) & 
                (faces_cen[:,2]>self.bounds_base[p,2,0]+tol) & (faces_cen[:,2]<self.bounds_base[p,2,1]+tol)) & 
               (((faces_cen[:,0]-self.bounds_base[p,0,0]).abs()<tol) | ((faces_cen[:,0]-self.bounds_base[p,0,1]).abs()<tol) | 
                ((faces_cen[:,1]-self.bounds_base[p,1,0]).abs()<tol) | ((faces_cen[:,1]-self.bounds_base[p,1,1]).abs()<tol)))
        faces_idx_new = torch.cat([faces_idx_new, faces_idx[idx,:]])
        faces_new = torch.cat([faces_new, faces[idx,:]])
        
        bounds_tmp = torch.zeros(self.dim,2)
        for i in range(self.fins_num-1):
            bounds_tmp[0,0] = self.bounds_base[p,0,0]; bounds_tmp[0,1] = self.bounds_base[p,0,1]
            bounds_tmp[1,0] = self.bounds_base[p,1,0] + i*(self.fins_wid[p]+self.fins_gap[p])+self.fins_wid[p]
            bounds_tmp[1,1] = self.bounds_base[p,1,0] + (i+1)*(self.fins_wid[p]+self.fins_gap[p])
            idx = ((faces_cen[:,0]>bounds_tmp[0,0]-tol) & (faces_cen[:,0]<bounds_tmp[0,1]+tol) & 
                   (faces_cen[:,1]>bounds_tmp[1,0]-tol) & (faces_cen[:,1]<bounds_tmp[1,1]+tol) & 
                   ((faces_cen[:,2]-self.bounds_base[p,2,1]).abs()<tol))
            faces_idx_new = torch.cat([faces_idx_new, faces_idx[idx,:]])
            faces_new = torch.cat([faces_new, faces[idx,:]])

        for i in range(self.fins_num):
            bounds_tmp[0,0] = self.bounds_base[p,0,0]; bounds_tmp[0,1] = self.bounds_base[p,0,1]
            bounds_tmp[1,0] = self.bounds_base[p,1,0] + i*(self.fins_wid[p]+self.fins_gap[p])
            bounds_tmp[1,1] = self.bounds_base[p,1,0] + i*(self.fins_wid[p]+self.fins_gap[p]) + self.fins_wid[p]
            bounds_tmp[2,0] = self.bounds_base[p,2,1]; bounds_tmp[2,1] = self.bounds_base[p,2,1] + self.fins_hei[p]
            idx = (((faces_cen[:,0]>bounds_tmp[0,0]-tol) & (faces_cen[:,0]<bounds_tmp[0,1]+tol) & 
                    (faces_cen[:,1]>bounds_tmp[1,0]-tol) & (faces_cen[:,1]<bounds_tmp[1,1]+tol) & 
                    (faces_cen[:,2]>bounds_tmp[2,0]+tol) & (faces_cen[:,2]<bounds_tmp[2,1]+tol)) & 
                   (((faces_cen[:,0]-bounds_tmp[0,0]).abs()<tol) | ((faces_cen[:,0]-bounds_tmp[0,1]).abs()<tol) | 
                    ((faces_cen[:,1]-bounds_tmp[1,0]).abs()<tol) | ((faces_cen[:,1]-bounds_tmp[1,1]).abs()<tol) | 
                    ((faces_cen[:,2]-bounds_tmp[2,1]).abs()<tol)))
            faces_idx_new = torch.cat([faces_idx_new, faces_idx[idx,:]])
            faces_new = torch.cat([faces_new, faces[idx,:]])
        
        idx = (faces_cen[:,0]-self.bounds[0,0]).abs()<tol
        faces_idx_new = torch.cat([faces_idx_new, faces_idx[idx,:]])
        faces_new = torch.cat([faces_new, faces[idx,:]])
        idx = (faces_cen[:,0]-self.bounds[0,1]).abs()<tol
        faces_idx_new = torch.cat([faces_idx_new, faces_idx[idx,:]])
        faces_new = torch.cat([faces_new, faces[idx,:]])
        idx = (faces_cen[:,1]-self.bounds[1,0]).abs()<tol
        faces_idx_new = torch.cat([faces_idx_new, faces_idx[idx,:]])
        faces_new = torch.cat([faces_new, faces[idx,:]])
        idx = (faces_cen[:,1]-self.bounds[1,1]).abs()<tol
        faces_idx_new = torch.cat([faces_idx_new, faces_idx[idx,:]])
        faces_new = torch.cat([faces_new, faces[idx,:]])
        idx = (faces_cen[:,2]-self.bounds[2,1]).abs()<tol
        faces_idx_new = torch.cat([faces_idx_new, faces_idx[idx,:]])
        faces_new = torch.cat([faces_new, faces[idx,:]])

        faces = faces_new
        faces_idx = faces_idx_new
        return faces, faces_new
        
    def intersection(self, x0, x1, p):
        """ Compute the intersection between the segement x0x1 and the domain boundary. """
        x0 = x0.clone()
        x1 = x1.clone()

        faces = self.faces[p]
        dim = 3
        tol = 1e-4
        
        z0 = x0.reshape(x0.shape[0],1,dim)
        z1 = x1.reshape(x0.shape[0],1,dim)
        z = torch.zeros(x0.shape[0],faces.shape[0],dim)

        y0 = faces[:,0:3]
        y1 = faces[:,3:6]
        y2 = faces[:,6:9]
        n = torch.cross(y1-y0, y2-y0)
        n = n/((n**2).sum(1,keepdims=True))**0.5

        y0 = y0.reshape(1,faces.shape[0],dim)
        y1 = y1.reshape(1,faces.shape[0],dim)
        y2 = y2.reshape(1,faces.shape[0],dim)
        n = n.reshape(1,faces.shape[0],dim)

        # L(t) = z0 + t*(z1-z0)
        # n * (L(t)-y0) = 0
        t = (((y0[:,:,0]-z0[:,:,0])*n[:,:,0] + (y0[:,:,1]-z0[:,:,1])*n[:,:,1] + 
              (y0[:,:,2]-z0[:,:,2])*n[:,:,2]) / 
             ((z1[:,:,0]-z0[:,:,0])*n[:,:,0] + (z1[:,:,1]-z0[:,:,1])*n[:,:,1] + 
              (z1[:,:,2]-z0[:,:,2])*n[:,:,2]))
        z[:,:,0] = z0[:,:,0] + t*(z1[:,:,0]-z0[:,:,0])
        z[:,:,1] = z0[:,:,1] + t*(z1[:,:,1]-z0[:,:,1])
        z[:,:,2] = z0[:,:,2] + t*(z1[:,:,2]-z0[:,:,2])

        # 
        a0 = torch.zeros(1,faces.shape[0])
        a1 = torch.zeros(x0.shape[0],faces.shape[0])
        a2 = torch.zeros(x0.shape[0],faces.shape[0])
        a3 = torch.zeros(x0.shape[0],faces.shape[0])

        idx = ((n[0,:,0].abs()<tol) & (n[0,:,1].abs()<tol))
        a0[:,idx] = 0.5 * (y0[:,idx,0]*(y1[:,idx,1]-y2[:,idx,1]) + y1[:,idx,0]*(y2[:,idx,1]-y0[:,idx,1]) + 
                           y2[:,idx,0]*(y0[:,idx,1]-y1[:,idx,1])).abs()
        a1[:,idx] = 0.5 * ( z[:,idx,0]*(y1[:,idx,1]-y2[:,idx,1]) + y1[:,idx,0]*(y2[:,idx,1]- z[:,idx,1]) + 
                           y2[:,idx,0]*( z[:,idx,1]-y1[:,idx,1])).abs()
        a2[:,idx] = 0.5 * (y0[:,idx,0]*( z[:,idx,1]-y2[:,idx,1]) +  z[:,idx,0]*(y2[:,idx,1]-y0[:,idx,1]) + 
                           y2[:,idx,0]*(y0[:,idx,1]- z[:,idx,1])).abs()
        a3[:,idx] = 0.5 * (y0[:,idx,0]*(y1[:,idx,1]- z[:,idx,1]) + y1[:,idx,0]*( z[:,idx,1]-y0[:,idx,1]) + 
                            z[:,idx,0]*(y0[:,idx,1]-y1[:,idx,1])).abs()
        
        idx = ((n[0,:,1].abs()<tol) & (n[0,:,2].abs()<tol))
        a0[:,idx] = 0.5 * (y0[:,idx,1]*(y1[:,idx,2]-y2[:,idx,2]) + y1[:,idx,1]*(y2[:,idx,2]-y0[:,idx,2]) + 
                           y2[:,idx,1]*(y0[:,idx,2]-y1[:,idx,2])).abs()
        a1[:,idx] = 0.5 * ( z[:,idx,1]*(y1[:,idx,2]-y2[:,idx,2]) + y1[:,idx,1]*(y2[:,idx,2]- z[:,idx,2]) + 
                           y2[:,idx,1]*( z[:,idx,2]-y1[:,idx,2])).abs()
        a2[:,idx] = 0.5 * (y0[:,idx,1]*( z[:,idx,2]-y2[:,idx,2]) +  z[:,idx,1]*(y2[:,idx,2]-y0[:,idx,2]) + 
                           y2[:,idx,1]*(y0[:,idx,2]- z[:,idx,2])).abs()
        a3[:,idx] = 0.5 * (y0[:,idx,1]*(y1[:,idx,2]- z[:,idx,2]) + y1[:,idx,1]*( z[:,idx,2]-y0[:,idx,2]) + 
                            z[:,idx,1]*(y0[:,idx,2]-y1[:,idx,2])).abs()

        idx = ((n[0,:,2].abs()<tol) & (n[0,:,0].abs()<tol))
        a0[:,idx] = 0.5 * (y0[:,idx,2]*(y1[:,idx,0]-y2[:,idx,0]) + y1[:,idx,2]*(y2[:,idx,0]-y0[:,idx,0]) + 
                           y2[:,idx,2]*(y0[:,idx,0]-y1[:,idx,0])).abs()
        a1[:,idx] = 0.5 * ( z[:,idx,2]*(y1[:,idx,0]-y2[:,idx,0]) + y1[:,idx,2]*(y2[:,idx,0]- z[:,idx,0]) + 
                           y2[:,idx,2]*( z[:,idx,0]-y1[:,idx,0])).abs()
        a2[:,idx] = 0.5 * (y0[:,idx,2]*( z[:,idx,0]-y2[:,idx,0]) +  z[:,idx,2]*(y2[:,idx,0]-y0[:,idx,0]) + 
                           y2[:,idx,2]*(y0[:,idx,0]- z[:,idx,0])).abs()
        a3[:,idx] = 0.5 * (y0[:,idx,2]*(y1[:,idx,0]- z[:,idx,0]) + y1[:,idx,2]*( z[:,idx,0]-y0[:,idx,0]) + 
                            z[:,idx,2]*(y0[:,idx,0]-y1[:,idx,0])).abs()
        
        x = torch.zeros(x0.shape[0],dim)
        idx = ((t>0-tol) & (t<1+tol) & ((a0-(a1+a2+a3)).abs()<tol))
        
        for i in range(x0.shape[0]):
            tmp_x = z[i,idx[i,:],:]
            x[i,:] = tmp_x[0,:]
        return x
