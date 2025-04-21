import numpy as np

class InterpolationFunction():
    def __init__(self, geo, mesh, dtype=np.float32):
        self.geo = geo
        self.mesh = mesh
        self.dtype = dtype

        # interpolation node
        print('Genrating interpolation node ...')
        self.n_size = 3**2
        self.x = np.zeros([self.mesh.c_size,self.n_size,self.mesh.dim])
        self.i = np.zeros([self.mesh.c_size,self.n_size], np.int32)
        for i in range(self.mesh.nx[0]):
            for j in range(self.mesh.nx[1]):
                m = i*self.mesh.nx[1] + j
                if self.mesh.c_loc[m]!=1:
                    continue
                
                self.x[m,:,:], self.i[m,:] = self.intp_node([i,j])
        
        self.intp_coef_regular_cell()
        
    def intp_node(self, idx):
        n_size = 3**2
        xi = np.zeros([n_size,self.mesh.dim])
        ii = np.zeros(n_size, np.int32)

        dir = [[-1,-1],[-1, 0],[-1, 1], [0,-1],[0, 0],[0, 1], [1,-1],[1, 0],[1, 1]]
        dir = np.array(dir).reshape(n_size,self.mesh.dim)

        # regular point
        ix = idx[0]+dir[:,0]; iy = idx[1]+dir[:,1]
        m = ix*self.mesh.nx[1] + iy
        idx1 = ((ix>=0) & (ix<self.mesh.nx[0]) & (iy>=0) & (iy<self.mesh.nx[1]))
        idx1[idx1==True] = (self.mesh.c_loc[m[idx1]]==1)
        
        xi[idx1,:] = self.mesh.c_x[m[idx1],:]
        ii[idx1] = m[idx1]
        
        # irregular point
        idx1 = ~idx1
        m = idx[0]*self.mesh.nx[1] + idx[1]
        x1 = self.mesh.c_x[m,:]
        x1 = x1.reshape(1,self.mesh.dim).repeat(self.n_size, axis=0)
        x2 = x1 + np.array([self.mesh.hx[0],self.mesh.hx[1]]) * dir

        xi_tmp = self.geo.intersection(x1[idx1,:], x2[idx1,:])

        xi[idx1,:] = xi_tmp
        ii[idx1] = -1

        return xi, ii

    def intp_coef(self, xi, x):
        xi = xi.copy().astype(np.float64)
        x = x.copy().astype(np.float64)
        
        intp_n_size = 3**2
        p = np.zeros([intp_n_size,intp_n_size], dtype=np.float64)
        for r in range(3):
            for s in range(3):
                n = r*3 + s
                p[:,n] = xi[:,0]**r * xi[:,1]**s
        b = p
        b = np.linalg.inv(b)
        
        pp = np.zeros([1,intp_n_size], dtype=np.float64)
        pp_x0 = np.zeros([1,intp_n_size], dtype=np.float64)
        pp_x1 = np.zeros([1,intp_n_size], dtype=np.float64)
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
        
        c = c.copy().astype(self.dtype)
        c_x0 = c_x0.copy().astype(self.dtype)
        c_x1 = c_x1.copy().astype(self.dtype)
        return c, c_x0, c_x1

    def intp_a(self, param_a):
        self.a = np.zeros([param_a.shape[0],self.mesh.c_size,self.n_size])
        for p in range(param_a.shape[0]):
            a = param_a[p,0,:,:].reshape(self.mesh.c_size)
            for m in range(self.mesh.c_size):
                if self.mesh.c_loc[m]!=1:
                    continue
                
                ii = self.i[m,:]
                for n in range(self.n_size):
                    if ii[n]!=-1:
                        self.a[p,m,n] = a[ii[n]]
                    else:
                        self.a[p,m,n] = a[m]

    def intp_coef_regular_cell(self):
        self.re_c = np.zeros([4,self.n_size])
        self.re_c_x0 = np.zeros([4,self.n_size])
        self.re_c_x1 = np.zeros([4,self.n_size])
        flag = False
        for i in range(self.mesh.nx[0]):
            for j in range(self.mesh.nx[1]):
                m = i*self.mesh.nx[1] + j
                if self.mesh.c_loc[m]!=1:
                    continue
                
                xi = self.x[m,:,:]
                ii = self.i[m,:]

                if (ii!=-1).all():
                    self.re_c[0,:], self.re_c_x0[0,:], self.re_c_x1[0,:] = \
                        self.intp_coef(xi, self.mesh.fw_x[m,:])
                    self.re_c[1,:], self.re_c_x0[1,:], self.re_c_x1[1,:] = \
                        self.intp_coef(xi, self.mesh.fe_x[m,:])
                    self.re_c[2,:], self.re_c_x0[2,:], self.re_c_x1[2,:] = \
                        self.intp_coef(xi, self.mesh.fs_x[m,:])
                    self.re_c[3,:], self.re_c_x0[3,:], self.re_c_x1[3,:] = \
                        self.intp_coef(xi, self.mesh.fn_x[m,:])
                    flag = True

                if flag: break
            if flag: break