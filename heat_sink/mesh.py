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
        self.zz = torch.linspace(self.bounds[2,0]+self.hx[2]/2,self.bounds[2,1]-self.hx[2]/2,self.nx[2])

        """ cell center """
        self.c_size = self.nx[0]*self.nx[1]*self.nx[2]
        self.c_x = torch.zeros(self.parm_size,self.c_size,self.dim)
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    m = (i*self.nx[1] + j)*self.nx[2] + k
                    self.c_x[:,m,0] = self.bounds[0,0] + (i+0.5)*self.hx[0] 
                    self.c_x[:,m,1] = self.bounds[1,0] + (j+0.5)*self.hx[1]
                    self.c_x[:,m,2] = self.bounds[2,0] + (k+0.5)*self.hx[2]
        
        self.c_loc = torch.zeros(self.parm_size,self.c_size)
        for p in range(self.parm_size):
            self.c_loc[p,:] = self.geo.location(self.c_x[p,:,:], p)
        
        """ cell corner
            w: west; e: east; s: south; n: north; b: bottom; t: top
        """
        self.cwsb_x = self.c_x + torch.tensor([-0.5*self.hx[0],-0.5*self.hx[1],-0.5*self.hx[2]])
        self.cwst_x = self.c_x + torch.tensor([-0.5*self.hx[0],-0.5*self.hx[1], 0.5*self.hx[2]])
        self.cwnb_x = self.c_x + torch.tensor([-0.5*self.hx[0], 0.5*self.hx[1],-0.5*self.hx[2]])
        self.cwnt_x = self.c_x + torch.tensor([-0.5*self.hx[0], 0.5*self.hx[1], 0.5*self.hx[2]])
        self.cesb_x = self.c_x + torch.tensor([ 0.5*self.hx[0],-0.5*self.hx[1],-0.5*self.hx[2]])
        self.cest_x = self.c_x + torch.tensor([ 0.5*self.hx[0],-0.5*self.hx[1], 0.5*self.hx[2]])
        self.cenb_x = self.c_x + torch.tensor([ 0.5*self.hx[0], 0.5*self.hx[1],-0.5*self.hx[2]])
        self.cent_x = self.c_x + torch.tensor([ 0.5*self.hx[0], 0.5*self.hx[1], 0.5*self.hx[2]])
        self.cwsb_loc = torch.zeros(self.parm_size,self.c_size)
        self.cwst_loc = torch.zeros(self.parm_size,self.c_size)
        self.cwnb_loc = torch.zeros(self.parm_size,self.c_size)
        self.cwnt_loc = torch.zeros(self.parm_size,self.c_size)
        self.cesb_loc = torch.zeros(self.parm_size,self.c_size)
        self.cest_loc = torch.zeros(self.parm_size,self.c_size)
        self.cenb_loc = torch.zeros(self.parm_size,self.c_size)
        self.cent_loc = torch.zeros(self.parm_size,self.c_size)
        for p in range(self.parm_size):
            self.cwsb_loc[p,:] = self.geo.location(self.cwsb_x[p,:,:], p)
            self.cwst_loc[p,:] = self.geo.location(self.cwst_x[p,:,:], p)
            self.cwnb_loc[p,:] = self.geo.location(self.cwnb_x[p,:,:], p)
            self.cwnt_loc[p,:] = self.geo.location(self.cwnt_x[p,:,:], p)
            self.cesb_loc[p,:] = self.geo.location(self.cesb_x[p,:,:], p)
            self.cest_loc[p,:] = self.geo.location(self.cest_x[p,:,:], p)
            self.cenb_loc[p,:] = self.geo.location(self.cenb_x[p,:,:], p)
            self.cent_loc[p,:] = self.geo.location(self.cent_x[p,:,:], p)
        
        """ neighbor cell """
        self.nw_x = self.c_x + torch.tensor([-self.hx[0],0,0])
        self.ne_x = self.c_x + torch.tensor([ self.hx[0],0,0])
        self.ns_x = self.c_x + torch.tensor([0,-self.hx[1],0])
        self.nn_x = self.c_x + torch.tensor([0, self.hx[1],0])
        self.nb_x = self.c_x + torch.tensor([0,0,-self.hx[2]])
        self.nt_x = self.c_x + torch.tensor([0,0, self.hx[2]])
        self.nw_loc = torch.zeros(self.parm_size,self.c_size)
        self.ne_loc = torch.zeros(self.parm_size,self.c_size)
        self.ns_loc = torch.zeros(self.parm_size,self.c_size)
        self.nn_loc = torch.zeros(self.parm_size,self.c_size)
        self.nb_loc = torch.zeros(self.parm_size,self.c_size)
        self.nt_loc = torch.zeros(self.parm_size,self.c_size)
        for p in range(self.parm_size):
            self.nw_loc[p,:] = self.geo.location(self.nw_x[p,:,:], p)
            self.ne_loc[p,:] = self.geo.location(self.ne_x[p,:,:], p)
            self.ns_loc[p,:] = self.geo.location(self.ns_x[p,:,:], p)
            self.nn_loc[p,:] = self.geo.location(self.nn_x[p,:,:], p)
            self.nb_loc[p,:] = self.geo.location(self.nb_x[p,:,:], p)
            self.nt_loc[p,:] = self.geo.location(self.nt_x[p,:,:], p)
        
        """ cell edge """
        self.ews_st = torch.zeros(self.parm_size,self.c_size); self.ews_ed = torch.zeros(self.parm_size,self.c_size)
        self.ewn_st = torch.zeros(self.parm_size,self.c_size); self.ewn_ed = torch.zeros(self.parm_size,self.c_size)
        self.ewb_st = torch.zeros(self.parm_size,self.c_size); self.ewb_ed = torch.zeros(self.parm_size,self.c_size)
        self.ewt_st = torch.zeros(self.parm_size,self.c_size); self.ewt_ed = torch.zeros(self.parm_size,self.c_size)
        self.esb_st = torch.zeros(self.parm_size,self.c_size); self.esb_ed = torch.zeros(self.parm_size,self.c_size)
        self.est_st = torch.zeros(self.parm_size,self.c_size); self.est_ed = torch.zeros(self.parm_size,self.c_size)
        self.enb_st = torch.zeros(self.parm_size,self.c_size); self.enb_ed = torch.zeros(self.parm_size,self.c_size)
        self.ent_st = torch.zeros(self.parm_size,self.c_size); self.ent_ed = torch.zeros(self.parm_size,self.c_size)
        self.ees_st = torch.zeros(self.parm_size,self.c_size); self.ees_ed = torch.zeros(self.parm_size,self.c_size)
        self.een_st = torch.zeros(self.parm_size,self.c_size); self.een_ed = torch.zeros(self.parm_size,self.c_size)
        self.eeb_st = torch.zeros(self.parm_size,self.c_size); self.eeb_ed = torch.zeros(self.parm_size,self.c_size)
        self.eet_st = torch.zeros(self.parm_size,self.c_size); self.eet_ed = torch.zeros(self.parm_size,self.c_size)
        self.ews_l = torch.zeros(self.parm_size,self.c_size)
        self.ewn_l = torch.zeros(self.parm_size,self.c_size)
        self.ewb_l = torch.zeros(self.parm_size,self.c_size)
        self.ewt_l = torch.zeros(self.parm_size,self.c_size)
        self.esb_l = torch.zeros(self.parm_size,self.c_size)
        self.est_l = torch.zeros(self.parm_size,self.c_size)
        self.enb_l = torch.zeros(self.parm_size,self.c_size)
        self.ent_l = torch.zeros(self.parm_size,self.c_size)
        self.ees_l = torch.zeros(self.parm_size,self.c_size)
        self.een_l = torch.zeros(self.parm_size,self.c_size)
        self.eeb_l = torch.zeros(self.parm_size,self.c_size)
        self.eet_l = torch.zeros(self.parm_size,self.c_size)

        for p in range(self.parm_size):
            idx = (self.c_loc[p,:]==1)

            self.esb_st[p,idx], self.esb_ed[p,idx], self.esb_l[p,idx] = self.cell_edge(
                self.c_x[p,idx,:], self.cwsb_x[p,idx,:], self.cesb_x[p,idx,:], 0, p)
            self.est_st[p,idx], self.est_ed[p,idx], self.est_l[p,idx] = self.cell_edge(
                self.c_x[p,idx,:], self.cwst_x[p,idx,:], self.cest_x[p,idx,:], 0, p)
            self.enb_st[p,idx], self.enb_ed[p,idx], self.enb_l[p,idx] = self.cell_edge(
                self.c_x[p,idx,:], self.cwnb_x[p,idx,:], self.cenb_x[p,idx,:], 0, p)
            self.ent_st[p,idx], self.ent_ed[p,idx], self.ent_l[p,idx] = self.cell_edge(
                self.c_x[p,idx,:], self.cwnt_x[p,idx,:], self.cent_x[p,idx,:], 0, p)
            
            self.ewb_st[p,idx], self.ewb_ed[p,idx], self.ewb_l[p,idx] = self.cell_edge(
                self.c_x[p,idx,:], self.cwsb_x[p,idx,:], self.cwnb_x[p,idx,:], 1, p)
            self.ewt_st[p,idx], self.ewt_ed[p,idx], self.ewt_l[p,idx] = self.cell_edge(
                self.c_x[p,idx,:], self.cwst_x[p,idx,:], self.cwnt_x[p,idx,:], 1, p)
            self.eeb_st[p,idx], self.eeb_ed[p,idx], self.eeb_l[p,idx] = self.cell_edge(
                self.c_x[p,idx,:], self.cesb_x[p,idx,:], self.cenb_x[p,idx,:], 1, p)
            self.eet_st[p,idx], self.eet_ed[p,idx], self.eet_l[p,idx] = self.cell_edge(
                self.c_x[p,idx,:], self.cest_x[p,idx,:], self.cent_x[p,idx,:], 1, p)

            self.ews_st[p,idx], self.ews_ed[p,idx], self.ews_l[p,idx] = self.cell_edge(
                self.c_x[p,idx,:], self.cwsb_x[p,idx,:], self.cwst_x[p,idx,:], 2, p)
            self.ewn_st[p,idx], self.ewn_ed[p,idx], self.ewn_l[p,idx] = self.cell_edge(
                self.c_x[p,idx,:], self.cwnb_x[p,idx,:], self.cwnt_x[p,idx,:], 2, p)
            self.ees_st[p,idx], self.ees_ed[p,idx], self.ees_l[p,idx] = self.cell_edge(
                self.c_x[p,idx,:], self.cesb_x[p,idx,:], self.cest_x[p,idx,:], 2, p)
            self.een_st[p,idx], self.een_ed[p,idx], self.een_l[p,idx] = self.cell_edge(
                self.c_x[p,idx,:], self.cenb_x[p,idx,:], self.cent_x[p,idx,:], 2, p)
        
        """ cell face """
        self.fw_x_in = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fe_x_in = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fs_x_in = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fn_x_in = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fb_x_in = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.ft_x_in = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fw_x_bd = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fe_x_bd = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fs_x_bd = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fn_x_bd = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fb_x_bd = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.ft_x_bd = torch.zeros(self.parm_size,self.c_size,self.dim)
        self.fw_a_in = torch.zeros(self.parm_size,self.c_size); self.fe_a_in = torch.zeros(self.parm_size,self.c_size)
        self.fs_a_in = torch.zeros(self.parm_size,self.c_size); self.fn_a_in = torch.zeros(self.parm_size,self.c_size)
        self.fb_a_in = torch.zeros(self.parm_size,self.c_size); self.ft_a_in = torch.zeros(self.parm_size,self.c_size)
        self.fw_a_bd = torch.zeros(self.parm_size,self.c_size); self.fe_a_bd = torch.zeros(self.parm_size,self.c_size)
        self.fs_a_bd = torch.zeros(self.parm_size,self.c_size); self.fn_a_bd = torch.zeros(self.parm_size,self.c_size)
        self.fb_a_bd = torch.zeros(self.parm_size,self.c_size); self.ft_a_bd = torch.zeros(self.parm_size,self.c_size)
        
        for p in range(self.parm_size):
            idx = (self.c_loc[p,:]==1)
        
            # west face
            self.fw_x_in[p,idx,:], self.fw_x_bd[p,idx,:], self.fw_a_in[p,idx], self.fw_a_bd[p,idx] = self.cell_face(
                self.c_x[p,idx,:], self.cwsb_x[p,idx,:], self.cwst_x[p,idx,:], self.cwnb_x[p,idx,:], self.cwnt_x[p,idx,:],
                self.ews_st[p,idx], self.ews_ed[p,idx], self.ewn_st[p,idx], self.ewn_ed[p,idx],
                self.ewb_st[p,idx], self.ewb_ed[p,idx], self.ewt_st[p,idx], self.ewt_ed[p,idx], 1, p)
            
            # east face
            self.fe_x_in[p,idx,:], self.fe_x_bd[p,idx,:], self.fe_a_in[p,idx], self.fe_a_bd[p,idx] = self.cell_face(
                self.c_x[p,idx,:], self.cesb_x[p,idx,:], self.cest_x[p,idx,:], self.cenb_x[p,idx,:], self.cent_x[p,idx,:], 
                self.ees_st[p,idx], self.ees_ed[p,idx], self.een_st[p,idx], self.een_ed[p,idx], 
                self.eeb_st[p,idx], self.eeb_ed[p,idx], self.eet_st[p,idx], self.eet_ed[p,idx], -1, p)
            
            # south face
            self.fs_x_in[p,idx,:], self.fs_x_bd[p,idx,:], self.fs_a_in[p,idx], self.fs_a_bd[p,idx] = self.cell_face(
                self.c_x[p,idx,:], self.cwsb_x[p,idx,:], self.cwst_x[p,idx,:], self.cesb_x[p,idx,:], self.cest_x[p,idx,:], 
                self.ews_st[p,idx], self.ews_ed[p,idx], self.ees_st[p,idx], self.ees_ed[p,idx], 
                self.esb_st[p,idx], self.esb_ed[p,idx], self.est_st[p,idx], self.est_ed[p,idx], 2, p)
            
            # north face
            self.fn_x_in[p,idx,:], self.fn_x_bd[p,idx,:], self.fn_a_in[p,idx], self.fn_a_bd[p,idx] = self.cell_face(
                self.c_x[p,idx,:], self.cwnb_x[p,idx,:], self.cwnt_x[p,idx,:], self.cenb_x[p,idx,:], self.cent_x[p,idx,:],
                self.ewn_st[p,idx], self.ewn_ed[p,idx], self.een_st[p,idx], self.een_ed[p,idx],
                self.enb_st[p,idx], self.enb_ed[p,idx], self.ent_st[p,idx], self.ent_ed[p,idx], -2, p)
            
            # bottom face
            self.fb_x_in[p,idx,:], self.fb_x_bd[p,idx,:], self.fb_a_in[p,idx], self.fb_a_bd[p,idx] = self.cell_face(
                self.c_x[p,idx,:], self.cwsb_x[p,idx,:], self.cwnb_x[p,idx,:], self.cesb_x[p,idx,:], self.cenb_x[p,idx,:],
                self.ewb_st[p,idx], self.ewb_ed[p,idx], self.eeb_st[p,idx], self.eeb_ed[p,idx],
                self.esb_st[p,idx], self.esb_ed[p,idx], self.enb_st[p,idx], self.enb_ed[p,idx], 3, p)
            
            # top face
            self.ft_x_in[p,idx,:], self.ft_x_bd[p,idx,:], self.ft_a_in[p,idx], self.ft_a_bd[p,idx] = self.cell_face(
                self.c_x[p,idx,:], self.cwst_x[p,idx,:], self.cwnt_x[p,idx,:], self.cest_x[p,idx,:], self.cent_x[p,idx,:],
                self.ewt_st[p,idx], self.ewt_ed[p,idx], self.eet_st[p,idx], self.eet_ed[p,idx],
                self.est_st[p,idx], self.est_ed[p,idx], self.ent_st[p,idx], self.ent_ed[p,idx], -3, p)
        
        self.c_v = torch.zeros(self.parm_size,self.c_size,1)
        for p in range(self.parm_size):
            for i in range(self.nx[0]):
                for j in range(self.nx[1]):
                    for k in range(self.nx[2]):
                        m = (i*self.nx[1] + j)*self.nx[2] + k
                        if self.c_loc[p,m]!=1:
                            continue

                        self.c_v[p,m] = self.cell_volumn(m, p)
        
    def cell_edge(self, c_x, x0, x1, d, p):
        tol = 1e-2

        x0_n = x0 + tol*(c_x-x0)
        x1_n = x1 + tol*(c_x-x1)
        loc0_n = self.geo.location(x0_n, p)
        loc1_n = self.geo.location(x1_n, p)

        if d==0:
            h = self.hx[0]
        if d==1:
            h = self.hx[1]
        if d==2:
            h = self.hx[2]
        
        st = torch.zeros(x0.shape[0])
        ed = torch.zeros(x0.shape[0])
        
        idx = ((loc0_n==1) & (loc1_n==1))
        st[idx] = 0.0; ed[idx] = 1.0

        idx = ((loc0_n==1) & (loc1_n!=1))
        x = self.geo.intersection(x0[idx,:], x1[idx,:], p)
        st[idx] = 0.0; ed[idx] = (x[:,d]-x0[idx,d])/h

        idx = ((loc0_n!=1) & (loc1_n==1))
        x = self.geo.intersection(x0[idx,:], x1[idx,:], p)
        st[idx] = (x[:,d]-x0[idx,d])/h; ed[idx] = 1.0

        idx = ((loc0_n!=1) & (loc1_n!=1))
        st[idx] = 0.0; ed[idx] = 1.0
        
        l = (ed-st) * h
        return st, ed, l

    def cell_face(self, c_x, x_ws, x_wn, x_es, x_en, w_st, w_ed, e_st, e_ed, s_st, s_ed, n_st, n_ed, d, p):
        w_st = w_st.view(-1,1)
        w_ed = w_ed.view(-1,1)
        e_st = e_st.view(-1,1)
        e_ed = e_ed.view(-1,1)
        s_st = s_st.view(-1,1)
        s_ed = s_ed.view(-1,1)
        n_st = n_st.view(-1,1)
        n_ed = n_ed.view(-1,1)

        tol = 1e-2

        x_ws_n = x_ws + tol*(c_x-x_ws)
        x_wn_n = x_wn + tol*(c_x-x_wn)
        x_es_n = x_es + tol*(c_x-x_es)
        x_en_n = x_en + tol*(c_x-x_en)
        loc_ws_n = self.geo.location(x_ws_n, p)
        loc_wn_n = self.geo.location(x_wn_n, p)
        loc_es_n = self.geo.location(x_es_n, p)
        loc_en_n = self.geo.location(x_en_n, p)

        x_in = torch.zeros(c_x.shape[0],2)
        x_bd = torch.zeros(c_x.shape[0],2)
        a_in = torch.zeros(c_x.shape[0],1)
        a_bd = torch.zeros(c_x.shape[0],1)

        if d==1 or d==-1:
            h = torch.tensor([self.hx[1],self.hx[2]])
        if d==2 or d==-2:
            h = torch.tensor([self.hx[0],self.hx[2]])
        if d==3 or d==-3:
            h = torch.tensor([self.hx[0],self.hx[1]])

        tol = 1e-6
        
        # 4 corner in the domain
        idx = ((loc_ws_n==1) & (loc_wn_n==1) & (loc_es_n==1) & (loc_en_n==1))
        x_in[idx,:] = torch.tensor([0.5*h[0],0.5*h[1]])
        x_bd[idx,:] = torch.tensor([0.0,0.0])
        a_in[idx,:] = h[0]*h[1]
        a_bd[idx,:] = 0.0
        
        # 3 corner in the domain
        idx = ((loc_ws_n!=1) & (loc_wn_n==1) & (loc_es_n==1) & (loc_en_n==1))
        a1 = s_st*w_st
        a2 = s_st*(1-w_st)
        a3 = (1-s_st)*w_st
        a4 = (1-s_st)*(1-w_st)
        x1 = torch.cat([(0.5*s_st)*h[0],(0.5*w_st)*h[1]],1)
        x2 = torch.cat([(0.5*s_st)*h[0],(0.5*w_st+0.5)*h[1]],1)
        x3 = torch.cat([(0.5*s_st+0.5)*h[0],(0.5*w_st)*h[1]],1)
        x4 = torch.cat([(0.5*s_st+0.5)*h[0],(0.5*w_st+0.5)*h[1]],1)
        a_bd[idx,:] = (a1 * h[0]*h[1])[idx,:]
        a_in[idx,:] = ((a2+a3+a4) * h[0]*h[1])[idx,:]
        x_bd[idx,:] = (x1)[idx,:]
        x_in[idx,:] = ((a2*x2+a3*x3+a4*x4)/(a2+a3+a4+tol))[idx,:]
        
        idx = ((loc_ws_n==1) & (loc_wn_n!=1) & (loc_es_n==1) & (loc_en_n==1))
        a1 = n_st*w_ed
        a2 = n_st*(1-w_ed)
        a3 = (1-n_st)*w_ed
        a4 = (1-n_st)*(1-w_ed)
        x1 = torch.cat([(0.5*n_st)*h[0],(0.5*w_ed)*h[1]],1)
        x2 = torch.cat([(0.5*n_st)*h[0],(0.5*w_ed+0.5)*h[1]],1)
        x3 = torch.cat([(0.5*n_st+0.5)*h[0],(0.5*w_ed)*h[1]],1)
        x4 = torch.cat([(0.5*n_st+0.5)*h[0],(0.5*w_ed+0.5)*h[1]],1)
        a_bd[idx,:] = (a2 * h[0]*h[1])[idx,:]
        a_in[idx,:] = ((a1+a3+a4) * h[0]*h[1])[idx,:]
        x_bd[idx,:] = (x2)[idx,:]
        x_in[idx,:] = ((a1*x1+a3*x3+a4*x4)/(a1+a3+a4+tol))[idx,:]
        
        idx = ((loc_ws_n==1) & (loc_wn_n==1) & (loc_es_n!=1) & (loc_en_n==1))
        a1 = s_ed*e_st
        a2 = s_ed*(1-e_st)
        a3 = (1-s_ed)*e_st
        a4 = (1-s_ed)*(1-e_st)
        x1 = torch.cat([(0.5*s_ed)*h[0],(0.5*e_st)*h[1]],1)
        x2 = torch.cat([(0.5*s_ed)*h[0],(0.5*e_st+0.5)*h[1]],1)
        x3 = torch.cat([(0.5*s_ed+0.5)*h[0],(0.5*e_st)*h[1]],1)
        x4 = torch.cat([(0.5*s_ed+0.5)*h[0],(0.5*e_st+0.5)*h[1]],1)
        a_bd[idx,:] = (a3 * h[0]*h[1])[idx,:]
        a_in[idx,:] = ((1-(1-s_ed)*e_st) * h[0]*h[1])[idx,:]
        x_bd[idx,:] = (x3)[idx,:]
        x_in[idx,:] = ((a1*x1+a2*x2+a4*x4)/(a1+a2+a4+tol))[idx,:]
        
        idx = ((loc_ws_n==1) & (loc_wn_n==1) & (loc_es_n==1) & (loc_en_n!=1))
        a1 = n_ed*e_ed
        a2 = n_ed*(1-e_ed)
        a3 = (1-n_ed)*e_ed
        a4 = (1-n_ed)*(1-e_ed)
        x1 = torch.cat([(0.5*n_ed)*h[0],(0.5*e_ed)*h[1]],1)
        x2 = torch.cat([(0.5*n_ed)*h[0],(0.5*e_ed+0.5)*h[1]],1)
        x3 = torch.cat([(0.5*n_ed+0.5)*h[0],(0.5*e_ed)*h[1]],1)
        x4 = torch.cat([(0.5*n_ed+0.5)*h[0],(0.5*e_ed+0.5)*h[1]],1)
        a_bd[idx,:] = (a4 * h[0]*h[1])[idx,:]
        a_in[idx,:] = ((a1+a2+a3) * h[0]*h[1])[idx,:]
        x_bd[idx,:] = (x4)[idx,:]
        x_in[idx,:] = ((a1*x1+a2*x2+a3*x3)/(a1+a2+a3+tol))[idx,:]
        
        # 2 corner in the domain
        idx = ((loc_ws_n!=1) & (loc_wn_n!=1) & (loc_es_n==1) & (loc_en_n==1))
        x_in[idx,0:1] = ((0.5*s_st+0.5)*h[0])[idx,:]
        x_in[idx,1:2] = 0.5*h[1]
        x_bd[idx,0:1] = ((0.5*s_st)*h[0])[idx,:]
        x_bd[idx,1:2] = 0.5*h[1]
        a_in[idx,:] = ((1-s_st)*h[0]*h[1])[idx,:]
        a_bd[idx,:] = (s_st * h[0]*h[1])[idx,:]
        
        idx = ((loc_ws_n==1) & (loc_wn_n==1) & (loc_es_n!=1) & (loc_en_n!=1))
        x_in[idx,0:1] = ((0.5*s_ed)*h[0])[idx,:]
        x_in[idx,1:2] = 0.5*h[1]
        x_bd[idx,0:1] = ((0.5*s_ed+0.5)*h[0])[idx,:]
        x_bd[idx,1:2] = 0.5*h[1]
        a_in[idx,:] = (s_ed * h[0]*h[1])[idx,:]
        a_bd[idx,:] = ((1-s_ed) * h[0]*h[1])[idx,:]
        
        idx = ((loc_ws_n!=1) & (loc_wn_n==1) & (loc_es_n!=1) & (loc_en_n==1))
        x_in[idx,0:1] = 0.5*h[0]
        x_in[idx,1:2] = ((0.5*w_st+0.5)*h[1])[idx,:]
        x_bd[idx,0:1] = 0.5*h[0]
        x_bd[idx,1:2] = ((0.5*w_st)*h[1])[idx,:]
        a_in[idx,:] = ((1-w_st) * h[0]*h[1])[idx,:]
        a_bd[idx,:] = (w_st * h[0]*h[1])[idx,:]
        
        idx = ((loc_ws_n==1) & (loc_wn_n!=1) & (loc_es_n==1) & (loc_en_n!=1))
        x_in[idx,0:1] = 0.5*h[0]
        x_in[idx,1:2] = ((0.5*w_ed)*h[1])[idx,:]
        x_bd[idx,0:1] = 0.5*h[0]
        x_bd[idx,1:2] = ((0.5*w_ed+0.5)*h[1])[idx,:]
        a_in[idx,:] = (w_ed * h[0]*h[1])[idx,:]
        a_bd[idx,:] = ((1-w_ed) * h[0]*h[1])[idx,:]
        
        # 1 corner in the domain
        idx = ((loc_ws_n==1) & (loc_wn_n!=1) & (loc_es_n!=1) & (loc_en_n!=1))
        a1 = s_ed*w_ed
        a2 = s_ed*(1-w_ed)
        a3 = (1-s_ed)*w_ed
        a4 = (1-s_ed)*(1-w_ed)
        x1 = torch.cat([(0.5*s_ed)*h[0],(0.5*w_ed)*h[1]],1)
        x2 = torch.cat([(0.5*s_ed)*h[0],(0.5*w_ed+0.5)*h[1]],1)
        x3 = torch.cat([(0.5*s_ed+0.5)*h[0],(0.5*w_ed)*h[1]],1)
        x4 = torch.cat([(0.5*s_ed+0.5)*h[0],(0.5*w_ed+0.5)*h[1]],1)
        a_in[idx,:] = (a1 * h[0]*h[1])[idx,:]
        a_bd[idx,:] = ((a2+a3+a4) * h[0]*h[1])[idx,:]
        x_in[idx,:] = (x1)[idx,:]
        x_bd[idx,:] = ((a2*x2+a3*x3+a4*x4)/(a2+a3+a4+tol))[idx,:]
        
        idx = ((loc_ws_n!=1) & (loc_wn_n==1) & (loc_es_n!=1) & (loc_en_n!=1))
        a1 = n_ed*w_st
        a2 = n_ed*(1-w_st)
        a3 = (1-n_ed)*w_st
        a4 = (1-n_ed)*(1-w_st)
        x1 = torch.cat([(0.5*n_ed)*h[0],(0.5*w_st)*h[1]],1)
        x2 = torch.cat([(0.5*n_ed)*h[0],(0.5*w_st+0.5)*h[1]],1)
        x3 = torch.cat([(0.5*n_ed+0.5)*h[0],(0.5*w_st)*h[1]],1)
        x4 = torch.cat([(0.5*n_ed+0.5)*h[0],(0.5*w_st+0.5)*h[1]],1)
        a_in[idx,:] = (a2 * h[0]*h[1])[idx,:]
        a_bd[idx,:] = ((a1+a3+a4) * h[0]*h[1])[idx,:]
        x_in[idx,:] = (x2)[idx,:]
        x_bd[idx,:] = ((a1*x1+a3*x3+a4*x4)/(a1+a3+a4+tol))[idx,:]
        
        idx = ((loc_ws_n!=1) & (loc_wn_n!=1) & (loc_es_n==1) & (loc_en_n!=1))
        a1 = s_st*e_ed
        a2 = s_st*(1-e_ed)
        a3 = (1-s_st)*e_ed
        a4 = (1-s_st)*(1-e_ed)
        x1 = torch.cat([(0.5*s_st)*h[0],(0.5*e_ed)*h[1]],1)
        x2 = torch.cat([(0.5*s_st)*h[0],(0.5*e_ed+0.5)*h[1]],1)
        x3 = torch.cat([(0.5*s_st+0.5)*h[0],(0.5*e_ed)*h[1]],1)
        x4 = torch.cat([(0.5*s_st+0.5)*h[0],(0.5*e_ed+0.5)*h[1]],1)
        a_in[idx,:] = (a3 * h[0]*h[1])[idx,:]
        a_bd[idx,:] = ((a1+a2+a4) * h[0]*h[1])[idx,:]
        x_in[idx,:] = (x3)[idx,:]
        x_bd[idx,:] = ((a1*x1+a2*x2+a4*x4)/(a1+a2+a4+tol))[idx,:]
        
        idx = ((loc_ws_n!=1) & (loc_wn_n!=1) & (loc_es_n!=1) & (loc_en_n==1))
        a1 = n_st*e_st
        a2 = n_st*(1-e_st)
        a3 = (1-n_st)*e_st
        a4 = (1-n_st)*(1-e_st)
        x1 = torch.cat([(0.5*n_st)*h[0],(0.5*e_st)*h[1]],1)
        x2 = torch.cat([(0.5*n_st)*h[0],(0.5*e_st+0.5)*h[1]],1)
        x3 = torch.cat([(0.5*n_st+0.5)*h[0],(0.5*e_st)*h[1]],1)
        x4 = torch.cat([(0.5*n_st+0.5)*h[0],(0.5*e_st+0.5)*h[1]],1)
        a_in[idx,:] = (a4 * h[0]*h[1])[idx,:]
        a_bd[idx,:] = ((a1+a2+a3) * h[0]*h[1])[idx,:]
        x_in[idx,:] = (x4)[idx,:]
        x_bd[idx,:] = ((a1*x1+a2*x2+a3*x3)/(a1+a2+a3+tol))[idx,:]
        
        # 0 corner in the domain
        idx = ((loc_ws_n!=1) & (loc_wn_n!=1) & (loc_es_n!=1) & (loc_en_n!=1))
        x_bd[idx,:] = torch.tensor([0.5*h[0],0.5*h[1]])
        x_in[idx,:] = torch.tensor([0.0,0.0])
        a_bd[idx,:] = h[0]*h[1]
        a_in[idx,:] = 0.0

        # Elevate to three dimensions
        if d==1 or d==-1:
            x_in = torch.stack([torch.zeros(x_in.shape[0]),x_in[:,0],x_in[:,1]],1)
            x_bd = torch.stack([torch.zeros(x_in.shape[0]),x_bd[:,0],x_bd[:,1]],1)
        if d==2 or d==-2:
            x_in = torch.stack([x_in[:,0],torch.zeros(x_in.shape[0]),x_in[:,1]],1)
            x_bd = torch.stack([x_bd[:,0],torch.zeros(x_in.shape[0]),x_bd[:,1]],1)
        if d==3 or d==-3:
            x_in = torch.stack([x_in[:,0],x_in[:,1],torch.zeros(x_in.shape[0])],1)
            x_bd = torch.stack([x_bd[:,0],x_bd[:,1],torch.zeros(x_in.shape[0])],1)
        
        idx = (a_bd > tol)[:,0]
        x0 = torch.zeros(x_ws.shape[0],self.dim)
        x1 = torch.zeros(x_ws.shape[0],self.dim)
        if d==1:
            x0[idx,:] = x_ws[idx,:] + torch.stack([self.hx[0]*torch.ones(idx.sum()),x_bd[idx,1],x_bd[idx,2]],1)
            x1[idx,:] = x_ws[idx,:] + torch.stack([          torch.zeros(idx.sum()),x_bd[idx,1],x_bd[idx,2]],1)
        if d==-1:
            x0[idx,:] = x_ws[idx,:] + torch.stack([-self.hx[0]*torch.ones(idx.sum()),x_bd[idx,1],x_bd[idx,2]],1)
            x1[idx,:] = x_ws[idx,:] + torch.stack([           torch.zeros(idx.sum()),x_bd[idx,1],x_bd[idx,2]],1)
        if d==2:
            x0[idx,:] = x_ws[idx,:] + torch.stack([x_bd[idx,0],self.hx[1]*torch.ones(idx.sum()),x_bd[idx,2]],1)
            x1[idx,:] = x_ws[idx,:] + torch.stack([x_bd[idx,0],          torch.zeros(idx.sum()),x_bd[idx,2]],1)
        if d==-2:
            x0[idx,:] = x_ws[idx,:] + torch.stack([x_bd[idx,0],-self.hx[1]*torch.ones(idx.sum()),x_bd[idx,2]],1)
            x1[idx,:] = x_ws[idx,:] + torch.stack([x_bd[idx,0],           torch.zeros(idx.sum()),x_bd[idx,2]],1)
        if d==3:
            x0[idx,:] = x_ws[idx,:] + torch.stack([x_bd[idx,0],x_bd[idx,1],self.hx[2]*torch.ones(idx.sum())],1)
            x1[idx,:] = x_ws[idx,:] + torch.stack([x_bd[idx,0],x_bd[idx,1],          torch.zeros(idx.sum())],1)
        if d==-3:
            x0[idx,:] = x_ws[idx,:] + torch.stack([x_bd[idx,0],x_bd[idx,1],-self.hx[2]*torch.ones(idx.sum())],1)
            x1[idx,:] = x_ws[idx,:] + torch.stack([x_bd[idx,0],x_bd[idx,1],           torch.zeros(idx.sum())],1)
        
        idx1 = idx & (self.geo.location(x0, p)==1)
        x = self.geo.intersection(x0[idx1,:], x1[idx1,:], p)
        x_bd[idx1,:] = x - x_ws[idx1,:]
        
        idx2 = idx & (self.geo.location(x0, p)!=1)
        a_bd[idx2,:] = 0
        x_bd[idx2,:] = torch.tensor([0.0,0.0,0.0])
        
        # transform to original axis
        x_in = x_ws + x_in
        x_bd = x_ws + x_bd

        # several x_in are located on the boundary, conduct correction
        loc = self.geo.location(x_in, p)
        idx = ((loc==0) & (a_in > tol)[:,0])
        x_bd[idx,:] = x_in[idx,:].clone()
        a_bd[idx,:] = a_in[idx,:].clone()
        x_in[idx,:] = x_ws[idx,:].clone()
        a_in[idx,:] = 0

        # filter area with tiny value
        idx = (a_in < tol)
        a_in[idx] = 0.0
        idx = (a_bd < tol)
        a_bd[idx] = 0.0
        
        return x_in, x_bd, a_in[:,0], a_bd[:,0]

    def cell_volumn(self, m, p):
        c_v = self.hx[0]*self.hx[1]*self.hx[2]

        if self.cwsb_loc[p,m]!=1 and self.cwst_loc[p,m]==1 and self.cwnb_loc[p,m]==1 and self.cesb_loc[p,m]==1:
            c_v -= (self.esb_st[p,m]*self.hx[0]) * (self.ewb_st[p,m]*self.hx[1]) * (self.ews_st[p,m]*self.hx[2])
        if self.cwst_loc[p,m]!=1 and self.cwsb_loc[p,m]==1 and self.cwnt_loc[p,m]==1 and self.cest_loc[p,m]==1:
            c_v -= (self.est_st[p,m]*self.hx[0]) * (self.ewt_st[p,m]*self.hx[1]) * ((1-self.ews_ed[p,m])*self.hx[2])
        if self.cwnb_loc[p,m]!=1 and self.cwnt_loc[p,m]==1 and self.cwsb_loc[p,m]==1 and self.cenb_loc[p,m]==1:
            c_v -= (self.enb_st[p,m]*self.hx[0]) * ((1-self.ewb_ed[p,m])*self.hx[1]) * (self.ewn_st[p,m]*self.hx[2])
        if self.cwnt_loc[p,m]!=1 and self.cwnb_loc[p,m]==1 and self.cwst_loc[p,m]==1 and self.cent_loc[p,m]==1:
            c_v -= (self.ent_st[p,m]*self.hx[0]) * ((1-self.ewt_ed[p,m])*self.hx[1]) * ((1-self.ewn_ed[p,m])*self.hx[2])
        if self.cesb_loc[p,m]!=1 and self.cest_loc[p,m]==1 and self.cenb_loc[p,m]==1 and self.cwsb_loc[p,m]==1:
            c_v -= ((1-self.esb_ed[p,m])*self.hx[0]) * (self.eeb_st[p,m]*self.hx[1]) * (self.ees_st[p,m]*self.hx[2])
        if self.cest_loc[p,m]!=1 and self.cesb_loc[p,m]==1 and self.cent_loc[p,m]==1 and self.cwst_loc[p,m]==1:
            c_v -= ((1-self.est_ed[p,m])*self.hx[0]) * (self.eet_st[p,m]*self.hx[1]) * ((1-self.ees_ed[p,m])*self.hx[2])
        if self.cenb_loc[p,m]!=1 and self.cent_loc[p,m]==1 and self.cesb_loc[p,m]==1 and self.cwnb_loc[p,m]==1:
            c_v -= ((1-self.enb_ed[p,m])*self.hx[0]) * ((1-self.eeb_ed[p,m])*self.hx[1]) * (self.een_st[p,m]*self.hx[2])
        if self.cent_loc[p,m]!=1 and self.cenb_loc[p,m]==1 and self.cest_loc[p,m]==1 and self.cwnt_loc[p,m]==1:
            c_v -= ((1-self.ent_ed[p,m])*self.hx[0]) * ((1-self.eet_ed[p,m])*self.hx[1]) * ((1-self.een_ed[p,m])*self.hx[2])

        if (self.cwsb_loc[p,m]!=1 and self.cwst_loc[p,m]!=1 and 
            (self.cwnb_loc[p,m]==1 or self.cwnt_loc[p,m]==1) and (self.cwnb_loc[p,m]==1 or self.cwnt_loc[p,m]==1)):
            c_v -= (self.esb_st[p,m]*self.hx[0]) * (self.ewb_st[p,m]*self.hx[1]) * self.hx[2]
        if (self.cwnb_loc[p,m]!=1 and self.cwnt_loc[p,m]!=1 and 
            (self.cwsb_loc[p,m]==1 or self.cwst_loc[p,m]==1) and (self.cenb_loc[p,m]==1 or self.cent_loc[p,m]==1)):
            c_v -= (self.enb_st[p,m]*self.hx[0]) * ((1-self.ewb_ed[p,m])*self.hx[1]) * self.hx[2]
        if (self.cesb_loc[p,m]!=1 and self.cest_loc[p,m]!=1 and 
            (self.cenb_loc[p,m]==1 or self.cent_loc[p,m]==1) and (self.cwsb_loc[p,m]==1 or self.cwst_loc[p,m]==1)):
            c_v -= ((1-self.esb_ed[p,m])*self.hx[0]) * (self.eeb_st[p,m]*self.hx[1]) * self.hx[2]
        if (self.cenb_loc[p,m]!=1 and self.cent_loc[p,m]!=1 and 
            (self.cesb_loc[p,m]==1 or self.cest_loc[p,m]==1) and (self.cwnb_loc[p,m]==1 or self.cwnt_loc[p,m]==1)):
            c_v -= ((1-self.enb_ed[p,m])*self.hx[0]) * ((1-self.eeb_ed[p,m])*self.hx[1]) * self.hx[2]

        if (self.cwsb_loc[p,m]!=1 and self.cwnb_loc[p,m]!=1 and 
            (self.cwst_loc[p,m]==1 or self.cwnt_loc[p,m]==1) and (self.cesb_loc[p,m]==1 or self.cenb_loc[p,m]==1)):
            c_v -= (self.esb_st[p,m]*self.hx[0]) * self.hx[1] * (self.ews_st[p,m]*self.hx[2])
        if (self.cwst_loc[p,m]!=1 and self.cwnt_loc[p,m]!=1 and 
            (self.cwsb_loc[p,m]==1 or self.cwnb_loc[p,m]==1) and (self.cest_loc[p,m]==1 or self.cent_loc[p,m]==1)):
            c_v -= (self.est_st[p,m]*self.hx[0]) * self.hx[1] * ((1-self.ews_ed[p,m])*self.hx[2])
        if (self.cesb_loc[p,m]!=1 and self.cenb_loc[p,m]!=1 and 
            (self.cest_loc[p,m]==1 or self.cent_loc[p,m]==1) and (self.cwsb_loc[p,m]==1 or self.cwnb_loc[p,m]==1)):
            c_v -= ((1-self.esb_ed[p,m])*self.hx[0]) * self.hx[1] * (self.ees_st[p,m]*self.hx[2])
        if (self.cest_loc[p,m]!=1 and self.cent_loc[p,m]!=1 and 
            (self.cesb_loc[p,m]==1 or self.cenb_loc[p,m]==1) and (self.cwst_loc[p,m]==1 or self.cwnt_loc[p,m]==1)):
            c_v -= ((1-self.est_ed[p,m])*self.hx[0]) * self.hx[1] * ((1-self.ees_ed[p,m])*self.hx[2])

        if (self.cwsb_loc[p,m]!=1 and self.cesb_loc[p,m]!=1 and 
            (self.cwst_loc[p,m]==1 or self.cest_loc[p,m]==1) and (self.cwnb_loc[p,m]==1 or self.cenb_loc[p,m]==1)):
            c_v -= self.hx[0] * (self.ewb_st[p,m]*self.hx[1]) * (self.ews_st[p,m]*self.hx[2])
        if (self.cwst_loc[p,m]!=1 and self.cest_loc[p,m]!=1 and 
            (self.cwsb_loc[p,m]==1 or self.cesb_loc[p,m]==1) and (self.cwnt_loc[p,m]==1 or self.cent_loc[p,m]==1)):
            c_v -= self.hx[0] * (self.ewt_st[p,m]*self.hx[1]) * ((1-self.ews_ed[p,m])*self.hx[2])
        if (self.cwnb_loc[p,m]!=1 and self.cenb_loc[p,m]!=1 and 
            (self.cwnt_loc[p,m]==1 or self.cent_loc[p,m]==1) and (self.cwsb_loc[p,m]==1 or self.cesb_loc[p,m]==1)):
            c_v -= self.hx[0] * ((1-self.ewb_ed[p,m])*self.hx[1]) * (self.ewn_st[p,m]*self.hx[2])
        if (self.cwnt_loc[p,m]!=1 and self.cent_loc[p,m]!=1 and 
            (self.cwnb_loc[p,m]==1 or self.cenb_loc[p,m]==1) and (self.cwst_loc[p,m]==1 or self.cest_loc[p,m]==1)):
            c_v -= self.hx[0] * ((1-self.ewt_ed[p,m])*self.hx[1]) * ((1-self.ewn_ed[p,m])*self.hx[2])

        if (self.cwsb_loc[p,m]!=1 and self.cwst_loc[p,m]!=1 and self.cwnb_loc[p,m]!=1 and self.cwnt_loc[p,m]!=1):
            c_v -= (self.esb_st[p,m]*self.hx[0]) * self.hx[1] * self.hx[2]
        if (self.cesb_loc[p,m]!=1 and self.cest_loc[p,m]!=1 and self.cenb_loc[p,m]!=1 and self.cent_loc[p,m]!=1):
            c_v -= ((1-self.esb_ed[p,m])*self.hx[0]) * self.hx[1] * self.hx[2]
        if (self.cwsb_loc[p,m]!=1 and self.cwst_loc[p,m]!=1 and self.cesb_loc[p,m]!=1 and self.cest_loc[p,m]!=1):
            c_v -= self.hx[0] * (self.ewb_st[p,m]*self.hx[1]) * self.hx[2]
        if (self.cwnb_loc[p,m]!=1 and self.cwnt_loc[p,m]!=1 and self.cenb_loc[p,m]!=1 and self.cent_loc[p,m]!=1):
            c_v -= self.hx[0] * ((1-self.ewb_ed[p,m])*self.hx[1]) * self.hx[2]
        if (self.cwsb_loc[p,m]!=1 and self.cwnb_loc[p,m]!=1 and self.cesb_loc[p,m]!=1 and self.cenb_loc[p,m]!=1):
            c_v -= self.hx[0] * self.hx[1] * (self.ews_st[p,m]*self.hx[2])
        if (self.cwst_loc[p,m]!=1 and self.cwnt_loc[p,m]!=1 and self.cest_loc[p,m]!=1 and self.cent_loc[p,m]!=1):
            c_v -= self.hx[0] * self.hx[1] * ((1-self.ews_ed[p,m])*self.hx[2])
        
        return c_v
