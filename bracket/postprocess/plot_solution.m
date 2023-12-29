clear; clc
format short
format compact

tol = 1e-4;

%%
geo.dim = 3;
geo.center = [0.0,0.0]; geo.radius = 0.5;

geo.p1_xa = -1.0; geo.p1_xb = -0.8; geo.p1_ya = -1.0; geo.p1_yb = 1.0;
geo.p1_za = -1.0; geo.p1_zb = 1.0;
geo.p2_xa = -1.0; geo.p2_xb = 1.0; geo.p2_ya = -1.0; geo.p2_yb = 1.0;
geo.p2_za = -0.2; geo.p2_zb = 0.2;

geo.ta = 0.0; geo.tb = 2*pi;
geo.nt = 200; geo.ht = (geo.tb-geo.ta)/geo.nt;
geo.ht = (geo.tb-geo.ta)/geo.nt;
geo.tt = geo.ta:geo.ht:geo.tb;

%%
mesh.dim = 3;
mesh.xa = -1.0; mesh.xb = 1.0; mesh.ya = -1.0; mesh.yb = 1.0;
mesh.za = -1.0; mesh.zb = 1.0;
mesh.nx = 20; mesh.ny = 20; mesh.nz = 20;
mesh.hx = (mesh.xb-mesh.xa)/mesh.nx;
mesh.hy = (mesh.yb-mesh.ya)/mesh.ny;
mesh.hz = (mesh.zb-mesh.za)/mesh.nz;
mesh.xx = mesh.xa+mesh.hx/2:mesh.hx:mesh.xb-mesh.hx/2;
mesh.yy = mesh.ya+mesh.hy/2:mesh.hy:mesh.yb-mesh.hy/2;
mesh.zz = mesh.za+mesh.hz/2:mesh.hz:mesh.zb-mesh.hz/2;

mesh.c_size = mesh.nx*mesh.ny*mesh.nz;

%%
solution = xlsread('solution.csv');
x1 = solution(:,1); x2 = solution(:,2); x3 = solution(:,3);
u1 = solution(:,4); u2 = solution(:,5); u3 = solution(:,6);
mask = solution(:,7);
u1 = u1.*mask./mask; u2 = u2.*mask./mask; u3 = u3.*mask./mask;

bd_x = []; bd_u1 = []; bd_u2 = []; bd_u3 = [];
tmp_x = zeros(geo.nt,geo.dim);
for i = 1:geo.nt
    tmp_x(i,1) = geo.center(1) + geo.radius * cos(geo.tt(i));
    tmp_x(i,2) = geo.center(2) + geo.radius * sin(geo.tt(i));
    tmp_x(i,3) = geo.p2_za+0.5*mesh.hz;
end
bd_x = [bd_x; tmp_x];

idx = (abs(x3-(geo.p2_za+0.5*mesh.hz))<tol) & mask;
node = [x1(idx),x2(idx),x3(idx)];

coef = rbf_intp_coef(node(:,1:2), u1(idx));
tmp_u1 = rbf_intp(tmp_x(:,1:2), node(:,1:2), coef);
bd_u1 = [bd_u1; tmp_u1];
coef = rbf_intp_coef(node(:,1:2), u2(idx));
tmp_u2 = rbf_intp(tmp_x(:,1:2), node(:,1:2), coef);
bd_u2 = [bd_u2; tmp_u2];
coef = rbf_intp_coef(node(:,1:2), u3(idx));
tmp_u3 = rbf_intp(tmp_x(:,1:2), node(:,1:2), coef);
bd_u3 = [bd_u3; tmp_u3];

% plot3(node(:,1),node(:,2),u2(idx),'k.'); hold on
% plot3(bd_x(:,1),bd_x(:,2),bd_u2,'k.')

tmp_x = zeros(geo.nt,geo.dim);
for i = 1:geo.nt
    tmp_x(i,1) = geo.center(1) + geo.radius * cos(geo.tt(i));
    tmp_x(i,2) = geo.center(2) + geo.radius * sin(geo.tt(i));
    tmp_x(i,3) = geo.p2_zb-0.5*mesh.hz;
end
bd_x = [bd_x; tmp_x];

idx = (abs(x3-(geo.p2_zb-0.5*mesh.hz))<tol) & mask;
node = [x1(idx),x2(idx),x3(idx)];

coef = rbf_intp_coef(node(:,1:2), u1(idx));
tmp_u1 = rbf_intp(tmp_x(:,1:2), node(:,1:2), coef);
bd_u1 = [bd_u1; tmp_u1];
coef = rbf_intp_coef(node(:,1:2), u2(idx));
tmp_u2 = rbf_intp(tmp_x(:,1:2), node(:,1:2), coef);
bd_u2 = [bd_u2; tmp_u2];
coef = rbf_intp_coef(node(:,1:2), u3(idx));
tmp_u3 = rbf_intp(tmp_x(:,1:2), node(:,1:2), coef);
bd_u3 = [bd_u3; tmp_u3];

bd_x1 = bd_x(:,1); bd_x2 = bd_x(:,2); bd_x3 = bd_x(:,3);
x1 = [x1; bd_x1]; x2 = [x2; bd_x2]; x3 = [x3; bd_x3];
u1 = [u1; bd_u1]; u2 = [u2; bd_u2]; u3 = [u3; bd_u3];

%%
for i = 1:geo.nt
    idx = (abs(x1-geo.radius*cos(geo.tt(i)))<tol & abs(x2-geo.radius*sin(geo.tt(i)))<tol) | ...
          (abs(x1-geo.radius*cos(geo.tt(i+1)))<tol & abs(x2-geo.radius*sin(geo.tt(i+1)))<tol);
    tmp_x1 = x1(idx,:); tmp_x2 = x2(idx,:); tmp_x3 = x3(idx,:);
    tmp_u1 = u1(idx,:); tmp_u2 = u2(idx,:); tmp_u3 = u3(idx,:);
    tmp_idx = delaunay(tmp_x1,tmp_x3);
    trisurf(tmp_idx, tmp_x1, tmp_x2, tmp_x3, tmp_u1); hold on
end

idx = abs(x1-(geo.p1_xa+0.5*mesh.hx))<tol;
tmp_x1 = x1(idx,:); tmp_x2 = x2(idx,:); tmp_x3 = x3(idx,:);
tmp_u1 = u1(idx,:); tmp_u2 = u2(idx,:); tmp_u3 = u3(idx,:);
tmp_idx = delaunay(tmp_x2,tmp_x3);
trisurf(tmp_idx, tmp_x1, tmp_x2, tmp_x3, tmp_u1); hold on

idx = abs(x1-(geo.p1_xb-0.5*mesh.hx))<tol;
tmp_x1 = x1(idx,:); tmp_x2 = x2(idx,:); tmp_x3 = x3(idx,:);
tmp_u1 = u1(idx,:); tmp_u2 = u2(idx,:); tmp_u3 = u3(idx,:);
tmp_idx = delaunay(tmp_x2,tmp_x3);
trisurf(tmp_idx, tmp_x1, tmp_x2, tmp_x3, tmp_u1); hold on

idx = abs(x1-(geo.p2_xb-0.5*mesh.hx))<tol;
tmp_x1 = x1(idx,:); tmp_x2 = x2(idx,:); tmp_x3 = x3(idx,:);
tmp_u1 = u1(idx,:); tmp_u2 = u2(idx,:); tmp_u3 = u3(idx,:);
tmp_idx = delaunay(tmp_x2,tmp_x3);
trisurf(tmp_idx, tmp_x1, tmp_x2, tmp_x3, tmp_u1); hold on

idx = abs(x2-(geo.p2_ya+0.5*mesh.hy))<tol;
tmp_x1 = x1(idx,:); tmp_x2 = x2(idx,:); tmp_x3 = x3(idx,:);
tmp_u1 = u1(idx,:); tmp_u2 = u2(idx,:); tmp_u3 = u3(idx,:);
tmp_idx = delaunay(tmp_x1,tmp_x3);
trisurf(tmp_idx, tmp_x1, tmp_x2, tmp_x3, tmp_u1); hold on

idx = abs(x2-(geo.p2_yb-0.5*mesh.hy))<tol;
tmp_x1 = x1(idx,:); tmp_x2 = x2(idx,:); tmp_x3 = x3(idx,:);
tmp_u1 = u1(idx,:); tmp_u2 = u2(idx,:); tmp_u3 = u3(idx,:);
tmp_idx = delaunay(tmp_x1,tmp_x3);
trisurf(tmp_idx, tmp_x1, tmp_x2, tmp_x3, tmp_u1); hold on

idx = abs(x3-(geo.p2_za+0.5*mesh.hz))<tol;
tmp_x1 = x1(idx,:); tmp_x2 = x2(idx,:); tmp_x3 = x3(idx,:);
tmp_u1 = u1(idx,:); tmp_u2 = u2(idx,:); tmp_u3 = u3(idx,:);
tmp_idx = delaunay(tmp_x1,tmp_x2);
trisurf(tmp_idx, tmp_x1, tmp_x2, tmp_x3, tmp_u1); hold on

idx = abs(x3-(geo.p2_zb-0.5*mesh.hz))<tol;
tmp_x1 = x1(idx,:); tmp_x2 = x2(idx,:); tmp_x3 = x3(idx,:);
tmp_u1 = u1(idx,:); tmp_u2 = u2(idx,:); tmp_u3 = u3(idx,:);
tmp_idx = delaunay(tmp_x1,tmp_x2);
trisurf(tmp_idx, tmp_x1, tmp_x2, tmp_x3, tmp_u1); hold on

idx = abs(x3-(geo.p1_za+0.5*mesh.hz))<tol;
tmp_x1 = x1(idx,:); tmp_x2 = x2(idx,:); tmp_x3 = x3(idx,:);
tmp_u1 = u1(idx,:); tmp_u2 = u2(idx,:); tmp_u3 = u3(idx,:);
tmp_idx = delaunay(tmp_x1,tmp_x2);
trisurf(tmp_idx, tmp_x1, tmp_x2, tmp_x3, tmp_u1); hold on

idx = abs(x3-(geo.p1_zb-0.5*mesh.hz))<tol;
tmp_x1 = x1(idx,:); tmp_x2 = x2(idx,:); tmp_x3 = x3(idx,:);
tmp_u1 = u1(idx,:); tmp_u2 = u2(idx,:); tmp_u3 = u3(idx,:);
tmp_idx = delaunay(tmp_x1,tmp_x2);
trisurf(tmp_idx, tmp_x1, tmp_x2, tmp_x3, tmp_u1); hold on

shading interp
colormap('jet')
axis([-1,1, -1,1, -1,1])
view([30,30])
set(gcf,'unit','normalized','position',[0.2,0.2,0.32,0.4])