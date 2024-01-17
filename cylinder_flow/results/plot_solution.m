clear; clc
format short
format compact

%%
geo.dim = 2;
geo.xa = -1.5; geo.xb = 1.5; geo.ya = -0.5; geo.yb = 0.5;
geo.center = [-0.4,-0.1]; geo.radius = 0.2;

mesh.nx = 60; mesh.ny = 20;
mesh.c_size = mesh.nx*mesh.ny;

%%
solution = textread('solution.txt');
c_x1 = solution(:,1); c_x2 = solution(:,2); c_a = solution(:,3);
c_u1 = solution(:,4); c_u2 = solution(:,5); c_p = solution(:,6);
c_mask = solution(:,7);

idx = 1;
mesh.c_x1 = c_x1((idx-1)*mesh.c_size+1:idx*mesh.c_size,:);
mesh.c_x2 = c_x2((idx-1)*mesh.c_size+1:idx*mesh.c_size,:);
mesh.c_u1 = c_u1((idx-1)*mesh.c_size+1:idx*mesh.c_size,:);
mesh.c_u2 = c_u2((idx-1)*mesh.c_size+1:idx*mesh.c_size,:);
mesh.c_p = c_p((idx-1)*mesh.c_size+1:idx*mesh.c_size,:);
mesh.c_mask = c_mask((idx-1)*mesh.c_size+1:idx*mesh.c_size,:);

mesh.c_u1 = mesh.c_u1.* mesh.c_mask./ mesh.c_mask;
mesh.c_u2 = mesh.c_u2.* mesh.c_mask./ mesh.c_mask;
mesh.c_p = mesh.c_p.* mesh.c_mask./ mesh.c_mask;

%%
tha = 0.0; thb = 2*pi;
nth = 100; hth = (thb-tha)/nth;
th = tha:hth:thb;
x1 = [mesh.c_x1; geo.center(1) + geo.radius * cos((th(1:end-1))')];
x2 = [mesh.c_x2; geo.center(2) + geo.radius * sin((th(1:end-1))')];
u1 = [mesh.c_u1; zeros(nth,1)];
u2 = [mesh.c_u2; zeros(nth,1)];

units = delaunay(x1,x2);
trisurf(units,x1,x2,u1)
shading interp
colormap('jet')
axis([geo.xa,geo.xb,geo.ya,geo.yb])
set(gcf,'unit','normalized','position',[0.2,0.2,0.6,0.25])