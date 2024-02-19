clear; clc
format short
format compact

%%
geo.dim = 2;
geo.xa = -1.5; geo.xb = 1.5; geo.ya = -0.5; geo.yb = 0.5;
geo.center = [-0.3,0.0]; geo.radius = 0.2;

%%
mesh.xa = geo.xa; mesh.xb = geo.xb; mesh.ya = geo.ya; mesh.yb = geo.yb;
mesh.nx = 90; mesh.ny = 30;
mesh.hx = (mesh.xb-mesh.xa)/mesh.nx; mesh.hy = (mesh.yb-mesh.ya)/mesh.ny;
mesh.c_size = mesh.nx*mesh.ny;

%%
te_set_prd = textread('te_set_pred.txt');
te_set_ref = xlsread('te_set.csv');
mesh.c_x1 = te_set_prd(:,1); mesh.c_x2 = te_set_prd(:,2); mesh.c_a = te_set_prd(:,3);
mesh.c_u1_prd = te_set_prd(:,5); mesh.c_u2_prd = te_set_prd(:,6); mesh.c_p_prd = te_set_prd(:,7);
mesh.c_u1_ref = te_set_ref(:,5); mesh.c_u2_ref = te_set_ref(:,6); mesh.c_p_ref = te_set_ref(:,7);
mesh.c_mask = te_set_prd(:,8);

mesh.c_u1_prd = mesh.c_u1_prd.* mesh.c_mask./ mesh.c_mask;
mesh.c_u2_prd = mesh.c_u2_prd.* mesh.c_mask./ mesh.c_mask;
mesh.c_p_prd = mesh.c_p_prd.* mesh.c_mask./ mesh.c_mask;

mesh.c_u1_ref = mesh.c_u1_ref.* mesh.c_mask./ mesh.c_mask;
mesh.c_u2_ref = mesh.c_u2_ref.* mesh.c_mask./ mesh.c_mask;
mesh.c_p_ref = mesh.c_p_ref.* mesh.c_mask./ mesh.c_mask;

%%
tha = 0.0; thb = 2*pi;
nth = 100; hth = (thb-tha)/nth;
th = tha:hth:thb;
x1_bd = geo.center(1) + geo.radius * cos((th(1:end-1))');
x2_bd = geo.center(2) + geo.radius * sin((th(1:end-1))');
u1_bd_prd = zeros(nth,1); u2_bd_prd = zeros(nth,1);
u1_bd_ref = zeros(nth,1); u2_bd_ref = zeros(nth,1);

idx = (mesh.c_x1>(geo.center(1)-1.5*geo.radius)) & (mesh.c_x1<(geo.center(1)+1.5*geo.radius)) & ...
      (mesh.c_x2>(geo.center(2)-1.5*geo.radius)) & (mesh.c_x2<(geo.center(2)+1.5*geo.radius)) & ...
      (mesh.c_mask==1);
intp_x1 = mesh.c_x1(idx,:); intp_x2 = mesh.c_x2(idx,:);

intp_p = mesh.c_p_prd(idx,:);
intp_coef = rbf_intp_coef([intp_x1,intp_x2], intp_p);
p_bd_prd = rbf_intp([x1_bd,x2_bd], [intp_x1,intp_x2], intp_coef);

intp_p = mesh.c_p_ref(idx,:);
intp_coef = rbf_intp_coef([intp_x1,intp_x2], intp_p);
p_bd_ref = rbf_intp([x1_bd,x2_bd], [intp_x1,intp_x2], intp_coef);

%%
x1 = [mesh.c_x1; x1_bd];
x2 = [mesh.c_x2; x2_bd];

u1_prd = [mesh.c_u1_prd; u1_bd_prd];
u2_prd = [mesh.c_u2_prd; u2_bd_prd];
p_prd = [mesh.c_p_prd; p_bd_prd];

u1_ref = [mesh.c_u1_ref; u1_bd_ref];
u2_ref = [mesh.c_u2_ref; u2_bd_ref];
p_ref = [mesh.c_p_ref; p_bd_ref];

u1_err = abs(u1_prd-u1_ref);
u2_err = abs(u2_prd-u2_ref);
p_err = abs(p_prd-p_ref);

units = delaunay(x1,x2);
trisurf(units,x1,x2,u1_prd)
shading interp
map = [70/256,7/256,90/256; 72/256,22/256,104/256; 72/256,35/256,116/256;
       71/256,47/256,125/256; 68/256,59/256,132/256; 63/256,71/256,136/256; 
       59/256,82/256,139/256; 54/256,92/256,141/256; 49/256,102/256,142/256; 
       45/256,112/256,142/256; 41/256,121/256,142/256; 38/256,130/256,142/256; 
       34/256,140/256,141/256; 31/256,148/256,140/256; 31/256,158/256,137/256; 
       34/256,167/256,133/256; 42/256,176/256,127/256; 56/256,185/256,119/256; 
       74/256,193/256,109/256; 94/256,201/256,98/256; 115/256,208/256,86/256; 
       139/256,214/256,70/256; 165/256,219/256,54/256; 192/256,223/256,27/256; 
       218/256,227/256,25/256; 244/256,230/256,20/256];
colormap(map)
axis([geo.xa,geo.xb,geo.ya,geo.yb])
set(gcf,'unit','normalized','position',[0.2,0.2,0.6,0.25])