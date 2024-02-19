function coef = rbf_intp_coef(node, value)

[n_size,dim] = size(node);
c_size = n_size+dim+1;

ee = 1e-2;

a = zeros(c_size,c_size);
for j = 1:n_size
    dist = (sum((node-node(j,:)).^2,2)).^ 0.5;
    phi = (ee + dist.^2).^ (-0.5);
    a(1:n_size,j) = phi;
end
a(1:n_size,n_size+1:c_size-1) = node;
a(n_size+1:c_size-1,1:n_size) = node';
a(1:n_size,c_size) = 1;
a(c_size,1:n_size) = 1;

b = zeros(c_size,1);
b(1:n_size,:) = value;
coef = a\b;